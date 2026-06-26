#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tayseer_interfaces.srv import GetObjectLocation, ListObjects
from tayseer_interfaces.action import ArmManipulation, SlideObject
from tayseer_interfaces.action import Navigate
from tayseer_commander.llm_client import LLMClient
import json
import threading
import time


class CommanderNode(Node):
    def __init__(self):
        super().__init__('commander')

        self.declare_parameter('api_key', '') 
        self.declare_parameter('max_replan_atmts', 2)

        api_key = self.get_parameter('api_key').value
        self.llm = LLMClient(api_key)

        #srv clients
        self.get_location_cli = self.create_client(GetObjectLocation, '/get_object_location')
        self.list_objects_cli = self.create_client(ListObjects, '/list_objects')
        self.get_logger().info("Waiting for World Model services...")
        self.get_location_cli.wait_for_service(timeout_sec=10.0)
        self.list_objects_cli.wait_for_service(timeout_sec=10.0)

        #act clients
        self.navigate_client = ActionClient(self, Navigate, '/navigate_to_goal')
        self.arm_client = ActionClient(self, ArmManipulation, '/arm_manipulate')
        self.slide_client = ActionClient(self, SlideObject, '/slide_object')

        self.get_logger().info("Waiting for action servers...")
        if not self.navigate_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("A* Navigator /navigate_to_goal not available yet — will retry on first use")
        if self.arm_client is not None and self.arm_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().info("Action server connected! Ready to send goals.")
        else:
            self.get_logger().error("Action client not initialized or server timed out.")

        self.slide_client.wait_for_server(timeout_sec=10.0)

        self.status_pub = self.create_publisher(String, '/commander_status', 10)
        self.plan_pub = self.create_publisher(String, '/commander_plan', 10)
        self.chat_pub = self.create_publisher(String, '/chat_message', 10)

        self.create_subscription(String, '/user_prompt', self.msg_cb, 10)
        self.create_service(Trigger, '/execute_prompt', self.manual_exe_cb)

        self.state = "IDLE"
        self.conv_h = [] #converstion history
        self.msg_q = [] #queue for msgs
        self.replan_atmts = 0 #to count hte number of replans
        self.max_replans = self.get_parameter('max_replan_atmts').value

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info("  Commander ready. Waiting for prompts...")

    def publish_status(self, status, detail= ""):
        msg = String()
        msg.data = json.dumps({
            "status": status,
            "detail": detail,
            "timestamp": self.get_clock().now().to_msg().sec
        })
        self.status_pub.publish(msg)

    def publish_chat(self, role, content, options = None):
        msg = String()
        msg.data = json.dumps({
            "role": role,
            "content": content,
            "options": options or [],
            "timestamp": self.get_clock().now().to_msg().sec
        })
        self.chat_pub.publish(msg)
    def publish_plan(self, plan_res):
        msg = String()
        msg.data = json.dumps(plan_res)
        self.plan_pub.publish(msg)

    def get_world_state(self):
        future = self.list_objects_cli.call_async(ListObjects.Request())
        start = time.time()
        while not future.done() and time.time() - start < 3.0:
            time.sleep(0.01)
        if not future.done():
            self.get_logger().error("World Model list timeout")
            return {}

        res = future.result()
        world_state = {}
        for obj_name in res.object_names:
            loc_future = self.get_location_cli.call_async(
                GetObjectLocation.Request(object_name=obj_name))
            start = time.time()
            while not loc_future.done() and time.time() - start < 3.0:
                time.sleep(0.01)
            if loc_future.done() and loc_future.result().found:
                loc = loc_future.result()
                world_state[obj_name] = {
                    "position": [loc.position.x, loc.position.y, loc.position.z],
                    "frame_id": loc.frame_id
                }
        return world_state

    def _reset_to_idle(self):
        self.state = "IDLE"
        self.conv_h = []
        self.replan_atmts = 0

    def msg_cb(self, msg):
        if self.state == "EXECUTING":
            self.get_logger().warn("Currently executing, message ignored")
            self.publish_chat("assistant", "I'm currently busy executing a plan. Please wait.")
            return
        self.msg_q.append(msg.data)
        self.get_logger().info(f"Queued message: {msg.data}")

    def _worker_loop(self):
        while rclpy.ok():
            try:
                if self.msg_q and self.state in ("IDLE", "CLARIFYING"):
                    user_msg = self.msg_q.pop(0)
                    self.conv_h.append({"role": "user", "content": user_msg})
                    self._process_conversation()
            except Exception as e:
                self.get_logger().error(f"Worker loop error (recovering): {e}", throttle_duration_sec=5)
                self._reset_to_idle()
            time.sleep(0.1)

    def _process_conversation(self):
        self.publish_status("thinking", "Tayseer is thinking...")

        world_state = self.get_world_state()
        res = self.llm.generate_response(self.conv_h, world_state)

        if isinstance(res, str):
            try:
                res = json.loads(res)
            except json.JSONDecodeError:
                self.get_logger().error(f"LLM returned non-JSON string: {res[:200]}")
                self.publish_chat("assistant", "I received an unexpected res format. Please try again.")
                self._reset_to_idle()
                return

        if not isinstance(res, dict):
            self.get_logger().error(f"LLM returned unexpected type: {type(res)}")
            self.publish_chat("assistant", "Internal error: invalid res type from language model.")
            self._reset_to_idle()
            return

        if res.get("error"):
            self.publish_chat("assistant", f"Sorry, I encountered an error: {res.get('reasoning', 'Unknown')}")
            self._reset_to_idle()
            return

        mode = res.get("mode", "clarify")

        if mode == "clarify":
            self.state = "CLARIFYING"
            question = res.get("question", "I need more information.")
            options = res.get("options", [])
            self.publish_chat("assistant", question, options)
            self.conv_h.append({"role": "assistant", "content": question})
            self.publish_status("clarifying", question)
        elif mode == "denied":
            reason = res.get("reason", res.get("reasoning", "I cannot perform that action."))
            self.publish_chat("assistant", f"Action denied: {reason}")
            self.publish_status("denied", reason)
            self._reset_to_idle()
            return    

        elif mode == "plan":
            self.state = "EXECUTING"
            self.replan_atmts = 0
            reasoning = res.get("reasoning", "Executing plan...")
            self.publish_chat("assistant", reasoning)
            self.conv_h.append({"role": "assistant", "content": reasoning})

            plan_msg = String()
            plan_msg.data = json.dumps(res)
            self.plan_pub.publish(plan_msg)

            self._execute_plan_blocking(res)
        else:
            self.publish_chat("assistant", "I didn't understand. Can you rephrase?")
            self._reset_to_idle()


    def _execute_plan_blocking(self, plan_res):
        plan = plan_res.get('plan', [])
        if not plan:
            self.publish_chat("assistant", "I couldn't generate a valid plan.")
            self._reset_to_idle()
            return

        self.publish_status("executing", f"Executing {len(plan)} actions")

        i = 0
        while i < len(plan):
            step = plan[i]
            action_type = step.get('action')
            params = step.get('params')

            if action_type and not isinstance(params, dict):
                params = {k: v for k, v in step.items() if k != 'action'}
                if params:
                    self.get_logger().warn(
                        f"Step {i+1}: params were flat, normalized from: {step}"
                    )

            if not action_type or not isinstance(params, dict):
                self.get_logger().error(
                    f"Step {i+1} is malformed (missing 'action' or 'params'): {step}"
                )
                result = {
                    "success": False,
                    "message": f"Malformed plan step: {step}"
                }
            else:
                self.get_logger().info(f"Step {i+1}/{len(plan)}: {action_type}")
                self.publish_status("executing", f"Step {i+1}: {action_type}")
                try:
                    if action_type == 'navigate_to':
                        result =  self._do_navigate(params)
                    elif action_type == 'pick':
                        result =  self._do_pick(params)
                    elif action_type == 'place':
                        result =  self._do_place(params)
                    elif action_type == 'slide':
                        result =  self._do_slide(params)
                    else:
                        result =  {"success": False, "message": f"Unknown action: {action_type}"}
                except KeyError as e:
                    result =  {"success": False, "message": f"Missing required parameter for '{action_type}': {e}"}
                except Exception as e:
                    result =  {"success": False, "message": f"Unexpected error in '{action_type}': {e}"}


            if result['success']:
                self.get_logger().info(f"  Step {i+1} done: {result['message']}")
                i += 1
            else:
                self.get_logger().error(f"  Step {i+1} failed: {result['message']}")

                if self.replan_atmts < self.max_replans:
                    self.replan_atmts += 1
                    self.conv_h.append({
                        "role": "user",
                        "content": f"The action failed: {json.dumps(step)}. Error: {result['message']}. Please replan."
                    })
                    self.publish_chat("assistant", f"That didn't work ({result['message']}). Let me try a different approach...")

                    new_res = self.llm.generate_response(self.conv_h, self.get_world_state())

                    # FIX: Guard against non-dict ress during replan
                    if not isinstance(new_res, dict):
                        self.get_logger().error(f"Replan returned non-dict: {type(new_res)}")
                        self.publish_chat("assistant", "Internal error during replanning. Please try again.")
                        self._reset_to_idle()
                        return

                    new_mode = new_res.get("mode")
                    if new_mode == "plan":
                        plan = new_res['plan']
                        i = 0
                        self.publish_plan(new_res)
                        self.publish_chat("assistant", new_res.get('reasoning', 'Replanning...'))
                        continue
                    elif new_mode == "denied":
                        reason = new_res.get("reason", new_res.get("reasoning", "Cannot complete action."))
                        self.publish_chat("assistant", f"Action denied during replan: {reason}")
                        self.publish_status("denied", reason)
                        self._reset_to_idle()
                        return
                    else:
                        self._handle_clarify_res(new_res)
                        return
                else:
                    self.publish_chat("assistant", f"I failed after {self.max_replans} attempts. Last error: {result['message']}")
                    self._reset_to_idle()
                    return

        self.publish_chat("assistant", "DONE!")
        self.publish_status("completed", "All actions executed successfully")
        self._reset_to_idle()

    def _handle_clarify_res(self, res):
        self.state = "CLARIFYING"
        question = res.get("question", "I need more information.")
        options = res.get("options", [])
        self.publish_chat("assistant", question, options)
        self.conv_h.append({"role": "assistant", "content": question})
        self.publish_status("clarifying", question)

    def _do_navigate(self, params):
        obj_name = params['object_name']
        obj_data = self.get_world_state().get(obj_name, {})
        position = obj_data.get('position')

        if not position:
            return {"success": False, "message": f"'{obj_name}' not found in world model"}

        # Wait for server if it wasn't ready at startup
        if not self.navigate_client.wait_for_server(timeout_sec=5.0):
            return {"success": False, "message": "A* Navigator /navigate_to_goal unavailable"}

        # Build the goal — your A* navigator uses simple x, y floats
        goal = Navigate.Goal()
        goal.x = float(position[0])
        goal.y = float(position[1])

        self.get_logger().info(f"[NAVIGATE] Sending goal → {obj_name} "
                            f"({position[0]:.2f}, {position[1]:.2f})")

        future = self.navigate_client.send_goal_async(goal)
        start = time.time()
        while not future.done() and time.time() - start < 5.0:
            time.sleep(0.01)
        if not future.done():
            return {"success": False, "message": "Navigation goal send timeout"}

        goal_handle = future.result()
        if not goal_handle.accepted:
            return {"success": False, "message": "Navigation goal rejected"}

        # Wait for result
        result_future = goal_handle.get_result_async()
        start = time.time()
        while not result_future.done() and time.time() - start < 120.0:
            time.sleep(0.1)

        if not result_future.done():
            goal_handle.cancel_goal_async()
            return {"success": False, "message": "Navigation timed out (2 min)"}

        # Your A* navigator returns: Result(success=bool, message=string)
        result = result_future.result().result
        return {"success": result.success, "message": result.message}

    def _do_pick(self, params):
        self.get_logger().info("Called Pick Action")
        obj_name = params['object_name']
        obj_data = self.get_world_state().get(obj_name, {})
        position = obj_data.get('position', [0, 0, 0])
        grasp_direction = "Top"

        goal = ArmManipulation.Goal()
        goal.object_name = obj_name
        goal.object_position = [float(position[0]), float(position[1]), float(position[2])]
        goal.orientation = [-80.0, 10.0, -90.0]
        goal.grasp_direction = grasp_direction

        future = self.arm_client.send_goal_async(goal)
        self.get_logger().info("Sent pick action")
        while not future.done():
            time.sleep(0.01)
        goal_handle = future.result()
        if not goal_handle.accepted:
            return {"success": False, "message": "Pick rejected"}

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)
        result = result_future.result().result
        return {"success": result.success, "message": result.message}

    def _do_place(self, params):
        obj_name = params['object_name']
        target = params.get('target_location', '')
        target_data = self.get_world_state().get(target, {})
        position = target_data.get('position', [0, 0, 0])

        goal = ArmManipulation.Goal()
        goal.object_name = target
        goal.object_position = [float(position[0]), float(position[1]), float(position[2])]
        goal.orientation = [-80.0, 10.0, -90.0]

        future = self.arm_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.01)
        goal_handle = future.result()
        if not goal_handle.accepted:
            return {"success": False, "message": "Place rejected"}

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)
        result = result_future.result().result
        return {"success": result.success, "message": result.message}

    def _do_slide(self, params):
        obj_name = params['object_name']
        direction = params['direction']
        distance = params['distance_meters']
        obj_data = self.get_world_state().get(obj_name, {})
        position = obj_data.get('position', [0, 0, 0])

        goal = SlideObject.Goal()
        goal.object_name = obj_name
        goal.object_position = [float(position[0]), float(position[1]), float(position[2])]
        goal.direction = direction
        goal.distance = distance

        future = self.slide_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.01)
        goal_handle = future.result()
        if not goal_handle.accepted:
            return {"success": False, "message": "Slide rejected"}

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)
        result = result_future.result().result
        return {"success": result.success, "message": result.message}


    def manual_exe_cb(self, req, res):
        test_prompt = "Move the blue cube to the shelf"
        self.msg_q.append(test_prompt)
        res.success = True
        res.message = f"Triggered: {test_prompt}"
        return res


def main(args=None):
    rclpy.init(args=args)
    node = CommanderNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()