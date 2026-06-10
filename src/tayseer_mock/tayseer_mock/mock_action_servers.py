#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from tayseer_interfaces.action import NavigateToObject, PickObject, PlaceObject, SlideObject
from geometry_msgs.msg import Point
import time

class MockActionServers(Node):
    def __init__(self):
        super().__init__('mock_action_servers')
        
        self._nav_server = ActionServer(
            self, NavigateToObject, '/navigate_to_object', self.execute_navigate)
        self._pick_server = ActionServer(
            self, PickObject, '/pick_object', self.execute_pick)
        self._place_server = ActionServer(
            self, PlaceObject, '/place_object', self.execute_place)
        self._slide_server = ActionServer(
            self, SlideObject, '/slide_object', self.execute_slide)
            
        self.get_logger().info("  All mock action servers ready!")

    def execute_navigate(self, goal_handle):
        obj = goal_handle.request.object_name
        self.get_logger().info(f'  [NAVIGATE] Moving to {obj}...')
        
        feedback = NavigateToObject.Feedback()
        for i in range(5, 0, -1):
            feedback.distance_remaining = float(i)
            feedback.current_state = 'moving'
            goal_handle.publish_feedback(feedback)
            time.sleep(0.8)
        
        goal_handle.succeed()
        result = NavigateToObject.Result()
        result.success = True
        result.message = f"Arrived at {obj}"
        self.get_logger().info(f'  [NAVIGATE] Done')
        return result

    def execute_pick(self, goal_handle):
        obj = goal_handle.request.object_name
        self.get_logger().info(f'  [PICK] Picking {obj}...')
        
        feedback = PickObject.Feedback()
        for stage, prog in [("approaching", 0.3), ("grasping", 0.6), ("lifting", 0.9)]:
            feedback.stage = stage
            feedback.progress = prog
            goal_handle.publish_feedback(feedback)
            time.sleep(1.0)
        
        goal_handle.succeed()
        result = PickObject.Result()
        result.success = True
        result.message = f"Picked {obj}"
        self.get_logger().info(f'  [PICK] Done')
        return result

    def execute_place(self, goal_handle):
        obj = goal_handle.request.object_name
        loc = goal_handle.request.target_location_name or "target"
        self.get_logger().info(f'  [PLACE] Placing {obj} at {loc}...')
        
        feedback = PlaceObject.Feedback()
        for stage, prog in [("approaching", 0.3), ("placing", 0.6), ("releasing", 0.9)]:
            feedback.stage = stage
            feedback.progress = prog
            goal_handle.publish_feedback(feedback)
            time.sleep(1.0)
        
        goal_handle.succeed()
        result = PlaceObject.Result()
        result.success = True
        result.message = f"Placed {obj} at {loc}"
        self.get_logger().info(f'  [PLACE] Done')
        return result

    def execute_slide(self, goal_handle):
        obj = goal_handle.request.object_name
        direction = goal_handle.request.direction
        dist = goal_handle.request.distance
        self.get_logger().info(f' ️ [SLIDE] Sliding {obj} {direction} by {dist}m...')
        
        feedback = SlideObject.Feedback()
        feedback.stage = "sliding"
        feedback.progress = 0.5
        goal_handle.publish_feedback(feedback)
        time.sleep(2.0)
        
        goal_handle.succeed()
        result = SlideObject.Result()
        result.success = True
        result.message = f"Slid {obj} {direction} by {dist}m"
        # Return updated position
        pos = goal_handle.request.object_position
        result.final_position = [pos.x + 0.1, pos.y, pos.z]
        self.get_logger().info(f'  [SLIDE] Done')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = MockActionServers()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()