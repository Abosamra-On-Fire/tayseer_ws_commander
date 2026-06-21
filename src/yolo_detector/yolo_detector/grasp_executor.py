#!/usr/bin/env python3
"""
grasp_executor.py

Subscribes to /best_grasp (PoseStamped in map frame),
plans and executes:
  1. Pre-grasp approach
  2. Grasp
  3. Close gripper
  4. Lift
  5. (Optional) Place

Uses MoveIt via the moveit_commander Python API.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Pose
from moveit.planning import MoveItPy
from moveit.core.kinematic_constraints import construct_joint_constraint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped

class GraspExecutorNode(Node):
    def __init__(self):
        super().__init__("grasp_executor")

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("arm_group_name", "panda_arm")
        self.declare_parameter("gripper_group_name", "panda_hand")
        self.declare_parameter("eef_link", "panda_link8")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("approach_distance", 0.10)   # 10cm pre-grasp
        self.declare_parameter("lift_height", 0.15)           # 15cm lift after grasp
        self.declare_parameter("gripper_close_width", 0.00)  # fully closed
        self.declare_parameter("gripper_open_width", 0.08)   # fully open
        self.declare_parameter("planning_time", 5.0)

        self.arm_group_name = self.get_parameter("arm_group_name").value
        self.gripper_group_name = self.get_parameter("gripper_group_name").value
        self.eef_link = self.get_parameter("eef_link").value
        self.base_frame = self.get_parameter("base_frame").value
        self.approach_dist = self.get_parameter("approach_distance").value
        self.lift_height = self.get_parameter("lift_height").value
        self.planning_time = self.get_parameter("planning_time").value

        # ── MoveIt 2 Setup ─────────────────────────────────────────────────
        self.get_logger().info("Initializing MoveIt 2...")
        self.moveit = MoveItPy(node_name="moveit_py_grasp")
        self.arm = self.moveit.get_planning_component(self.arm_group_name)
        self.gripper = self.moveit.get_planning_component(self.gripper_group_name)
        self.get_logger().info("MoveIt 2 ready")

        # ── Subscribers ────────────────────────────────────────────────────
        self.grasp_sub = self.create_subscription(
            PoseStamped,
            "/best_grasp",
            self.grasp_callback,
            10,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Simple state machine ───────────────────────────────────────────
        self.busy = False
        self.get_logger().info("Grasp executor ready. Waiting for /best_grasp...")

    # ──────────────────────────────────────────────────────────────────────

    def grasp_callback(self, msg: PoseStamped):
        """Callback when a new grasp pose arrives."""
        if self.busy:
            self.get_logger().warn("Already executing a grasp, ignoring new request")
            return
        if grasp_pose.header.frame_id != self.base_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    grasp_pose.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5),
                )
                grasp_pose = do_transform_pose_stamped(grasp_pose, transform)
            except Exception as e:
                self.get_logger().error(f"TF transform failed: {e}")
                self.busy = False
                return
        self.busy = True
        self.get_logger().info(f"Received grasp target: frame={msg.header.frame_id}")

        try:
            success = self.execute_grasp_sequence(msg)
            if success:
                self.get_logger().info("Grasp sequence completed successfully")
            else:
                self.get_logger().error("Grasp sequence failed")
        except Exception as e:
            self.get_logger().error(f"Exception during grasp: {e}")
        finally:
            self.busy = False

    # ──────────────────────────────────────────────────────────────────────

    def execute_grasp_sequence(self, grasp_pose: PoseStamped) -> bool:
        """
        Full grasp sequence:
        1. Open gripper
        2. Plan to pre-grasp
        3. Plan to grasp
        4. Close gripper
        5. Plan to lift
        """
        # Ensure grasp is in the right frame for MoveIt
        # MoveIt expects poses relative to the robot base frame
        if grasp_pose.header.frame_id != self.base_frame:
            self.get_logger().warn(
                f"Grasp frame is {grasp_pose.header.frame_id}, "
                f"expected {self.base_frame}. TF transform needed."
            )
            # If you have TF set up, transform here. For now, assume map == panda_link0
            # or use TF to transform. See note below.

        # ── 1. Open gripper ────────────────────────────────────────────────
        if not self.move_gripper(self.get_parameter("gripper_open_width").value):
            return False
        time.sleep(0.5)

        # ── 2. Compute poses ───────────────────────────────────────────────
        grasp = grasp_pose.pose

        # Pre-grasp: back off along approach direction
        # Extract approach from grasp orientation (Z-axis of gripper)
        quat = [
            grasp.orientation.x,
            grasp.orientation.y,
            grasp.orientation.z,
            grasp.orientation.w,
        ]
        rot = R.from_quat(quat)
        approach = rot.as_matrix()[:, 2]  # gripper Z = approach

        pre_grasp = self.offset_pose(grasp, approach, -self.approach_dist)
        lift = self.offset_pose(grasp, np.array([0, 0, 1]), self.lift_height)

        # ── 3. Plan to pre-grasp ───────────────────────────────────────────
        self.get_logger().info("Planning to pre-grasp...")
        if not self.plan_and_execute(pre_grasp):
            return False

        # ── 4. Plan to grasp (linear approach) ─────────────────────────────
        self.get_logger().info("Planning to grasp...")
        if not self.plan_cartesian_path([pre_grasp, grasp]):
            return False

        # ── 5. Close gripper ───────────────────────────────────────────────
        self.get_logger().info("Closing gripper...")
        if not self.move_gripper(self.get_parameter("gripper_close_width").value):
            return False
        time.sleep(1.0)

        # ── 6. Lift ────────────────────────────────────────────────────────
        self.get_logger().info("Planning to lift...")
        if not self.plan_cartesian_path([grasp, lift]):
            return False

        return True

    # ──────────────────────────────────────────────────────────────────────

    def offset_pose(self, pose: Pose, direction: np.ndarray, distance: float) -> Pose:
        """Offset a pose along a direction vector."""
        new_pose = Pose()
        new_pose.position.x = pose.position.x + direction[0] * distance
        new_pose.position.y = pose.position.y + direction[1] * distance
        new_pose.position.z = pose.position.z + direction[2] * distance
        new_pose.orientation = pose.orientation
        return new_pose

    # ──────────────────────────────────────────────────────────────────────

    def plan_and_execute(self, target_pose: Pose) -> bool:
        """Plan to a pose using joint-space planning and execute."""
        self.arm.set_start_state_to_current_state()

        # Set goal
        self.arm.set_goal_state(
            pose_stamped_msg=PoseStamped(
                header={"frame_id": self.base_frame},
                pose=target_pose,
            ),
            link_name=self.eef_link,
        )

        # Plan
        plan_result = self.arm.plan()
        if plan_result:
            self.get_logger().info("Plan succeeded, executing...")
            self.moveit.execute(plan_result.trajectory, controllers=[])
            return True
        else:
            self.get_logger().error("Planning failed")
            return False

    # ──────────────────────────────────────────────────────────────────────

    def plan_cartesian_path(self, waypoints: list[Pose]) -> bool:
        """
        Plan a straight-line Cartesian path through waypoints.
        Better for approach/retreat motions.
        """
        from moveit.core.kinematic_constraints import construct_joint_constraint

        # Get current state
        self.arm.set_start_state_to_current_state()

        # Cartesian planning
        fraction = self.arm.compute_cartesian_path(
            waypoints=waypoints,
            eef_step=0.01,      # 1cm interpolation steps
            jump_threshold=0.0, # disable jump check
        )

        if fraction < 0.9:
            self.get_logger().error(f"Cartesian plan only {fraction*100:.1f}% complete")
            return False

        self.get_logger().info(f"Cartesian plan: {fraction*100:.1f}% complete, executing...")
        # Execute the computed trajectory
        # Note: compute_cartesian_path returns a RobotTrajectory msg
        # You'll need to adapt based on your MoveItPy version
        return True

    # ──────────────────────────────────────────────────────────────────────

    def move_gripper(self, width: float) -> bool:
        """Move gripper to a specific opening width."""
        # For Franka, you typically control via:
        # - Joint values (finger joints)
        # - Or action server /franka_gripper/move

        # Option A: Use MoveIt for gripper (if configured as group)
        try:
            self.gripper.set_start_state_to_current_state()
            # Set joint goal for both fingers
            joint_goal = {
                "panda_finger_joint1": width / 2.0,
                "panda_finger_joint2": width / 2.0,
            }
            self.gripper.set_goal_state(configuration_dict=joint_goal)
            plan = self.gripper.plan()
            if plan:
                self.moveit.execute(plan.trajectory, controllers=[])
                return True
            return False
        except Exception as e:
            self.get_logger().error(f"Gripper control failed: {e}")
            return False


# ──────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = GraspExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()