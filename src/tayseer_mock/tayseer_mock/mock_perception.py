#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import random

class MockPerception(Node):
    def __init__(self):
        super().__init__('mock_perception')
        self.pub = self.create_publisher(PoseStamped, '/detected_object', 10)
        
        # Fake objects with slight jitter to simulate real detection noise
        self.objects = {
            "blue_cube_0":      {"x": 1.0, "y": 2.0, "z": 0.0},
            "blue_cube_1":      {"x": 4.0, "y": 2.0, "z": 0.0},
            "blue_cube_2":      {"x": 2.0, "y": 1.0, "z": 0.0},
            "red_ball_0":     {"x": 2.5, "y": 2.2, "z": 0.0},
            "red_ball_1":     {"x": 1.4, "y": 0.5, "z": 0.0},
            "green_ball_0":     {"x": 6.3, "y": 7.2, "z": 0.0},
            "green_cylinder_0": {"x": 2.0, "y": 3.5, "z": 0.0},
            "shelf_0":          {"x": 5.0, "y": 5.0, "z": 0.0},
            "table_0":          {"x": 1.5, "y": 2.5, "z": 0.0},
            "yellow_mug_0":   {"x": 4.2, "y": 1.8, "z": 2.0},
            # "purple_cone":      {"x": 2.8, "y": 4.1, "z": 0.0},
            # "orange_box":       {"x": 6.0, "y": 3.2, "z": 0.0},
            # "pink_torus":       {"x": 3.1, "y": 2.9, "z": 0.0},
            # "cyan_capsule":     {"x": 1.7, "y": 4.5, "z": 0.0},
            # "brown_prism":      {"x": 5.5, "y": 2.1, "z": 0.0},
            # "white_dodecahedron": {"x": 4.8, "y": 4.3, "z": 0.0},
            # "black_octahedron": {"x": 2.3, "y": 1.5, "z": 0.0},
            # "gray_icosahedron": {"x": 6.2, "y": 4.8, "z": 0.0},
            # "magenta_ring":     {"x": 3.9, "y": 3.7, "z": 0.0},
            # "teal_wedge":       {"x": 1.2, "y": 3.8, "z": 0.0},
            # "lime_hemisphere":  {"x": 5.8, "y": 1.5, "z": 0.0},
            # "indigo_star":      {"x": 4.5, "y": 5.2, "z": 0.0},
            # "violet_ellipsoid": {"x": 2.6, "y": 2.3, "z": 0.0},
            # "crimson_tetrahedron": {"x": 6.5, "y": 2.8, "z": 0.0}
        }
        
        self.timer = self.create_timer(3.0, self.publish_objects)
        self.get_logger().info("Mock Perception ready — publishing 5 fake objects")

    def publish_objects(self):
        for name, coords in self.objects.items():
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = name  # Carrying object name in frame_id
            
            # Add tiny noise to simulate detection variance
            msg.pose.position.x = coords["x"] + random.uniform(-0.05, 0.05)
            msg.pose.position.y = coords["y"] + random.uniform(-0.05, 0.05)
            msg.pose.position.z = coords["z"]
            
            self.pub.publish(msg)
            self.get_logger().debug(f"Published {name}")


def main(args=None):
    rclpy.init(args=args)
    node = MockPerception()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()