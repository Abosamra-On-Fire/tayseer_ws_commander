from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='detector_node',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': 'yolo26x.pt',
                'image_path': '/home/ubuntu/tayseer_ws/src/yolo_detector/yolo_detector/image.jpg',                    # <-- place image.jpg in ~/tayseer_ws/ or use full path
                'output_image_topic': '/yolo/annotated_image',
                'output_detections_topic': '/yolo/detections',
                'conf_threshold': 0.5,
                'device': 'auto',
                'publish_rate_hz': 1.0,
                'use_sim_time': True,
            }]
        ),
        # Node(
        #     package="yolo_detector",
        #     executable="grasp_executor",
        #     name="grasp_executor",
        #     parameters=[{
        #         "arm_group_name": "panda_arm",
        #         "gripper_group_name": "panda_hand",
        #         "approach_distance": 0.10,
        #         "lift_height": 0.15,
        #     }],
        # ),
    ])