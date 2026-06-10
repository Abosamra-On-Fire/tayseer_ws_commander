from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('Groq_api_key', default_value=''),
        
        # World Model Node
        Node(
            package='tayseer_commander',
            executable='world_model',
            name='world_model',
            output='screen'
        ),
        
        # Commander Node
        Node(
            package='tayseer_commander',
            executable='commander',
            name='commander',
            output='screen',
            parameters=[{
                'Groq_api_key': LaunchConfiguration('Groq_api_key'),
                'enable_replanning': True,
                'max_replan_attempts': 2
            }]
        ),
    ])