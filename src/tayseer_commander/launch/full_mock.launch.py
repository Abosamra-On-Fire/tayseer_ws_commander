from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # Optional: override .env via CLI if needed
        DeclareLaunchArgument(
            'Groq_api_key',
            default_value='',
            description='Override GROQ_API_KEY (optional, .env is preferred)'
        ),

        # ── Mock Perception ──
        Node(
            package='tayseer_mock',
            executable='mock_perception',
            name='mock_perception',
            output='screen'
        ),

        # ── Mock Action Servers ──
        Node(
            package='tayseer_mock',
            executable='mock_action_servers',
            name='mock_action_servers',
            output='screen'
        ),

        # ── World Model ──
        Node(
            package='tayseer_commander',
            executable='world_model',
            name='world_model',
            output='screen'
        ),

        # ── Commander ──
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

        # ── Web Dashboard ──
        Node(
            package='tayseer_web',
            executable='web_server',
            name='web_server',
            output='screen'
        ),
    ])