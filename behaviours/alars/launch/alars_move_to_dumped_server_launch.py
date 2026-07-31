import os

from ament_index_python.packages import get_package_share_directory
from launch.substitutions        import PythonExpression, PathJoinSubstitution, LaunchConfiguration
from launch                      import LaunchDescription
from launch.actions              import DeclareLaunchArgument
from launch_ros.actions          import Node

def generate_launch_description():
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='M350',
        description='Namespace for the robot'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation clock instead of wall clock'
    )

    # Exposed so the LQG can be turned off from the bringup without editing
    # files: enable_lqg:=False flies the ZVD feedforward open loop, which is the
    # A/B baseline for whether the feedback is helping or hurting.
    enable_lqg_arg = DeclareLaunchArgument(
        'enable_lqg',
        default_value='True',
        description='False = ZVD feedforward only, no LQG feedback trim'
    )
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='1.0',
        description='Speed limit [m/s]. Also sets the LQR control weight (1/v_max^2)'
    )

    robot_name = LaunchConfiguration('robot_name')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_lqg = LaunchConfiguration('enable_lqg')
    max_speed = LaunchConfiguration('max_speed')

    config_file_name = PythonExpression([
        "'alars_move_to_dumped_server_config_M350.yaml' if '", robot_name, "' == 'M350' else 'alars_move_to_dumped_server_config_FC30.yaml'"
    ])
    config_dir = os.path.join(get_package_share_directory('alars'), 'config')

    config_file = PathJoinSubstitution([config_dir, config_file_name])

    # The model paths in the config yaml are relative to the source tree and only
    # resolve by accident depending on cwd. Override them here with paths resolved
    # against dji_captain's installed share directory, which is stable regardless
    # of where the node is launched from.
    dji_captain_model_dir = PathJoinSubstitution([
        get_package_share_directory('dji_captain'), 'models', robot_name
    ])
    continuous_model_path = PathJoinSubstitution([
        dji_captain_model_dir, 'continuous_model', 'model_bla_diag_cmdvle.npz'
    ])
    discrete_model_path = PathJoinSubstitution([
        dji_captain_model_dir, 'discrete_model', 'discrete_models.npz'
    ])

    node = Node(
        package='alars',
        executable='alars_move_to_dumped_action_server',
        name='alars_move_to_dumped_server',
        namespace=robot_name,
        parameters=[
            config_file,
            {
                'robot_name': robot_name,
                'use_sim_time': use_sim_time,
                'continuous_model_path': continuous_model_path,
                'discrete_model_path': discrete_model_path,
                # After config_file in this list, so these win over the yaml.
                'enable_lqg': enable_lqg,
                'max_speed': max_speed,
            }
        ]
    )

    return LaunchDescription([
        robot_name_arg,
        use_sim_time_arg,
        enable_lqg_arg,
        max_speed_arg,
        node
    ])