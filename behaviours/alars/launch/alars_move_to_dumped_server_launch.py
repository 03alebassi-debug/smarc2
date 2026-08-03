import os

from ament_index_python.packages import get_package_share_directory
from launch.substitutions        import PythonExpression, PathJoinSubstitution, LaunchConfiguration
from launch                      import LaunchDescription
from launch.actions              import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions          import Node


def _as_bool(text: str) -> bool:
    return text.strip().lower() in ('true', '1', 'yes', 'on')


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
    # Everything below defaults to '' (empty) on purpose: an argument only
    # becomes a ROS parameter if it was actually typed, so anything left alone
    # falls through to the config yaml. A real default here would always beat
    # the yaml and it would stop being the fallback.
    enable_lqg_arg = DeclareLaunchArgument(
        'enable_lqg',
        default_value='',
        description='False = ZVD feedforward only, no LQG feedback trim (unset -> yaml)'
    )
    goal_tolerance_arg = DeclareLaunchArgument(
        'goal_tolerance',
        default_value='',
        description='Distance to the goal that counts as arrived, m (unset -> yaml)'
    )
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='',
        description='Speed limit, m/s. Also sets the LQR control weight 1/v_max^2 (unset -> yaml)'
    )
    max_acceleration_arg = DeclareLaunchArgument(
        'max_acceleration',
        default_value='',
        description='Acceleration limit used to time the path, m/s^2 (unset -> yaml)'
    )
    sattle_extra_arg = DeclareLaunchArgument(
        'sattle_extra',
        default_value='',
        description='Extra time allowed after the plan ends, s (unset -> yaml)'
    )
    lqg_rho_arg = DeclareLaunchArgument(
        'lqg_rho',
        default_value='',
        description='Mission control penalty. Enters the gain as ~sqrt(rho), so move it '
                    'by decades. Too low and the trim saturates (unset -> yaml)'
    )
    lqg_theta_max_arg = DeclareLaunchArgument(
        'lqg_theta_max',
        default_value='',
        description='Swing tolerance for the mission gain, rad, Bryson weight 1/x^2 (unset -> yaml)'
    )
    lqg_position_max_arg = DeclareLaunchArgument(
        'lqg_position_max',
        default_value='',
        description='Position tolerance for the mission gain, m (unset -> yaml)'
    )
    max_estimate_age_arg = DeclareLaunchArgument(
        'max_estimate_age',
        default_value='',
        description='Older than this the swing estimate is unusable -> feedforward only, s (unset -> yaml)'
    )
    max_trim_speed_arg = DeclareLaunchArgument(
        'max_trim_speed',
        default_value='',
        description='Cap on the feedback trim alone, m/s. Must exceed L*omega_peak or the '
                    'loop cannot chase the payload (unset -> yaml)'
    )
    max_theta_for_lqg_arg = DeclareLaunchArgument(
        'max_theta_for_lqg',
        default_value='',
        description='Beyond this the linearised model is invalid and feedback is dropped, rad (unset -> yaml)'
    )
    stabilize_before_mission_arg = DeclareLaunchArgument(
        'stabilize_before_mission',
        default_value='',
        description='Damp the payload before departing (unset -> yaml)'
    )
    stabilize_theta_tol_arg = DeclareLaunchArgument(
        'stabilize_theta_tol',
        default_value='',
        description='Swing considered settled below this, rad (unset -> yaml)'
    )
    stabilize_omega_tol_arg = DeclareLaunchArgument(
        'stabilize_omega_tol',
        default_value='',
        description='Swing rate considered settled below this, rad/s (unset -> yaml)'
    )
    stabilize_settle_time_arg = DeclareLaunchArgument(
        'stabilize_settle_time',
        default_value='',
        description='Time to stay within tolerance before departing, s (unset -> yaml)'
    )
    stabilize_timeout_arg = DeclareLaunchArgument(
        'stabilize_timeout',
        default_value='',
        description='Give up stabilising and depart anyway after this, s (unset -> yaml)'
    )
    stabilize_rho_arg = DeclareLaunchArgument(
        'stabilize_rho',
        default_value='',
        description='Control penalty while stabilising. Lower than lqg_rho because here '
                    'the command IS the whole control action (unset -> yaml)'
    )
    stabilize_theta_max_arg = DeclareLaunchArgument(
        'stabilize_theta_max',
        default_value='',
        description='Swing tolerance for the stabilising gain, rad (unset -> yaml)'
    )
    stabilize_position_max_arg = DeclareLaunchArgument(
        'stabilize_position_max',
        default_value='',
        description='Position tolerance while stabilising, m - loose on purpose (unset -> yaml)'
    )

    robot_name = LaunchConfiguration('robot_name')
    use_sim_time = LaunchConfiguration('use_sim_time')

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

    def make_node(context, *args, **kwargs):
        # A LaunchConfiguration can only be read inside a context, which is why
        # this lives in an OpaqueFunction: an argument becomes a ROS parameter
        # only if it was actually typed, otherwise the yaml value stands.
        overrides = {}
        for name, cast in (('enable_lqg', _as_bool),
                           ('goal_tolerance', float),
                           ('max_speed', float),
                           ('max_acceleration', float),
                           ('sattle_extra', float),
                           ('lqg_rho', float),
                           ('lqg_theta_max', float),
                           ('lqg_position_max', float),
                           ('max_estimate_age', float),
                           ('max_trim_speed', float),
                           ('max_theta_for_lqg', float),
                           ('stabilize_before_mission', _as_bool),
                           ('stabilize_theta_tol', float),
                           ('stabilize_omega_tol', float),
                           ('stabilize_settle_time', float),
                           ('stabilize_timeout', float),
                           ('stabilize_rho', float),
                           ('stabilize_theta_max', float),
                           ('stabilize_position_max', float)):
            value = LaunchConfiguration(name).perform(context)
            if value != '':
                overrides[name] = cast(value)

        node = Node(
            package='alars',
            executable='alars_move_to_dumped_action_server',
            name='alars_move_to_dumped_server',
            namespace=robot_name,
            output='screen',
            # Later entries win: yaml < robot_name/use_sim_time/model paths
            #                        < command line
            parameters=[
                config_file,
                {
                    'robot_name': robot_name,
                    'use_sim_time': use_sim_time,
                    'continuous_model_path': continuous_model_path,
                    'discrete_model_path': discrete_model_path,
                },
                overrides
            ]
        )
        return [node]

    return LaunchDescription([
        robot_name_arg,
        use_sim_time_arg,
        enable_lqg_arg,
        goal_tolerance_arg,
        max_speed_arg,
        max_acceleration_arg,
        sattle_extra_arg,
        lqg_rho_arg,
        lqg_theta_max_arg,
        lqg_position_max_arg,
        max_estimate_age_arg,
        max_trim_speed_arg,
        max_theta_for_lqg_arg,
        stabilize_before_mission_arg,
        stabilize_theta_tol_arg,
        stabilize_omega_tol_arg,
        stabilize_settle_time_arg,
        stabilize_timeout_arg,
        stabilize_rho_arg,
        stabilize_theta_max_arg,
        stabilize_position_max_arg,
        OpaqueFunction(function=make_node)
    ])
