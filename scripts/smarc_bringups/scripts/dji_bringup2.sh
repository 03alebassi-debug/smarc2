#! /bin/bash
#
# dji_bringup2.sh - a trimmed-down dji_bringup.sh for hook-pendulum /
# sway_controller work (hook Kalman filter, length+damping identification,
# ground-truth comparison).
#
# Same arguments and conventions as dji_bringup.sh, but it only launches what
# that workflow actually needs. This matters beyond tidiness: the hook filter
# has a documented problem where _prediction() ticks at ~0.3s instead of the
# requested 0.02s because the machine is saturated, so every node that isn't
# needed is actively harmful here.
#
# LAUNCHED:
#   - ros_tcp_endpoint (sim only)      - Unity bridge, publishes the TF tree
#   - dji_captain (+ services)         - base_flat_link TF, odom, cmd_vel
#   - alars_move_to_action_server      - used to excite the swing
#   - alars_move_to_dumped_action_server - the sway-damped mission (ZVD + LQG)
#   - YOLO detector                    - the hook detections the filter eats
#   - gimbal action server             - and a ready-made "point down" command
#   - estimate_length_and_damping_node - action server the filter calls at boot
#   - hook_ground_truth_comparator_node- republishes Unity GT into base_flat_link
#   - sway_plotter_node                - the ONLY node that plots; records the
#                                        estimator and controller topics and
#                                        writes every PNG on Ctrl+C
#
# NOT launched (present in dji_bringup.sh): behaviour trees, WASP/wara-ps,
# mqtt bridge + mosquitto, rosboard, geofence / log / wait / internet checker,
# NAU driver, succorfish modem, search / follow_auv / recover action servers,
# and the SAM+buoy projection EKF (auv_buoy_ekf_launch).
# Dropping the geofence node is safe: dji_captain guards its geofence checks
# with `if self._geofence_status is not None`.
#
# NOT started, but pre-typed into a pane for you to press Enter on:
#   - hook_kalman_filter_node  (Hook window, bottom pane)
#   - gimbal "point straight down"  (Gimbal window, bottom pane)
#
# Order of operations for a damped mission:
#   1. Captain up, drone flying, gimbal pointed down (Gimbal window).
#   2. Start hook_kalman_filter_node (Hook window). It runs the identification
#      itself on startup - which EXCITES a swing - and publishes L/xi on the
#      latched <robot>/hook_pendulum_params topic.
#   3. Send a move_to_dumped goal. Run open loop first with
#      ENABLE_LQG=False ./dji_bringup2.sh M350 5.0
#      It reads that topic, stabilises the payload first, then flies the
#      shaped+damped trajectory.
#   4. For plots: Ctrl+C the sway_plotter_node pane (Hook window) once the
#      mission is over. It writes <PLOT_OUTPUT_DIR>/estimator/*.png and
#      <PLOT_OUTPUT_DIR>/control/*.png, default /home/aleba/sway_plots.
#      `ros2 service call /<robot>/save_sway_plots std_srvs/srv/Trigger` does
#      the same without stopping the recording.
#
#   ros2 action send_goal /<robot>/move_to_dumped smarc_msgs/action/BaseAction \
#     '{goal: {data: "{\"waypoint\": {\"latitude\": 59.30651, \"longitude\": 18.70958, \"altitude\": 5.25, \"tolerance\": 0.5}}"}}' --feedback
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tmux_layout.sh"

# Type a command into a pane WITHOUT running it (no C-m), so it sits on the
# prompt waiting for you to press Enter. Called at the very end of this script,
# not inline: each pane runs a fresh shell whose rc file (conda init etc) takes
# a moment, and text sent before the prompt is drawn gets painted twice.
preload_pane() {
    local target="$1"
    local cmd="$2"
    # `clear` first: a slow rc file (conda init) can mean the shell is not in
    # readline yet, so the raw characters get echoed by the tty and then drawn
    # a second time when the prompt finally appears. Running clear guarantees a
    # settled prompt on a blank screen before the command is typed onto it.
    tmux send-keys -t "$target" "clear" C-m
    sleep 0.3
    tmux send-keys -t "$target" -- "$cmd"
}


ROBOT_NAME=$1
if [[ -z "$ROBOT_NAME" ]]; then
    echo "You must pass the robot name as the first argument! Pass one of: M350 or FC30"
    echo "This is required to namespace all the ROS2 nodes and topics correctly."
    echo "As well as to set parameters depending on the platform..."
    echo "Exiting."
    exit 1
fi

if [[ "$ROBOT_NAME" != "M350" && "$ROBOT_NAME" != "FC30" ]]; then
    echo "Invalid robot name: $ROBOT_NAME"
    echo "Please pass either M350 or FC30 as the first argument."
    echo "Exiting."
    exit 1
fi

HOME_ABOVE_WATER=$2
if [[ -z "$HOME_ABOVE_WATER" ]]; then
    echo "You must pass the home altitude above water level as the second argument!"
    echo "This is required for the dji_captain node to function properly."
    echo "Exiting."
    exit 1
fi

if ! [[ "$HOME_ABOVE_WATER" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "HOME_ABOVE_WATER must be a floating point number! Adding a decimal point for you..."
    HOME_ABOVE_WATER="${HOME_ABOVE_WATER}.0"
    echo "HOME_ABOVE_WATER is set to $HOME_ABOVE_WATER"
fi

NO_CAM=$3
if [[ "$NO_CAM" == "no_cam" ]]; then
    echo "Camera will be disabled."
    echo "NOTE: the hook filter is driven entirely by YOLO hook detections, so"
    echo "      with no camera there is nothing to estimate from. This is only"
    echo "      useful for checking the rest of the stack comes up."
    NO_CAM=True
else
    NO_CAM=False
fi

FAKE_IT_TILL_YOU_MAKE_IT=$4
if [[ "$FAKE_IT_TILL_YOU_MAKE_IT" == "fake_it" ]]; then
    FAKE_IT_TILL_YOU_MAKE_IT=True
else
    FAKE_IT_TILL_YOU_MAKE_IT=False
fi


# Deliberately a different session name from dji_bringup.sh's ${ROBOT_NAME}_bringup,
# so this can be checked/killed independently and the two never collide.
SESSION=${ROBOT_NAME}_bringup2

if tmux has-session -t $SESSION 2>/dev/null; then
    echo "There is already a tmux session named $SESSION."
    echo "Please close it before launching this script."
    echo "Exiting."
    exit 1
fi

if tmux has-session -t "${ROBOT_NAME}_bringup" 2>/dev/null; then
    echo "WARNING: the full bringup session '${ROBOT_NAME}_bringup' is also running."
    echo "         Running both will double up every node (two captains, two YOLOs, ...)."
    echo "         Kill it with: tmux kill-session -t ${ROBOT_NAME}_bringup"
    echo ""
fi


    USE_SIM_TIME=True

########
# PARAMS
########
if [[ $ROBOT_NAME == "M350" ]]; then
    MAX_LOAD_KG="7.0"
    MIN_ALTITUDE_ABOVE_WATER="1.5"
elif [[ $ROBOT_NAME == "FC30" ]]; then
    MAX_LOAD_KG="30.0"
    MIN_ALTITUDE_ABOVE_WATER="3.0"
fi
HOOK_LINE_LENGTH=10.0

# Real Z1 Pro hardware and the Unity sim have OPPOSITE pitch sign conventions on
# gimbal_cmd: on the real rig negative = down, in the sim +90 empirically points
# down. Don't "fix" one from the other.
if [[ $USE_SIM_TIME = "True" ]]; then
    GIMBAL_DOWN_PITCH="90.0"
else
    GIMBAL_DOWN_PITCH="-90.0"
fi

# create a tmux session with a name
tmux -2 new-session -d -x 220 -y 60 -s "$SESSION"


############
# 1 Sim connection (must be up first - it is what publishes the TF tree)
############
if [[ $USE_SIM_TIME = "True" ]]; then
    ROS_TCP_ENDPOINT_CMD="ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p tcp_ip:=localhost -p tcp_port:=10000"
    tmux_make_layout "$SESSION" SimConnection "row(var(ROS_TCP_ENDPOINT_CMD))"
fi


############
# 2 Captain
############
CAPTAIN_CMD="ros2 launch dji_captain alars_captain.launch \
    robot_name:=$ROBOT_NAME \
    use_sim_time:=$USE_SIM_TIME \
    home_altitude_above_water:=$HOME_ABOVE_WATER \
    max_load_kg:=$MAX_LOAD_KG \
    min_altitude_above_water:=$MIN_ALTITUDE_ABOVE_WATER \
    rope_length:=$HOOK_LINE_LENGTH"

CAPTAIN_STATUS_CMD="ros2 topic echo /$ROBOT_NAME/captain_status std_msgs/msg/String --field data"
WRAPPER_CMD="ros2 launch psdk_wrapper wrapper.launch.py namespace:=/$ROBOT_NAME/wrapper"
DISCOVERY_SERVER_CMD="export ZENOH_CONFIG_OVERRIDE='listen/endpoints=[\"tcp/0.0.0.0:7447\"]' && ros2 run rmw_zenoh_cpp rmw_zenohd"
SERVICE_CALLER_CMD="ros2 run dji_captain service_caller --ros-args -r __ns:=/$ROBOT_NAME -p use_sim_time:=$USE_SIM_TIME -p robot_name:=$ROBOT_NAME"
ALARS_SERVICES_CMD="ros2 launch dji_captain alars_services.launch.py robot_name:=$ROBOT_NAME use_sim_time:=$USE_SIM_TIME"

if [[ $FAKE_IT_TILL_YOU_MAKE_IT == "True" ]]; then
    WRAPPER_CMD="ros2 run dji_captain psdk_faker --ros-args -r __ns:=/$ROBOT_NAME -p robot_name:=$ROBOT_NAME -p use_sim_time:=$USE_SIM_TIME"
fi

if [[ $USE_SIM_TIME = "False" ]]; then
    tmux_make_layout "$SESSION" Captain "
    col(
        1:row(
            1:var(DISCOVERY_SERVER_CMD),
            3:var(WRAPPER_CMD)
        ),
        3:row(
            2:var(CAPTAIN_CMD),
            3:var(CAPTAIN_STATUS_CMD),
            2:col(
                var(SERVICE_CALLER_CMD),
                var(ALARS_SERVICES_CMD)
            )
        )
    )"
else
    tmux_make_layout "$SESSION" Captain "
    row(
        2:var(CAPTAIN_CMD),
        3:var(CAPTAIN_STATUS_CMD),
        2:col(
            var(SERVICE_CALLER_CMD),
            var(ALARS_SERVICES_CMD)
        )
    )"
fi


############
# 3 The two move actions
#   - move_to        : plain, unshaped. Used to EXCITE the swing for testing.
#   - move_to_dumped : the sway-damped mission (ZVD feedforward + LQG trim).
#     It reads the identified L/xi off the latched
#     /$ROBOT_NAME/hook_pendulum_params topic when a goal arrives, so
#     hook_kalman_filter_node (Hook window) must have run its identification
#     first - otherwise it waits 30s and REJECTS the goal rather than guessing.
############
ALARS_MOVE_TO_CMD="ros2 run alars alars_move_to_action_server --ros-args -r __ns:=/$ROBOT_NAME \
-p robot_name:=$ROBOT_NAME \
-p use_sim_time:=$USE_SIM_TIME"

# ENABLE_LQG=False flies the ZVD feedforward OPEN LOOP - the known-good
# baseline, and the A/B test for whether the feedback helps. The first
# closed-loop flight (2026-07-26) diverged, so start here when in doubt:
#   ENABLE_LQG=False ./dji_bringup2.sh M350 5.0
ENABLE_LQG=${ENABLE_LQG:-True}
ALARS_MOVE_TO_DUMPED_CMD="ros2 launch alars alars_move_to_dumped_server_launch.py \
robot_name:=$ROBOT_NAME \
use_sim_time:=$USE_SIM_TIME \
enable_lqg:=$ENABLE_LQG"

# Bottom pane is left empty and pre-typed below with the "where am I" helper,
# since a goal for either action needs a lat/lon near the current position.
tmux_make_layout "$SESSION" MoveTo "
col(
    2:row(
        var(ALARS_MOVE_TO_CMD),
        var(ALARS_MOVE_TO_DUMPED_CMD)
    ),
    1:pane
)"

# The bottom pane is pre-typed with the "where am I" helper; the goal template
# itself is in this script's header comment, because its JSON escaping does not
# survive being pushed through send-keys into a shell.
LATLON_CMD="ros2 topic echo /$ROBOT_NAME/smarc/latlon --once"


############
# 4 Camera and hook detection
############
# Two independent detector stacks run here, on the same camera images:
#
#   alars (YOLO_CMD)   -> alars_detection/labeled_obbs, auv_obb, buoy_obb,
#                         auv_head, cam_processor_happy   [normalized coords]
#   yolo_ros (YOLO_ROS_CMD) -> yolo/detections            [pixel coords]
#                           -> yolo/detections_with_corners (corners adapter)
#
# Both are kept because only the alars one publishes the auv/buoy topics that
# auv_state_estimation's projection nodes and the alars action servers consume.
# Intrinsics the hook filter STARTS from, until CameraInfo arrives and overrides
# them. Set outside the NO_CAM branch on purpose: the filter is launched either
# way, so it always needs a file to read. Lives in auv_state_estimation/config.
# NOTE: the sim camera is 1280x720 as of the Aug-2026 SMARCAssets update, so the
# 720p file is right for BOTH now - sim_1080p_cam_params.yaml no longer matches
# what Unity publishes. Only the startup values are at stake (CameraInfo
# overrides fx/fy/cx/cy once it arrives), but starting from the wrong image size
# skews the first measurements.
CAM_CALIBRATION_FILE="z1_720p_cam_params.yaml"

if [[ "$NO_CAM" == "True" ]]; then
    YOLO_CMD="echo 'Camera disabled, not launching YOLO detector - no hook detections will exist'"
    YOLO_ROS_CMD="echo 'Camera disabled, not launching yolo_ros - yolo/detections will not exist'"
else
    YOLO_DEVICE=0
    YOLO_MODEL="yolo_model_2cls_may.pt"
    if [[ $USE_SIM_TIME = "True" ]]; then
        YOLO_DEVICE=cpu
        # 4-class model: sam, buoy, hook, land_pad - the only one with 'hook'
        YOLO_MODEL="yolo_model_4cls_hook_sim.pt"
    fi
    # PYTHONNOUSERSITE=1 keeps this node's dedicated conda env (ros_yolo) from
    # silently picking up packages installed in ~/.local/lib/python3.10/site-packages.
    YOLO_CMD="PYTHONNOUSERSITE=1 ros2 launch alars_auv_perception alars_yolo_detector.launch.py \
    robot_name:=$ROBOT_NAME \
    device:=$YOLO_DEVICE \
    use_sim_time:=$USE_SIM_TIME \
    model_package:=alars_labeling_training \
    model_file:=$YOLO_MODEL"

    # yolo_ros wants ultralytics-style device names ('cuda:0'/'cpu'), not the
    # bare index the alars detector takes.
    YOLO_ROS_DEVICE="cuda:0"
    if [[ $USE_SIM_TIME = "True" ]]; then
        YOLO_ROS_DEVICE="cpu"
    fi
    # alars_yolo_corners.launch.py defaults this to 0.5, which is STRICTER than
    # the 0.4 the alars detector uses (detection_parameters.yaml: 'hook' is not
    # in per_class_confidence, so it falls back to confidence_threshold). Set
    # low while checking whether the hook is detected at all; once it is, raise
    # to 0.4 so both stacks agree on what counts as a detection.
    #   YOLO_ROS_THRESHOLD=0.4 ./dji_bringup2.sh ...   overrides without editing
    YOLO_ROS_THRESHOLD=${YOLO_ROS_THRESHOLD:-0.25}

    # Inference resolution. yolo_node ALWAYS passes imgsz to ultralytics, unlike
    # the alars detector which passes none and so runs at the model's own size.
    # MUST match the camera feed: rescaling changes the apparent pixel size of
    # the hook and the model then misses it entirely - upscaling 720p to 1088p
    # cost every detection and dropped the rate from 16Hz to 1.5Hz. Check with
    #   ros2 topic echo /$ROBOT_NAME/gimbal_camera/camera/camera_info --once
    # and round the height UP to a multiple of 32 (720 -> 736, 1080 -> 1088).
    YOLO_ROS_IMGSZ_H=${YOLO_ROS_IMGSZ_H:-736}
    YOLO_ROS_IMGSZ_W=${YOLO_ROS_IMGSZ_W:-1280}

    # Publishes /$ROBOT_NAME/yolo/detections (yolo_msgs/DetectionArray, pixel
    # coordinates) and, through yolo_corners_adapter, .../yolo/detections_with_corners.
    #
    # REQUIRES yolo_ros to have been built from inside the ros_yolo conda env,
    # the same way alars_auv_perception was - see HANDOFF.md. As installed by a
    # plain `colcon build`, yolo_node's shebang is /usr/bin/python3, which has
    # no ultralytics, and the node dies on import before publishing anything.
    # The corners adapter's shebang is /usr/bin/env python3, so the pane must
    # NOT be inside a conda env: base is python 3.13 and rclpy's C extension is
    # built for 3.10. `conda config --set auto_activate_base false` fixes it.
    YOLO_ROS_CMD="PYTHONNOUSERSITE=1 ros2 launch yolo_smarc_actions alars_yolo_corners.launch.py \
    robot_name:=$ROBOT_NAME \
    use_sim_time:=$USE_SIM_TIME \
    model_package:=alars_labeling_training \
    model_subdir:=trained_models \
    model_file:=$YOLO_MODEL \
    device:=$YOLO_ROS_DEVICE \
    threshold:=$YOLO_ROS_THRESHOLD \
    imgsz_height:=$YOLO_ROS_IMGSZ_H \
    imgsz_width:=$YOLO_ROS_IMGSZ_W"
fi

tmux_make_layout "$SESSION" Perception "
col(
    var(YOLO_CMD),
    var(YOLO_ROS_CMD)
)"


############
# 5 Gimbal
############
# The measurement model treats the hook's pixel offset as a pendulum angle about
# straight-down, so the camera MUST be pointed down or the filter drops every
# detection (HookKalmanFilter guards on max_boresight_tilt_deg). The sim comes up
# with the gimbal HORIZONTAL, so this is not optional - hence the ready-to-run
# command in the bottom pane.
if [[ "$NO_CAM" == "True" ]]; then
    GIMBAL_CAM_VIDEO_CMD="echo 'Camera disabled, not launching gscam node'"
    GIMBAL_CAM_DRIVER_CMD="echo 'Camera disabled, not launching gimbal driver node'"
    GIMBAL_CMD_ACTION_CMD="echo 'Camera disabled, not launching gimbal action server node'"
else
    GIMBAL_IP=192.168.1.108
    GIMBAL_PORT=2332
    GSCAM_CONFIG_GIMBAL="rtspsrc location=rtsp://$GIMBAL_IP latency=0 ! \
    rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! \
    video/x-raw,format=BGRx ! \
    videoconvert ! queue max-size-buffers=1 leaky=downstream"
    GIMBAL_CAM_TOPIC_NS=gimbal_camera
    GIMBAL_CAM_VIDEO_CMD="ros2 run gscam gscam_node --ros-args \
        -p gscam_config:=\"$GSCAM_CONFIG_GIMBAL\" \
        -p frame_id:=z1_optical_frame \
        -p image_encoding:=rgb8 \
        -p sync_sink:=false \
        -p camera.image_raw.enable_pub_plugins:="['image_transport/raw']" \
        -r __ns:=/$ROBOT_NAME/$GIMBAL_CAM_TOPIC_NS"
    GIMBAL_CAM_DRIVER_CMD="ros2 launch z1_pro_driver z1_pro_driver_launch.py \
        robot_name:=$ROBOT_NAME \
        camera_ip:=$GIMBAL_IP \
        camera_port:=$GIMBAL_PORT \
        camera_below_base:=True"
    GIMBAL_CMD_ACTION_CMD="ros2 launch z1_pro_driver z1_pro_action_launch.py \
        robot_name:=\"$ROBOT_NAME\" \
        use_sim_time:=$USE_SIM_TIME"
fi

if [[ $USE_SIM_TIME = "False" ]]; then
    # On the real rig the camera stream and the gimbal pose publisher both come
    # from the driver; in sim Unity provides them.
    tmux_make_layout "$SESSION" Gimbal "
    col(
        1:row(
            var(GIMBAL_CAM_VIDEO_CMD),
            var(GIMBAL_CAM_DRIVER_CMD),
            var(GIMBAL_CMD_ACTION_CMD)
        ),
        1:pane
    )"
else
    tmux_make_layout "$SESSION" Gimbal "
    col(
        2:var(GIMBAL_CMD_ACTION_CMD),
        1:pane
    )"
fi

GIMBAL_DOWN_CMD="ros2 topic pub -r 2 -t 10 /$ROBOT_NAME/gimbal_camera/gimbal_cmd geometry_msgs/msg/Vector3 \"{x: 0.0, y: $GIMBAL_DOWN_PITCH, z: 0.0}\""


############
# 6 Hook - sysid server + ground truth + plotter, filter left for you to start
############
# All sway_controller nodes are LAUNCHED (not `ros2 run`) so they land in the
# /$ROBOT_NAME namespace: their topic and action names are plain relative names,
# the same convention as the alars action servers.
if [[ "$NO_CAM" == "True" ]]; then
    ESTIMATE_LENGTH_AND_DAMPING_CMD="echo 'Camera disabled, not launching estimate_length_and_damping_node'"
else
    # This is an action SERVER - it must already be running when you start the
    # filter, because hook_kalman_filter_node calls it (blocking) at startup to
    # get L/xi unless you override both.
    ESTIMATE_LENGTH_AND_DAMPING_CMD="ros2 launch sway_controller estimate_length_and_damping_node_launch.py \
robot_name:=$ROBOT_NAME \
use_sim_time:=$USE_SIM_TIME"
fi

# Needs Unity's GT_TransformOdom_Pub attached to the hook GameObject to have
# anything to republish.
HOOK_GT_COMPARATOR_CMD="ros2 launch sway_controller hook_ground_truth_comparator_node_launch.py \
robot_name:=$ROBOT_NAME \
use_sim_time:=$USE_SIM_TIME"

# The ONLY node that plots. It is a pure subscriber, so it can sit here for the
# whole session: start it before the mission, then Ctrl+C this pane (or call
# `ros2 service call /$ROBOT_NAME/save_sway_plots std_srvs/srv/Trigger`) to
# write everything it recorded to $PLOT_OUTPUT_DIR.
PLOT_OUTPUT_DIR=${PLOT_OUTPUT_DIR:-/home/aleba/sway_plots}
SWAY_PLOTTER_CMD="ros2 launch sway_controller sway_plotter_node_launch.py \
robot_name:=$ROBOT_NAME \
use_sim_time:=$USE_SIM_TIME \
plot_output_dir:=$PLOT_OUTPUT_DIR"

tmux_make_layout "$SESSION" Hook "
col(
    1:var(ESTIMATE_LENGTH_AND_DAMPING_CMD),
    1:var(HOOK_GT_COMPARATOR_CMD),
    1:var(SWAY_PLOTTER_CMD),
    1:pane
)"

# Pre-typed, NOT executed - press Enter here once the drone is flying and the
# gimbal is pointed down. It runs the identification itself, which EXCITES a
# swing.
HOOK_KF_CMD="ros2 launch sway_controller hook_kalman_filter_node_launch.py \
robot_name:=$ROBOT_NAME \
use_sim_time:=$USE_SIM_TIME \
camera_calibration_file:=$CAM_CALIBRATION_FILE"


# All panes exist by now; give their shells a moment to finish drawing prompts,
# then type (but do not run) the commands you drive the test with by hand.
sleep 2
preload_pane "$SESSION:MoveTo.{bottom-right}"  "$LATLON_CMD"
preload_pane "$SESSION:Gimbal.{bottom-right}"  "$GIMBAL_DOWN_CMD"
preload_pane "$SESSION:Hook.{bottom-right}"    "$HOOK_KF_CMD"

tmux -2 attach-session -t "$SESSION"
tmux set-option -t "$SESSION" mouse on
tmux select-window -t "$SESSION:Captain"
