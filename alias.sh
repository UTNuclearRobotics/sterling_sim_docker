# Commands to run inside the container

# Step 1: Choose the terrain resolution you want ot run the Gazebo world
alias run_gazebo_low='ros2 launch sterling_gazebo sidewalks.launch.py res:=low namespace:=panther'
alias run_gazebo_medium='ros2 launch sterling_gazebo sidewalks.launch.py res:=medium namespace:=panther'
alias run_gazebo_high='ros2 launch sterling_gazebo sidewalks.launch.py res:=high namespace:=panther'

# Step 2: Launch Nav2
# Sterling parameters
#alias run_nav2='ros2 launch husarion_nav2 navigation2_bringup.launch.py use_rviz:=True use_sim_time:=True nav2_config_file_slam:=/root/ros2_ws/src/sterling/config/nav2_params.yaml'
alias run_nav2='ros2 launch utexas_panther bringup_launch.py namespace:=panther observation_topic:=ouster/scan observation_topic_type:=laserscan slam:=true use_sim_time:=true nav2_config_file_slam:=/root/ros2_ws/src/sterling/config/nav2_params.yaml'
# Default parameters, when want to collect rosbag data
alias run_nav2_default='ros2 launch husarion_nav2 navigation2_bringup.launch.py use_rviz:=True use_sim_time:=True'

# Step 3: Sterling costmap node
alias run_sterling_costmap='source /root/ros2_ws/install/setup.bash && ros2 launch sterling costmaps.launch.py'
# While node is running, call service to save the costmap
alias save_costmap='source /root/ros2_ws/install/setup.bash && ros2 service call /sterling/save_costmap std_srvs/srv/Trigger'