# Commands to run inside the container
alias run_gazebo_high='ros2 launch sterling_gazebo sidewalks.launch.py high_res:=True'
alias run_gazebo_low='ros2 launch sterling_gazebo sidewalks.launch.py high_res:=False'
alias run_nav2='ros2 launch husarion_nav2 navigation2_bringup.launch.py use_rviz:=True use_sim_time:=True nav2_config_file_slam:=/root/ros2_ws/src/sterling/config/nav2_params.yaml'
alias run_sterling_costmap='source /root/ros2_ws/install/setup.bash && ros2 launch sterling costmaps.launch.py'
alias save_costmap='source /root/ros2_ws/install/setup.bash && ros2 service call /sterling/save_costmap std_srvs/srv/Trigger'