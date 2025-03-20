# Commands to run inside the container
alias run_gazebo_high='ros2 launch sterling_gazebo sidewalks.launch.py high_res:=True namespace:=panther'
alias run_gazebo_low='ros2 launch sterling_gazebo sidewalks.launch.py high_res:=False namespace:=panther'
alias run_nav2='ros2 launch utexas_panther bringup_launch.py namespace:=panther observation_topic:=ouster/scan observation_topic_type:=laserscan slam:=true use_sim_time:=true'
alias run_sterling_costmap='source /root/ros2_ws/install/setup.bash && ros2 launch sterling costmaps.launch.py'
alias save_costmap='source /root/ros2_ws/install/setup.bash && ros2 service call /sterling/save_costmap std_srvs/srv/Trigger'