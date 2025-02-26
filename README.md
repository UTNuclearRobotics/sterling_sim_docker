# Pre-requisites
Install these ROS packages  
```
ros-${ROS_DISTRO}-cyclonedds
ros-${ROS_DISTRO}-rmw-cyclonedds-cpp
```
```bash 
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/path/to/this/repos/cyclonedds.xml
```

# Bash Commands
```
source bash_utils
sterling_build
sterling_start
sterling_shell
```

# Recording Data
```
record_bag_sim.sh
```

# Running Nav
```
run_gazebo_high
run_gazebo_low
run_nav2
ros2 launch sterling_gazebo sidewalks.launch.py high_res:=True
ros2 launch husarion_nav2 navigation2_bringup.launch.py use_rviz:=True use_sim_time:=True
ros2 launch husarion_nav2 navigation2_bringup.launch.py use_rviz:=True use_sim_time:=True nav2_config_file_slam:=src/nav2_sterling_costmap_plugin_py/nav2_slam_params.yaml
```