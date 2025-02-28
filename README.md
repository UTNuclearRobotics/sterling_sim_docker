# Pre-requisites
Create Python virtual environment and build the module
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Install these ROS packages  
```
ros-${ROS_DISTRO}-cyclonedds
ros-${ROS_DISTRO}-rmw-cyclonedds-cpp
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

ros2 launch husarion_nav2 navigation2_bringup.launch.py use_rviz:=True use_sim_time:=True nav2_config_file_slam:=/root/sterling/config/nav2_params.yaml
```