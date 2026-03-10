# Robot Dog Runtime Code

This folder snapshots the robot-side code used in our current pipeline.

## Scripts
- `unitree_robot_tcp_streamer.py`
  - Robot-side TCP JPEG streamer (used for Robot -> HPC bridge mode).
- `unitree_robot_ros2_publisher.py`
  - Robot-side native ROS2 publisher (used for Robot -> ROS2 mode).

## Which one is running?
From this repo alone we cannot query live robot processes. In your recent working setup, the robot path was:
- `python3 ~/Workspace/gio_ws/unitree_robot_tcp_streamer.py --interface eth0 --bind 127.0.0.1 --port 5001 --fps 2.0`

If you want to verify on robot:
```bash
ps -ef | grep -E "unitree_robot_tcp_streamer|unitree_robot_ros2_publisher" | grep -v grep
```

## Canonical location note
These files are copied from repo root to keep robot-specific code grouped in one place.
If you update one copy, update the other as well (or migrate imports to make this folder canonical).
