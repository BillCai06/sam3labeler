# KISS-ICP Setup & Run Guide

Platform: Ubuntu 20.04, Python 3.10 (conda), ROS1 bag input

---

## 1. Create Conda Environment

```bash
conda create -n kiss python=3.10 -y
conda activate kiss
Python 3.10 is required. polyscope (the visualizer) fails to build on Python 3.13.

2. Install Dependencies

pip install kiss-icp
pip install polyscope
pip install rosbags==0.9.12
rosbags==0.9.12 specifically — newer versions removed typesys.types and break the ROS1→ROS2 converter.

3. Convert ROS1 Bag → ROS2 Bag
KISS-ICP no longer supports ROS1 bags directly. Convert first:


rosbags-convert --src /path/to/your.bag --dst /path/to/output_ros2/
This produces a ROS2 bag directory (folder with .db3 + metadata.yaml).

4. Fix Visualizer Crash (polyscope API mismatch)
The installed visualizer.py crashes when you click in the 3D window due to a PickResult API change in newer polyscope. Patch it:

File:
~/miniconda3/envs/kiss/lib/python3.10/site-packages/kiss_icp/tools/visualizer.py

Find _trajectory_pick_callback and replace with:


def _trajectory_pick_callback(self):
    if self._gui.GetIO().MouseClicked[0]:
        try:
            result = self._ps.get_selection()
            if hasattr(result, 'name'):
                name, idx = result.name, result.index
            else:
                name, idx = tuple(result)
            if name == "trajectory" and self._ps.has_point_cloud(name):
                pose = self._trajectory[idx]
                self._selected_pose = f"x: {pose[0]:7.3f}, y: {pose[1]:7.3f}, z: {pose[2]:7.3f}>"
            else:
                self._selected_pose = ""
        except Exception:
            self._selected_pose = ""
5. Run KISS-ICP

conda activate kiss

kiss_icp_pipeline \
  --topic /velodyne_points \
  --visualize \
  /path/to/output_ros2/
--topic must match the LiDAR topic name in your bag
--visualize opens the 3D polyscope window
Omit --visualize to run headless (faster, no window needed)
Other useful flags:


--max-range 200.0       # filter points beyond N meters
--min-range 1.0         # filter points closer than N meters
--n-scans 500           # process only first N scans (for testing)
6. Output
Results are saved to ./results/<timestamp>/ with a latest/ symlink:

File	Contents
<name>_poses.npy	Raw 4×4 pose matrices (numpy)
<name>_poses_kitti.txt	KITTI format: 12 values/row (3×4 matrix)
<name>_poses_tum.txt	TUM format: timestamp x y z qx qy qz qw
config.yml	Config used for this run
result_metrics.log	FPS, runtime stats

ls ./results/latest/
Note: The point cloud map is NOT saved automatically. Only poses are saved.

7. Visualizer Controls
Key	Action
Space	Start / Pause
N	Next frame (when paused)
G	Toggle Global / Local view
C	Center viewpoint
S	Screenshot
Q / Esc	Quit
In Global View, the full trajectory is shown as a point cloud. Clicking a trajectory point displays its XYZ coordinates.


