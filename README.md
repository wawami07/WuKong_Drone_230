# WuKong_Drone_230: 面向复杂环境的智能无人机自主导航系统框架

**WuKong_Drone_230** 是一个高性能的自主无人机解决方案，在著名的 **[Fast-Drone-250](https://github.com/ZJU-FAST-Lab/Fast-Drone-250)** 架构基础上改进。本项目通过集成前沿的感知、定位与规划算法，实现了在复杂动态场景下的全自主、高机动飞行。

# 🧠 核心架构与技术栈

- **🛰️ 高精度状态估计**: 采用 **[FAST_LIO_LOCALIZATION_PLUS](https://github.com/iDonghq/FAST_LIO_LOCALIZATION_PLUS)** 系列算法，融合激光雷达与惯性测量单元，实现毫秒级、厘米精度的实时定位与建图。
- **🧭 实时运动规划**: 搭载 **[EGO-Planner-v2](https://github.com/ZJU-FAST-Lab/EGO-Planner-v2)** 规划器，能够在三维复杂环境中进行高速、无碰撞的轨迹生成，并具备动态重规划能力。
- **👁️ 动态障碍物感知**: 创新性地引入基于 **[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)** 框架的PointPillars模型，通过我们自建的专用数据集进行训练，可实时检测与追踪场景中的移动障碍物，为规划器提供环境动态信息。


### 运行环境
Nvidia Jetson Orin + Ubuntu20.04 + CUDA 11.4 + cuDNN 8.6.0 + TensorRT 8.5.2.2

运行FAST_LIO前需要安装livox_ros_driver2驱动，请借鉴官网或网络上的教程

运行PointPillars_ros包前需要安装一个openpcdet的conda环境，教程https://zhuanlan.zhihu.com/p/657200184 

### 克隆项目
```
# 新建终端
git clone https://github.com/wawami07/WuKong_iros2025.git
cd WuKong_iros2025/
conda activate openpcdet
cd src/identify_pointcloud/pointpillars_ros
# 安装PointPillars_ros包需要的依赖
pip install -r requirements.txt
python setup.py develop
cd ../../../
catkin_make
source devel/setup.bash
```

### 运行FAST_LIO重定位代码：
```
（可以参考FAST_LIO_LOCALIZATION链接中的说明）
# 需要将pcd点云地图放入/PCD文件下
# 新建终端
source devel/setup.bash
roslaunch livxo_ros_driver2 msg_MID360.launch
# 新建终端
source devel/setup.bash
roslaunch fast_lio mapping_mid360.launch
# 等待rviz开启后再新建终端继续执行：
source devel/setup.bash
roslaunch fast_lio localization_mid360.launch
# 此时rviz中等待一会儿后会加载出先前放入/PCD的地图，然后rviz中选择2D Nav Goal，把起始点选中即可开始匹配
```

### 运行 EGO-Planner-v2 规划代码：
```
# 新建终端
source devel/setup.bash
roslaunch ego_planner run_in_exp.launch 
# 新建终端
source devel/setup.bash
roslaunch ego_planner rviz.launch 
# 在 "run_in_exp.launch" 中 "flight_type" ：
#     1: use 3D Nav Goal to select goal 
#     2: use global waypoints below 
```

### 运行动态障碍物避障代码：
```
# 新建终端
source devel/setup.bash
rosrun moving_obstacles moving_obstacles_iros
# 新建终端
source devel/setup.bash
conda activate openpcdet
roslaunch pointpillars_ros tracker.launch
```



### 动态障碍物的识别测试说明

WuKong_Drone_230/src/identify_pointcloud/pointpillars_ros/tools/ros.py 里的best_1600_e80_s20.pth权重文件和测试的bag都在百度网盘的链接 https://pan.baidu.com/s/1ZhPZ_iUAoAfdQIvbAvTt7g?pwd=v92f 里，提取码: v92f

把best_1600_e80_s20.pth放到WuKong_Drone_230/src/identify_pointcloud/pointpillars_ros/models里

```
# 新建终端
conda activate opempcdet
source devel/setup.bash
roslaunch pointpillars_ros pointpillars.launch
# 新建终端
rosbag play dongtai.bag
```


