#include <Eigen/Eigen>
#include <ros/ros.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <sensor_msgs/Joy.h>
#include <nav_msgs/Odometry.h>
#include <traj_utils/MINCOTraj.h>
#include <traj_utils/Obspv.h>  
#include <traj_utils/planning_visualization.h>
#include <optimizer/poly_traj_utils.hpp>

using namespace std;

ros::Publisher obs1_odom_pub_, obs2_odom_pub_, traj_pub_, predicted_traj_pub_;

double obs1_id_, obs2_id_;
// Eigen::Vector2d pos1;
// Eigen::Vector2d pos2;

ego_planner::PlanningVisualization::Ptr visualization_;

class moving_obstacle
{
private:
  Eigen::Vector2d pos_{Eigen::Vector2d::Zero()};  // 2D位置 (x,y)
  Eigen::Vector2d vel_{Eigen::Vector2d::Zero()};  // 2D速度 (vx,vy)
  double yaw_{0};                                 // 朝向角度（弧度）
  ros::Time t_last_update_{ros::Time(0)};         // 上次更新时间

public:

  double des_clearance_;                          // 期望避障安全距离

  moving_obstacle(){};
  ~moving_obstacle(){};

  // 设置初始位置
  void set_position(Eigen::Vector2d pos)
  {
    pos_ = pos;
  }
  void set_velocity(Eigen::Vector2d vel)
  {
    vel_ = vel;
  }

  // 获取当前朝向
  double get_yaw() { return yaw_; }

  // 动力学模型更新（核心函数）
  void dyn_update(const double delta_t, const double acc, const double dir, double &yaw, Eigen::Vector2d &pos, Eigen::Vector2d &vel) const
  {
    // 1. 计算加速度向量（基于当前朝向）
    Eigen::Vector2d acc_vec = acc * Eigen::Vector2d(cos(yaw), sin(yaw));  //实际上是把a分解成x、y轴方向上
    // 2. 更新速度（欧拉积分）
    vel += acc_vec * delta_t;  //v = v + a*Δt
    // 3. 速度衰减（模拟摩擦力）
    vel *= 1; // 阻力使速度衰减
    // 4. 速度限幅（最大2m/s）
    constexpr double MAX_VEL = 1.0;
    if (vel.norm() > MAX_VEL)
    {
      vel /= vel.norm() / MAX_VEL;
    }//限制最大速度不超过1.0，防止速度无限增大
    // 5. 更新位置（带二阶项）
    pos += vel * delta_t + 0.5 * acc_vec * delta_t * delta_t;  //x = x + vΔt + 0.5a*Δt²
    // 6. 更新朝向（yaw角）
    yaw += dir * delta_t;  //θ = θ + ω*Δt
  }

  // 状态更新接口
  std::pair<Eigen::Vector2d, Eigen::Vector2d> update(const double acc, const double dir)
  {
    ros::Time t_now = ros::Time::now();

    // 初始化上次更新时间
    if (t_last_update_ == ros::Time(0))
    {
      t_last_update_ = t_now;
    }

    // 计算时间步长
    double delta_t = (t_now - t_last_update_).toSec();
    // 调用动力学更新
    dyn_update(delta_t, acc, dir, yaw_, pos_, vel_);
    // 更新最后时间戳
    t_last_update_ = t_now;
    // 返回当前位置和速度
    return std::pair<Eigen::Vector2d, Eigen::Vector2d>(pos_, vel_);
  }

  // 状态预测（用于轨迹生成）
  std::pair<Eigen::Vector2d, Eigen::Vector2d> predict(const double acc, const double dir, double predict_t) const
  {
    constexpr double STEP = 0.1;  // 预测步长0.1秒

    // 使用临时变量（避免修改实际状态）
    double yaw = yaw_;
    Eigen::Vector2d pos = pos_;
    Eigen::Vector2d vel = vel_;

    // 逐步预测（固定步长积分）
    for (double t = STEP; t <= predict_t; t += STEP)
    {
      dyn_update(STEP, acc, dir, yaw, pos, vel);
    }

    return std::pair<Eigen::Vector2d, Eigen::Vector2d>(pos, vel);
  }
};

// 创建两个障碍物实例
moving_obstacle obs1_, obs2_;


// 输入起点状态[p,v],a,w，获取预测时间段的预测轨迹
poly_traj::Trajectory predict_traj(const double acc, const double dir, const Eigen::Vector3d p, const Eigen::Vector3d v, const moving_obstacle &obstacle, vector<Eigen::Vector3d> &vis_pts)
{
  vis_pts.clear();  // 清空可视化点
  // 预测参数
  constexpr double PRED_TIME = 5.0;  // 预测5秒轨迹
  constexpr int SEG_NUM = 10;        // 分为10段
  poly_traj::MinJerkOpt predicted_traj;  // 轨迹优化器
  Eigen::Matrix<double, 3, 3> headState, tailState;  // 起点状态[位置, 速度, 加速度]  结束状态
  headState << p, v, Eigen::Vector3d::Zero();  //起点状态赋值[p, v, 0]
  Eigen::MatrixXd innerPts(3, SEG_NUM - 1);  // 存储中间点的矩阵
  Eigen::VectorXd ts(SEG_NUM);  //用于存储​​每段轨迹的时间分配的向量
  vis_pts.push_back(headState.col(0));  //把headState的第一列（位置）可视化

  // 生成中间路径点
  for (int i = 1; i < SEG_NUM; ++i)
  {
    // 预测特定时间点的状态
    auto pred_pv = obstacle.predict(acc, dir, PRED_TIME / SEG_NUM * i);  //(p,v)
    innerPts.col(i - 1) = Eigen::Vector3d(pred_pv.first(0), pred_pv.first(1), p(2));  //把预测的位置x、y值赋值，z值一直不变
    ts(i - 1) = PRED_TIME / SEG_NUM;
    vis_pts.push_back(innerPts.col(i - 1));  // 添加起点到可视化
  }
  ts(SEG_NUM - 1) = PRED_TIME / SEG_NUM;  //补充最后一段时间分配
  // 预测终点状态
  auto tail_pv = obstacle.predict(acc, dir, PRED_TIME);  //(p,v)
  tailState << Eigen::Vector3d(tail_pv.first(0), tail_pv.first(1), p(2)), Eigen::Vector3d(tail_pv.second(0), tail_pv.second(1), v(2)), Eigen::Vector3d::Zero();
  vis_pts.push_back(tailState.col(0));  //添加终点可视化
  predicted_traj.reset(headState, tailState, SEG_NUM);  // 初始化轨迹优化器
  predicted_traj.generate(innerPts, ts);  // 生成最小加加速度轨迹
  return predicted_traj.getTraj();  // 返回生成的轨迹
}

// 轨迹转ROS消息
void Traj2ROSMsg(const poly_traj::Trajectory &traj, const double des_clear, const int obstacle_id, traj_utils::MINCOTraj &MINCO_msg)
{
  // 获取轨迹参数
  Eigen::VectorXd durs = traj.getDurations();  // 获取轨迹 ​​每一段的时间长度​​，存储为Eigen的动态向量（VectorXd）
  int piece_num = traj.getPieceNum();  // 获取轨迹的分段数量​​（即有多少段独立的轨迹）
  double duration = durs.sum();  // 计算轨迹的​总持续时间​​（所有段时长之和）

  // 填充消息头
  MINCO_msg.drone_id = obstacle_id;
  MINCO_msg.traj_id = 0;
  MINCO_msg.start_time = ros::Time::now();
  MINCO_msg.order = 5; // 5阶多项式轨迹
  // 调整各段轨迹持续时间的数组长度为分段数量
  MINCO_msg.duration.resize(piece_num);
  // 设置安全距离
  MINCO_msg.des_clearance = des_clear;
  // 起点状态
  Eigen::Vector3d vec;  //单纯设置一个临时的向量并不表示速度
  vec = traj.getPos(0);  // 获取轨迹在t=0时刻的位置
  MINCO_msg.start_p[0] = vec(0), MINCO_msg.start_p[1] = vec(1), MINCO_msg.start_p[2] = vec(2);  // xyz坐标赋值给消息
  vec = traj.getVel(0);  // 获取t=0时刻的速度
  MINCO_msg.start_v[0] = vec(0), MINCO_msg.start_v[1] = vec(1), MINCO_msg.start_v[2] = vec(2);  // xyz速度赋值给消息
  vec = traj.getAcc(0);  // 获取t=0时刻的加速度
  MINCO_msg.start_a[0] = vec(0), MINCO_msg.start_a[1] = vec(1), MINCO_msg.start_a[2] = vec(2);  // xyz加速度赋值给消息

  // 终点状态
  vec = traj.getPos(duration);  // 获取轨迹在t=duration时刻的位置
  MINCO_msg.end_p[0] = vec(0), MINCO_msg.end_p[1] = vec(1), MINCO_msg.end_p[2] = vec(2);  // xyz坐标赋值给消息
  vec = traj.getVel(duration);  // 获取轨迹在t=duration时刻的速度
  MINCO_msg.end_v[0] = vec(0), MINCO_msg.end_v[1] = vec(1), MINCO_msg.end_v[2] = vec(2);  // xyz速度赋值给消息
  vec = traj.getAcc(duration);  // 获取轨迹在t=duration时刻的加速度
  MINCO_msg.end_a[0] = vec(0), MINCO_msg.end_a[1] = vec(1), MINCO_msg.end_a[2] = vec(2);  // xyz加速度赋值给消息

  // 中间点（对于MINCO轨迹）
  MINCO_msg.inner_x.resize(piece_num - 1);  //重新设置数组大小 如果轨迹分成 N 段（piece_num = N），则需要 N-1 个中间点来连接
  MINCO_msg.inner_y.resize(piece_num - 1);
  MINCO_msg.inner_z.resize(piece_num - 1);
  // 获取所有位置点
  Eigen::MatrixXd pos = traj.getPositions();
  // 填充中间点
  for (int i = 0; i < piece_num - 1; i++)
  {
    MINCO_msg.inner_x[i] = pos(0, i + 1);
    MINCO_msg.inner_y[i] = pos(1, i + 1);
    MINCO_msg.inner_z[i] = pos(2, i + 1);
  }

  //各段轨迹的时间赋值进msg.duration
  for (int i = 0; i < piece_num; i++)
    MINCO_msg.duration[i] = durs[i];
}

// #      ^                ^
// #    +1|              +4|
// # <-+0      ->     <-+3      ->
// #      |                |
// #      V                V

void obs_pv_cb(const traj_utils::Obspv::ConstPtr &msg)
{
  // 获取当前时间戳
  ros::Time t_now = ros::Time::now();
  // 更新pos_, vel_
  Eigen::Vector2d pos1(msg->p1[0], msg->p1[1]); 
  Eigen::Vector2d pos2(msg->p2[0], msg->p2[1]);
  Eigen::Vector2d vel1(msg->v1[0], msg->v1[1]);
  Eigen::Vector2d vel2(msg->v2[0], msg->v2[1]);
  obs1_.set_position(pos1);
  obs1_.set_velocity(vel1);
  obs2_.set_position(pos2);
  obs2_.set_velocity(vel2);
  //更新加速度和角速度
  double acc1 = 0;
  double dir1 = 0;
  double acc2 = 0;
  double dir2 = 0;
  // 更新障碍物状态(p,v)
  auto pv1 = obs1_.update(acc1, dir1);  //a = 0, w = 0
  auto pv2 = obs2_.update(acc2, dir2);
  // 设置障碍物固定高度
  constexpr double HEIGHT = 0.9;

  // 发布障碍物1的里程计信息
  // 创建里程计消息
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = t_now;
  odom_msg.header.frame_id = "world";  // 全局坐标系
  odom_msg.pose.pose.position.z = HEIGHT;  // Z固定高度
  odom_msg.twist.twist.linear.z = 0.0;  // Z方向速度为0
  odom_msg.pose.pose.orientation.x = 0.0;  // 四元数初始化
  odom_msg.pose.pose.orientation.y = 0.0;

  // 设置障碍物1的朝向（偏航角转四元数）
  Eigen::Quaterniond q1(Eigen::AngleAxisd(obs1_.get_yaw(), Eigen::Vector3d::UnitZ()));
  // 填充位置和速度
  odom_msg.pose.pose.position.x = pv1.first(0);
  odom_msg.pose.pose.position.y = pv1.first(1);
  odom_msg.twist.twist.linear.x = pv1.second(0);
  odom_msg.twist.twist.linear.y = pv1.second(1);
  odom_msg.pose.pose.orientation.w = q1.w();
  odom_msg.pose.pose.orientation.z = q1.z();
  obs1_odom_pub_.publish(odom_msg);  // 发布障碍物1里程计
  ros::Duration(0.005).sleep();  // 短暂延迟防止消息冲突

  // 设置障碍物2的朝向（偏航角转四元数）
  Eigen::Quaterniond q2(Eigen::AngleAxisd(obs2_.get_yaw(), Eigen::Vector3d::UnitZ()));
  // 填充位置和速度
  odom_msg.pose.pose.position.x = pv2.first(0);
  odom_msg.pose.pose.position.y = pv2.first(1);
  odom_msg.twist.twist.linear.x = pv2.second(0);
  odom_msg.twist.twist.linear.y = pv2.second(1);
  odom_msg.pose.pose.orientation.w = q2.w();
  odom_msg.pose.pose.orientation.z = q2.z();
  obs2_odom_pub_.publish(odom_msg);  // 发布障碍物2里程计
  ros::Duration(0.005).sleep();  // 短暂延迟防止消息冲突

  // 发布预测轨迹
  // 创建轨迹消息容器
  traj_utils::MINCOTraj MINCO_msg;
  vector<Eigen::Vector3d> vis_pts;
  // 预测障碍物1轨迹
  poly_traj::Trajectory traj1 = predict_traj(acc1, dir1, Eigen::Vector3d(pv1.first[0], pv1.first[1], HEIGHT), Eigen::Vector3d(pv1.second[0], pv1.second[1], 0), obs1_, vis_pts);
  Traj2ROSMsg(traj1, obs1_.des_clearance_, obs1_id_, MINCO_msg);  // 转换轨迹为ROS消息
  predicted_traj_pub_.publish(MINCO_msg);  // 发布轨迹
  ros::Duration(0.005).sleep();
  visualization_->displayInitPathList(vis_pts, 0.1, obs1_id_);  // 可视化预测路径
  ros::Duration(0.005).sleep();
  // 预测障碍物2轨迹
  poly_traj::Trajectory traj2 = predict_traj(acc2, dir2, Eigen::Vector3d(pv2.first[0], pv2.first[1], HEIGHT), Eigen::Vector3d(pv2.second[0], pv2.second[1], 0), obs2_, vis_pts);
  Traj2ROSMsg(traj2, obs2_.des_clearance_, obs2_id_, MINCO_msg);  // 转换轨迹为ROS消息
  predicted_traj_pub_.publish(MINCO_msg);  // 发布轨迹
  ros::Duration(0.005).sleep();
  visualization_->displayInitPathList(vis_pts, 0.1, obs2_id_);  // 可视化预测路径
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "moving_obstacles");
  ros::NodeHandle nh("~");

  obs1_odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom_obs1", 10);
  obs2_odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom_obs2", 10);
  predicted_traj_pub_ = nh.advertise<traj_utils::MINCOTraj>("/broadcast_traj_to_planner", 10);
//   ros::Subscriber joy_sub = nh.subscribe<sensor_msgs::Joy>("joy", 10, joy_sub_cb);
  ros::Subscriber obs_pv_sub = nh.subscribe<traj_utils::Obspv>("/obs_pos_and_vel", 10, obs_pv_cb);

  visualization_.reset(new ego_planner::PlanningVisualization(nh));

  std::vector<double> init_pos;  // 临时存储位置向量
  init_pos = {0, 0};
  // 障碍物1配置
  nh.getParam("obstacle1_init_pos", init_pos);  // 获取初始位置参数
  obs1_.set_position(Eigen::Vector2d(init_pos[0], init_pos[1]));  // 设置位置
  nh.param("desired_clearance1", obs1_.des_clearance_, 0.5);  // 获取安全距离

  // 障碍物2配置
  nh.getParam("obstacle2_init_pos", init_pos);  // 获取初始位置参数
  obs2_.set_position(Eigen::Vector2d(init_pos[0], init_pos[1]));  // 设置位置
  nh.param("desired_clearance2", obs2_.des_clearance_, 0.5);  // 获取安全距离
  // 获取障碍物ID
  nh.param("obstacle1_id", obs1_id_, 20.0);
  nh.param("obstacle2_id", obs2_id_, 21.0);
  
  while (ros::ok())
  {
    ros::Duration(0.01).sleep();
    ros::spinOnce();
  }

  return 0;
}
