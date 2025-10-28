#include <ros/ros.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/ColorRGBA.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>

#include <vector>
#include <deque>
#include <algorithm>
#include <memory>
#include <pointpillars_msgs/Obspv.h>

#include "nav_msgs/Odometry.h"


class BBoxTracker {
private:
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_;  //接收odom
    ros::Subscriber bbox_sub_;  // 接受发布的bbox消息
    ros::Publisher trajectory_pub_;  // 发布轨迹
    ros::Publisher velocity_pub_;  
    ros::Publisher prediction_pub_;
    ros::Publisher Obspv_pub;
    
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // 轨迹参数
    const int max_trajectory_length_ = 5;  // 最大轨迹长度
    const double prediction_time_ = 1.0;    // 预测时间（秒）
    const double association_threshold_ = 0.05; // 关联阈值（米）
    
    struct TrackedObject {  // 目标跟踪结构体
        int id;
        std::string label;
        std::deque<geometry_msgs::PointStamped> trajectory; // 历史轨迹
        std::deque<ros::Time> timestamps;                   // 时间戳
        geometry_msgs::Vector3 velocity;                    // 当前速度
        geometry_msgs::Point predicted_position;            // 预测位置
        bool is_active;
        ros::Time last_update_time;
        
        // 卡尔曼滤波器（简单版本）
        Eigen::Vector4d state; // [x, y, vx, vy]
        Eigen::Matrix4d P;     // 状态协方差矩阵
        
        TrackedObject(int obj_id, const std::string& obj_label) 
            : id(obj_id), label(obj_label), is_active(true) {
            // 初始化卡尔曼滤波器
            state = Eigen::Vector4d::Zero();
            P = Eigen::Matrix4d::Identity() * 0.1;
        }
    };
    
    std::vector<TrackedObject> tracked_objects_;  //存储所有正在被跟踪的目标对象
    int next_id_;

    //创建发布障碍物速度位置消息
    pointpillars_msgs::Obspv pv;

    //坐标转换
    Eigen::Matrix4d body2world;
    Eigen::Vector4d body_pose;
    Eigen::Quaterniond siyushu;
    Eigen::Matrix4d lidar2body;
    Eigen::Matrix4d lidar2world;
    Eigen::Vector4d box_pose;
    Eigen::Vector4d box_pose_world;

public:
    BBoxTracker() : 
        nh_("~"),
        tf_listener_(tf_buffer_),
        next_id_(0) {
        //初始化
        pv.p1, pv.p2 = {0, 0};
        pv.v1, pv.v2 = {0, 0};
        
        // 订阅BoundingBoxArray
        odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom", 1, &BBoxTracker::odomCallback, this);
        bbox_sub_ = nh_.subscribe("/detections", 10, &BBoxTracker::bboxCallback, this);
        
        // 发布轨迹和预测
        trajectory_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/object_trajectories", 10);
        velocity_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/object_velocities", 10);
        prediction_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/object_predictions", 10);
        Obspv_pub = nh_.advertise<pointpillars_msgs::Obspv>("/obs_pos_and_vel", 10);
        
        ROS_INFO("BBox Tracker initialized");
    }

private:
    void bboxCallback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& msg) {
        if (msg->boxes.empty()) {
            ROS_WARN_THROTTLE(5.0, "No bounding boxes received");
            return;
        }
        
        // 处理每个检测到的框
        for (const auto& bbox : msg->boxes) {  // 遍历boxes里的每个box赋值给bbox
            processBoundingBox(bbox);
        }
        
        // 更新轨迹和预测
        updateTrajectories();
        
        // 发布可视化信息
        publishVisualizations();
        
        // 清理不活跃的目标
        cleanupInactiveObjects();

        // 发布消息
        Obspv_pub.publish(pv);
    }
    
    void odomCallback(const nav_msgs::Odometry::ConstPtr &odom){
        body2world = Eigen::Matrix4d::Identity();
        body_pose(0) = odom->pose.pose.position.x;
        body_pose(1) = odom->pose.pose.position.y;
        body_pose(2) = odom->pose.pose.position.z;
        body_pose(3) = 1.0;
        siyushu.x() = odom->pose.pose.orientation.x;
        siyushu.y() = odom->pose.pose.orientation.y;
        siyushu.z() = odom->pose.pose.orientation.z;
        siyushu.w() = odom->pose.pose.orientation.w;

        body2world.block<3, 3>(0, 0) = siyushu.toRotationMatrix();
        body2world(0, 3) = body_pose(0);
        body2world(1, 3) = body_pose(0);
        body2world(2, 3) = body_pose(0);
        lidar2body << 1, 0, 0, -0.07,
                      0, 1, 0, 0,
                      0, 0, 1, -0.04,
                      0, 0, 0, 1;
        lidar2world = body2world * lidar2body;
        
    }

    //处理检测到的bbox
    void processBoundingBox(const jsk_recognition_msgs::BoundingBox& bbox) {
        geometry_msgs::PointStamped point;  //储存带时间戳的位姿
        point.header = bbox.header;
        point.point = bbox.pose.position;
        box_pose(0) = bbox.pose.position.x;
        box_pose(1) = bbox.pose.position.y;
        box_pose(2) = bbox.pose.position.z;
        box_pose(3) = 1.0;
        
        // 尝试将点转换到固定坐标系（例如map）
        geometry_msgs::PointStamped transformed_point;
        transformed_point.header = point.header;
        transformed_point.point = point.point;
        
        //坐标转换
        // box_pose_world = lidar2world * box_pose;
        // transformed_point.header = point.header;
        // transformed_point.header.frame_id = "world";
        // transformed_point.point.x = box_pose_world(0);
        // transformed_point.point.y = box_pose_world(1);
        // transformed_point.point.z = box_pose_world(2);
        
        // 关联到现有目标或创建新目标
        int best_match_id = -1;
        double best_distance = association_threshold_;  //设置关联的最大距离
        for (size_t i = 0; i < tracked_objects_.size(); ++i) {
            if (!tracked_objects_[i].is_active) continue;
            
            double distance = calculateDistance(
                transformed_point.point, 
                tracked_objects_[i].trajectory.back().point 
            );
            
            if (distance < best_distance) {  //判断bbox与跟踪轨迹的距离是否满足关联距离并找出其关联轨迹
                best_distance = distance;
                best_match_id = i;
            }
        }
        
        if (best_match_id != -1) {
            // 更新现有目标
            updateTrackedObject(best_match_id, transformed_point, bbox);  //更新轨迹
        } else {
            // 创建新目标（限制为2个目标）
            if (tracked_objects_.size() < 2) {
                createNewTrackedObject(transformed_point, bbox);  //创建新的轨迹
            }
        }
    }
    
    // 计算距离
    double calculateDistance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2) { 
        return sqrt(pow(p1.x - p2.x, 2));  // 只计算x轴距离
    }
    
    void createNewTrackedObject(const geometry_msgs::PointStamped& point, 
                               const jsk_recognition_msgs::BoundingBox& bbox) {
        std::string label = (bbox.label == 1) ? "object_1" : "object_2";
        TrackedObject new_obj(next_id_++, label);
        
        new_obj.trajectory.push_back(point);
        new_obj.timestamps.push_back(point.header.stamp);
        new_obj.last_update_time = ros::Time::now();
        
        // 初始化状态
        new_obj.state << point.point.x, point.point.y, 0, 0;
        
        tracked_objects_.push_back(new_obj);
        
        ROS_INFO("创建新的tracked object: %s (ID: %d)", label.c_str(), new_obj.id);
    }

    // 更新追踪目标    
    void updateTrackedObject(int obj_id, const geometry_msgs::PointStamped& point,
                           const jsk_recognition_msgs::BoundingBox& bbox) {
        TrackedObject& obj = tracked_objects_[obj_id];
        
        // 添加新位置到轨迹
        obj.trajectory.push_back(point);
        obj.timestamps.push_back(point.header.stamp);
        obj.last_update_time = ros::Time::now();
        
        // 限制轨迹长度
        if (obj.trajectory.size() > max_trajectory_length_) {
            obj.trajectory.pop_front();
            obj.timestamps.pop_front();
        }
        
        // 计算速度（如果至少有2个点）
        if (obj.trajectory.size() >= 2) {
            calculateVelocity(obj);

            // 将位置和速度赋值给Obspv消息
            if(obj_id == 0){
                pv.p1[0] = point.point.x;
                pv.p1[1] = point.point.y;
                pv.v1[0] = 0;
                pv.v1[1] = obj.velocity.y;
                std::cout << pv.v1[1] << std::endl;
            }
            else if(obj_id == 1){
                pv.p2[0] = point.point.x;
                pv.p2[1] = point.point.y;
                pv.v2[0] = 0;
                pv.v2[1] = obj.velocity.y;
                std::cout << pv.v2[1] << std::endl;
            }
        }
        
        // 更新卡尔曼滤波器
        updateKalmanFilter(obj, point.point);
        
        // 预测未来位置
        predictFuturePosition(obj);
    }
    

    // 计算速度
    void calculateVelocity(TrackedObject& obj) {
        const auto& recent_points = obj.trajectory;
        const auto& recent_times = obj.timestamps;
        
        if (recent_points.size() < 2) return;
        
        // 使用最近几个点计算平均速度
        int num_points = std::min(5, static_cast<int>(recent_points.size()));  //在轨迹中取5个点，当轨迹小于5个点时取轨迹的size
        double total_dt = 0;
        geometry_msgs::Vector3 total_velocity;
        total_velocity.x = 0; total_velocity.y = 0; total_velocity.z = 0;
        
        for (int i = 1; i < num_points; ++i) {
            double dt = (recent_times[i] - recent_times[i-1]).toSec();
            if (dt > 0) {
                total_velocity.x += (recent_points[i].point.x - recent_points[i-1].point.x) / dt;
                total_velocity.y += (recent_points[i].point.y - recent_points[i-1].point.y) / dt;
                total_velocity.z += (recent_points[i].point.z - recent_points[i-1].point.z) / dt;
                total_dt += dt;
            }
        }
        
        if (total_dt > 0) {  //求轨迹段的平均速度
            obj.velocity.x = total_velocity.x / (num_points - 1);
            obj.velocity.y = total_velocity.y / (num_points - 1);
            obj.velocity.z = total_velocity.z / (num_points - 1);
        }
    }
    
    // 卡尔曼滤波更新
    void updateKalmanFilter(TrackedObject& obj, const geometry_msgs::Point& measurement) {
        // 简化的卡尔曼滤波更新
        Eigen::Vector3d z(measurement.x, measurement.y, measurement.z);
        
        // 预测步骤
        double dt = (ros::Time::now() - obj.last_update_time).toSec();
        if (dt <= 0) dt = 0.1;
        
        Eigen::Matrix4d F; // 状态转移矩阵
        F << 1, 0, 0, dt,
             0, 1, 0, dt,
             0, 0, 1, dt,
             0, 0, 0, 1;
        
        obj.state = F * obj.state;
        obj.P = F * obj.P * F.transpose() + Eigen::Matrix4d::Identity() * 0.01;
        
        // 更新步骤
        Eigen::Matrix<double, 3, 4> H; // 观测矩阵
        H << 1, 0, 0, 0,
             0, 1, 0, 0, 
             0, 0, 1, 0;
        
        Eigen::Vector3d y = z - H * obj.state;
        Eigen::Matrix3d S = H * obj.P * H.transpose() + Eigen::Matrix3d::Identity() * 0.1;
        Eigen::Matrix<double, 4, 3> K = obj.P * H.transpose() * S.inverse();
        
        obj.state = obj.state + K * y;
        obj.P = (Eigen::Matrix4d::Identity() - K * H) * obj.P;
    }
    
    //预测 
    void predictFuturePosition(TrackedObject& obj) {
        // 使用当前状态和速度预测未来位置
        obj.predicted_position.x = obj.state[0] + obj.velocity.x * prediction_time_;
        obj.predicted_position.y = obj.state[1] + obj.velocity.y * prediction_time_;
    }
    
    //更新轨迹
    void updateTrajectories() {
        for (auto& obj : tracked_objects_) {
            if (!obj.is_active) continue;
            
            // 检查目标是否长时间未更新
            if ((ros::Time::now() - obj.last_update_time).toSec() > 2.0) {
                obj.is_active = false;
                ROS_WARN("Object %d marked as inactive", obj.id);
            }
        }
    }
    
    // 可视化
    void publishVisualizations() {
        visualization_msgs::MarkerArray trajectory_markers;
        visualization_msgs::MarkerArray velocity_markers;
        visualization_msgs::MarkerArray prediction_markers;
        
        for (const auto& obj : tracked_objects_) {
            if (!obj.is_active || obj.trajectory.size() < 2) continue;
            
            // 发布轨迹
            visualization_msgs::Marker trajectory_marker;
            trajectory_marker.header.frame_id = "map";
            trajectory_marker.header.stamp = ros::Time::now();
            trajectory_marker.ns = "trajectories";
            trajectory_marker.id = obj.id;
            trajectory_marker.type = visualization_msgs::Marker::LINE_STRIP;
            trajectory_marker.action = visualization_msgs::Marker::ADD;
            trajectory_marker.scale.x = 0.05;
            
            // 设置颜色（根据ID区分）
            if (obj.id == 0) {
                trajectory_marker.color.r = 1.0; // 红色
                trajectory_marker.color.g = 0.0;
                trajectory_marker.color.b = 0.0;
            } else {
                trajectory_marker.color.r = 0.0; // 蓝色
                trajectory_marker.color.g = 0.0;
                trajectory_marker.color.b = 1.0;
            }
            trajectory_marker.color.a = 1.0;
            
            // 添加轨迹点
            for (const auto& point : obj.trajectory) {
                trajectory_marker.points.push_back(point.point);
            }
            trajectory_markers.markers.push_back(trajectory_marker);
            
            // 发布速度箭头
            visualization_msgs::Marker velocity_marker;
            velocity_marker.header.frame_id = "map";
            velocity_marker.header.stamp = ros::Time::now();
            velocity_marker.ns = "velocities";
            velocity_marker.id = obj.id;
            velocity_marker.type = visualization_msgs::Marker::ARROW;
            velocity_marker.action = visualization_msgs::Marker::ADD;
            velocity_marker.scale.x = 0.1; // 箭头直径
            velocity_marker.scale.y = 0.2; // 箭头头部直径
            velocity_marker.scale.z = 0.3; // 箭头头部长度
            
            // 设置颜色
            if (obj.id == 0) {
                velocity_marker.color.r = 1.0;
                velocity_marker.color.g = 0.5;
                velocity_marker.color.b = 0.0;
            } else {
                velocity_marker.color.r = 0.0;
                velocity_marker.color.g = 0.5;
                velocity_marker.color.b = 1.0;
            }
            velocity_marker.color.a = 0.8;
            
            // 设置箭头位置和方向
            geometry_msgs::Point start_point = obj.trajectory.back().point;
            geometry_msgs::Point end_point;
            end_point.x = start_point.x + obj.velocity.x * 0.5; // 缩放速度显示
            end_point.y = start_point.y + obj.velocity.y * 0.5;
            end_point.z = start_point.z + obj.velocity.z * 0.5;
            
            velocity_marker.points.push_back(start_point);
            velocity_marker.points.push_back(end_point);
            velocity_markers.markers.push_back(velocity_marker);
            
            // 发布预测轨迹
            visualization_msgs::Marker prediction_marker;
            prediction_marker.header.frame_id = "map";
            prediction_marker.header.stamp = ros::Time::now();
            prediction_marker.ns = "predictions";
            prediction_marker.id = obj.id;
            prediction_marker.type = visualization_msgs::Marker::LINE_STRIP;
            prediction_marker.action = visualization_msgs::Marker::ADD;
            prediction_marker.scale.x = 0.03;
            
            // 设置预测轨迹颜色（半透明）
            if (obj.id == 0) {
                prediction_marker.color.r = 1.0;
                prediction_marker.color.g = 0.0;
                prediction_marker.color.b = 0.0;
            } else {
                prediction_marker.color.r = 0.0;
                prediction_marker.color.g = 0.0;
                prediction_marker.color.b = 1.0;
            }
            prediction_marker.color.a = 0.5;
            
            // 添加预测点
            geometry_msgs::Point current_pos = obj.trajectory.back().point;
            prediction_marker.points.push_back(current_pos);
            prediction_marker.points.push_back(obj.predicted_position);
            prediction_markers.markers.push_back(prediction_marker);
            
            // 打印信息
            std::cout << obj.id << std::endl;
            // ROS_INFO_THROTTLE(2.0, "Object %d - Pos: (%.2f, %.2f, %.2f), Vel: (%.2f, %.2f, %.2f)", 
            //                  obj.id, current_pos.x, current_pos.y, current_pos.z,
            //                  obj.velocity.x, obj.velocity.y, obj.velocity.z);
            ROS_INFO("Object %d - Pos: (%.2f, %.2f, %.2f), Vel: (%.2f, %.2f, %.2f)", 
                             obj.id, current_pos.x, current_pos.y, current_pos.z,
                             obj.velocity.x, obj.velocity.y, obj.velocity.z);
        }
        
        // 发布所有标记
        // trajectory_pub_.publish(trajectory_markers);
        // velocity_pub_.publish(velocity_markers);
        // prediction_pub_.publish(prediction_markers);
    }
    
    void cleanupInactiveObjects() {
        tracked_objects_.erase(
            std::remove_if(tracked_objects_.begin(), tracked_objects_.end(),
                [](const TrackedObject& obj) { return !obj.is_active; }),
            tracked_objects_.end()
        );
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "bbox_tracker_node");
    
    BBoxTracker tracker;
    
    ros::spin();
    
    return 0;
}