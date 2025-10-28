#include <ros/ros.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Dense>
#include <deque>
#include <map>

class ObstacleTracker {
private:
    ros::NodeHandle nh_;
    ros::Subscriber bbox_sub_;
    ros::Publisher obstacle1_pos_pub_;
    ros::Publisher obstacle1_vel_pub_;
    ros::Publisher obstacle2_pos_pub_;
    ros::Publisher obstacle2_vel_pub_;
    ros::Publisher twist_pub_;
    ros::Publisher marker_pub_;
    
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // 障碍物历史数据
    struct ObstacleData {
        std::deque<geometry_msgs::PointStamped> position_history;
        std::deque<ros::Time> time_history;
        geometry_msgs::Vector3 velocity;
        int id;
    };
    
    std::map<int, ObstacleData> obstacles_;
    int max_history_size_;
    double velocity_smoothing_factor_;
    
public:
    ObstacleTracker() : 
        tf_listener_(tf_buffer_),
        max_history_size_(10),
        velocity_smoothing_factor_(0.3) {
        
        // 订阅边界框
        bbox_sub_ = nh_.subscribe("/detections", 1, &ObstacleTracker::bboxCallback, this);
        
        // 发布障碍物1的位置和速度
        obstacle1_pos_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/obstacle1/position", 1);
        obstacle1_vel_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>("/obstacle1/velocity", 1);
        
        // 发布障碍物2的位置和速度
        obstacle2_pos_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/obstacle2/position", 1);
        obstacle2_vel_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>("/obstacle2/velocity", 1);
        
        // 发布综合速度信息
        twist_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/obstacles_twist", 1);
        
        // 发布可视化标记
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/obstacles_markers", 1);
        
        ROS_INFO("Obstacle Tracker initialized");
    }
    
    void bboxCallback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& msg) {
        if (msg->boxes.empty()) {
            ROS_WARN("No bounding boxes received");
            return;
        }
        
        // 按照x坐标排序，区分左右两个障碍物
        std::vector<jsk_recognition_msgs::BoundingBox> sorted_boxes = msg->boxes;
        std::sort(sorted_boxes.begin(), sorted_boxes.end(), 
                  [](const jsk_recognition_msgs::BoundingBox& a, 
                     const jsk_recognition_msgs::BoundingBox& b) {
                      return a.pose.position.x < b.pose.position.x;
                  });
        
        // 处理每个障碍物
        for (size_t i = 0; i < std::min(sorted_boxes.size(), size_t(2)); ++i) {
            int obstacle_id = i + 1;  // 障碍物ID: 1, 2
            
            // 获取当前障碍物的位置
            geometry_msgs::PointStamped obstacle_point;
            obstacle_point.header = msg->header;
            obstacle_point.point = sorted_boxes[i].pose.position;
            
            // 转换到世界坐标系（如果需要）
            geometry_msgs::PointStamped world_point;
            try {
                tf_buffer_.transform(obstacle_point, world_point, "world");
            } catch (tf2::TransformException& ex) {
                ROS_WARN("TF transform failed: %s", ex.what());
                world_point = obstacle_point;  // 使用原始坐标
            }
            
            // 更新障碍物数据
            updateObstacleData(obstacle_id, world_point, msg->header.stamp);
            
            // 计算并发布速度
            calculateAndPublishVelocity(obstacle_id, msg->header.stamp);
        }
        
        // 发布可视化标记
        publishVisualizationMarkers(msg->header);
        
        // 发布综合信息
        publishCombinedInfo(msg->header.stamp);
    }
    
    void updateObstacleData(int id, const geometry_msgs::PointStamped& point, const ros::Time& stamp) {
        // 查找或创建障碍物数据
        if (obstacles_.find(id) == obstacles_.end()) {
            ObstacleData new_obstacle;
            new_obstacle.id = id;
            obstacles_[id] = new_obstacle;
        }
        
        ObstacleData& obstacle = obstacles_[id];
        
        // 添加新位置和时间到历史记录
        obstacle.position_history.push_back(point);
        obstacle.time_history.push_back(stamp);
        
        // 保持历史记录大小
        if (obstacle.position_history.size() > max_history_size_) {
            obstacle.position_history.pop_front();
            obstacle.time_history.pop_front();
        }
    }
    
    void calculateAndPublishVelocity(int id, const ros::Time& stamp) {
        if (obstacles_.find(id) == obstacles_.end()) {
            return;
        }
        
        ObstacleData& obstacle = obstacles_[id];
        
        if (obstacle.position_history.size() < 2) {
            // 需要至少2个点才能计算速度
            return;
        }
        
        // 使用最近的两个点计算瞬时速度
        const auto& current_pos = obstacle.position_history.back();
        const auto& prev_pos = obstacle.position_history[obstacle.position_history.size() - 2];
        const auto& current_time = obstacle.time_history.back();
        const auto& prev_time = obstacle.time_history[obstacle.time_history.size() - 2];
        
        double dt = (current_time - prev_time).toSec();
        if (dt <= 0) {
            return;
        }
        
        // 计算瞬时速度
        geometry_msgs::Vector3 instantaneous_vel;
        instantaneous_vel.x = (current_pos.point.x - prev_pos.point.x) / dt;
        instantaneous_vel.y = (current_pos.point.y - prev_pos.point.y) / dt;
        instantaneous_vel.z = (current_pos.point.z - prev_pos.point.z) / dt;
        
        // 平滑速度（指数移动平均）
        if (obstacle.position_history.size() == 2) {
            // 第一次计算，直接使用瞬时速度
            obstacle.velocity = instantaneous_vel;
        } else {
            obstacle.velocity.x = velocity_smoothing_factor_ * instantaneous_vel.x + 
                                 (1 - velocity_smoothing_factor_) * obstacle.velocity.x;
            obstacle.velocity.y = velocity_smoothing_factor_ * instantaneous_vel.y + 
                                 (1 - velocity_smoothing_factor_) * obstacle.velocity.y;
            obstacle.velocity.z = velocity_smoothing_factor_ * instantaneous_vel.z + 
                                 (1 - velocity_smoothing_factor_) * obstacle.velocity.z;
        }
        
        // 发布位置
        geometry_msgs::PointStamped pos_msg;
        pos_msg.header.stamp = stamp;
        pos_msg.header.frame_id = current_pos.header.frame_id;
        pos_msg.point = current_pos.point;
        
        // 发布速度
        geometry_msgs::Vector3Stamped vel_msg;
        vel_msg.header.stamp = stamp;
        vel_msg.header.frame_id = current_pos.header.frame_id;
        vel_msg.vector = obstacle.velocity;
        
        // 根据障碍物ID发布到相应的话题
        if (id == 1) {
            obstacle1_pos_pub_.publish(pos_msg);
            obstacle1_vel_pub_.publish(vel_msg);
            
            ROS_INFO("Obstacle 1 - Position: (%.3f, %.3f, %.3f), Velocity: (%.3f, %.3f, %.3f) m/s",
                     pos_msg.point.x, pos_msg.point.y, pos_msg.point.z,
                     vel_msg.vector.x, vel_msg.vector.y, vel_msg.vector.z);
        } else if (id == 2) {
            obstacle2_pos_pub_.publish(pos_msg);
            obstacle2_vel_pub_.publish(vel_msg);
            
            ROS_INFO("Obstacle 2 - Position: (%.3f, %.3f, %.3f), Velocity: (%.3f, %.3f, %.3f) m/s",
                     pos_msg.point.x, pos_msg.point.y, pos_msg.point.z,
                     vel_msg.vector.x, vel_msg.vector.y, vel_msg.vector.z);
        }
    }
    
    void publishVisualizationMarkers(const std_msgs::Header& header) {
        visualization_msgs::MarkerArray marker_array;
        
        for (const auto& pair : obstacles_) {
            int id = pair.first;
            const ObstacleData& obstacle = pair.second;
            
            if (obstacle.position_history.empty()) {
                continue;
            }
            
            const auto& current_pos = obstacle.position_history.back();
            
            // 位置标记
            visualization_msgs::Marker pos_marker;
            pos_marker.header = header;
            pos_marker.ns = "obstacles";
            pos_marker.id = id * 2;
            pos_marker.type = visualization_msgs::Marker::SPHERE;
            pos_marker.action = visualization_msgs::Marker::ADD;
            pos_marker.pose.position = current_pos.point;
            pos_marker.pose.orientation.w = 1.0;
            pos_marker.scale.x = 0.3;
            pos_marker.scale.y = 0.3;
            pos_marker.scale.z = 0.3;
            pos_marker.color.r = (id == 1) ? 1.0 : 0.0;
            pos_marker.color.g = (id == 2) ? 1.0 : 0.0;
            pos_marker.color.b = 0.0;
            pos_marker.color.a = 0.8;
            pos_marker.lifetime = ros::Duration(0.5);
            
            // 速度向量标记
            visualization_msgs::Marker vel_marker;
            vel_marker.header = header;
            vel_marker.ns = "velocities";
            vel_marker.id = id * 2 + 1;
            vel_marker.type = visualization_msgs::Marker::ARROW;
            vel_marker.action = visualization_msgs::Marker::ADD;
            vel_marker.pose.position = current_pos.point;
            
            // 设置箭头方向
            double speed = sqrt(obstacle.velocity.x * obstacle.velocity.x + 
                               obstacle.velocity.y * obstacle.velocity.y + 
                               obstacle.velocity.z * obstacle.velocity.z);
            
            if (speed > 0.01) {
                // 计算箭头方向
                geometry_msgs::Point arrow_end;
                arrow_end.x = current_pos.point.x + obstacle.velocity.x * 0.5;  // 缩放因子
                arrow_end.y = current_pos.point.y + obstacle.velocity.y * 0.5;
                arrow_end.z = current_pos.point.z + obstacle.velocity.z * 0.5;
                
                vel_marker.points.push_back(current_pos.point);
                vel_marker.points.push_back(arrow_end);
                
                vel_marker.scale.x = 0.05;  // 箭头直径
                vel_marker.scale.y = 0.1;   // 箭头头部直径
                vel_marker.scale.z = 0.2;   // 箭头头部长度
                
                vel_marker.color.r = (id == 1) ? 1.0 : 0.0;
                vel_marker.color.g = (id == 2) ? 1.0 : 0.0;
                vel_marker.color.b = 1.0;
                vel_marker.color.a = 0.8;
                vel_marker.lifetime = ros::Duration(0.5);
                
                marker_array.markers.push_back(vel_marker);
            }
            
            marker_array.markers.push_back(pos_marker);
        }
        
        marker_pub_.publish(marker_array);
    }
    
    void publishCombinedInfo(const ros::Time& stamp) {
        if (obstacles_.size() < 2) {
            return;
        }
        
        geometry_msgs::TwistStamped twist_msg;
        twist_msg.header.stamp = stamp;
        twist_msg.header.frame_id = "world";
        
        // 计算相对速度等信息
        const auto& obs1 = obstacles_[1];
        const auto& obs2 = obstacles_[2];
        
        if (!obs1.position_history.empty() && !obs2.position_history.empty()) {
            const auto& pos1 = obs1.position_history.back().point;
            const auto& pos2 = obs2.position_history.back().point;
            
            // 相对位置
            twist_msg.twist.linear.x = pos1.x - pos2.x;  // x方向相对距离
            twist_msg.twist.linear.y = pos1.y - pos2.y;  // y方向相对距离
            twist_msg.twist.linear.z = pos1.z - pos2.z;  // z方向相对距离
            
            // 相对速度
            twist_msg.twist.angular.x = obs1.velocity.x - obs2.velocity.x;
            twist_msg.twist.angular.y = obs1.velocity.y - obs2.velocity.y;
            twist_msg.twist.angular.z = obs1.velocity.z - obs2.velocity.z;
            
            twist_pub_.publish(twist_msg);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "obstacle_tracker");
    
    ObstacleTracker tracker;
    
    ROS_INFO("Obstacle Tracker node started");
    
    ros::spin();
    
    return 0;
}