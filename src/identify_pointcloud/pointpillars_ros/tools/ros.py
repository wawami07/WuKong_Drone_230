#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import time
import numpy as np
from pyquaternion import Quaternion

import argparse
import glob
from pathlib import Path

# import mayavi.mlab as mlab
import numpy as np
import torch
import scipy.linalg as linalg

import sys
sys.path.append("/home/wk2/PointPillars_ROS/src/pointpillars_ros")

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path: 根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 训练模式
            logger: 日志
            ext: 扩展名
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        


class Pointpillars_ROS:
    def __init__(self):
        # 初始化计时变量
        self.total_inference_time = 0.0
        self.total_preprocessing_time = 0.0
        self.frame_count = 0
        self.avg_inference_time = 0.0
        self.avg_preprocessing_time = 0.0
        self.avg_fps = 0.0
        
        # 置信度阈值
        self.confidence_threshold = 0.6
    
        config_path, ckpt_path = self.init_ros()
        self.init_pointpillars(config_path, ckpt_path)


    def init_ros(self):
        """ Initialize ros parameters """
        config_path = rospy.get_param("/config_path", "/home/wk2/PointPillars_ROS/src/pointpillars_ros/cfgs/custom_models/custom.yaml")
        ckpt_path = rospy.get_param("/ckpt_path", "/home/wk2/PointPillars_ROS/src/pointpillars_ros/models/best_1600_e80_s20.pth")
        self.sub_velo = rospy.Subscriber("/cloud_registered", PointCloud2, self.lidar_callback, queue_size=1,  buff_size=2**12)
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=1)
        
        # 发布推理时间信息
        self.pub_inference_time = rospy.Publisher("/inference_time", Header, queue_size=1)
        
        return config_path, ckpt_path


    def init_pointpillars(self, config_path, ckpt_path):
        """ Initialize second model """
        import os
        os.chdir("/home/wk2/PointPillars_ROS/src/pointpillars_ros")
        logger = common_utils.create_logger() # 创建日志
        logger.info('-----------------Quick Demo of Pointpillars-------------------------')
        cfg_from_yaml_file(config_path, cfg)  # 加载配置文件
        
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            ext='.bin', logger=logger
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        # 加载权重文件
        self.model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        self.model.cuda() # 将网络放到GPU上
        self.model.eval() # 开启评估模式


    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix


    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """
        # 记录开始时间
        total_start_time = time.time()
        
        pcl_msg = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z","intensity","ring"))
        np_p = np.array(list(pcl_msg), dtype=np.float32)
        # 旋转轴
        rand_axis = [0,1,0]  # y轴
        #旋转角度0.1047
        yaw =  0
        #返回旋转矩阵
        rot_matrix = self.rotate_mat(rand_axis, yaw)  # 绕y轴旋转的旋转矩阵
        np_p_rot = np.dot(rot_matrix, np_p[:,:3].T).T
        
        # convert to xyzi point cloud
        x = np_p_rot[:, 0].reshape(-1)
        y = np_p_rot[:, 1].reshape(-1)
        z = np_p_rot[:, 2].reshape(-1)
        if np_p.shape[1] == 4: # if intensity field exists
            i = np_p[:, 3].reshape(-1)
        else:
            i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        points = np.stack((x, y, z, i)).T
        print(points.shape)
        
        # 数据预处理开始时间
        preprocessing_start_time = time.time()
        
        # 组装数组字典
        input_dict = {
            'points': points,
            'frame_id': 0,
        }
        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict) # 数据预处理
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict) # 将数据放到GPU上
        
        # 数据预处理结束时间
        preprocessing_time = time.time() - preprocessing_start_time
        
        # 模型推理开始时间
        inference_start_time = time.time()
        
        with torch.no_grad():  # 确保在推理时不计算梯度
            pred_dicts, _ = self.model.forward(data_dict) # 模型前向传播
        
        # 模型推理结束时间
        inference_time = time.time() - inference_start_time
        
        boxes_lidar = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
        scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        label = pred_dicts[0]['pred_labels'].detach().cpu().numpy()
        
        # 应用置信度过滤
        high_confidence_mask = scores > self.confidence_threshold
        boxes_lidar = boxes_lidar[high_confidence_mask]
        scores = scores[high_confidence_mask]
        label = label[high_confidence_mask]
        
        num_detections = boxes_lidar.shape[0]
        
        # 总处理时间
        total_time = time.time() - total_start_time
        
        # 更新统计信息
        self.frame_count += 1
        self.total_inference_time += inference_time
        self.total_preprocessing_time += preprocessing_time
        
        # 计算平均值
        self.avg_inference_time = self.total_inference_time / self.frame_count
        self.avg_preprocessing_time = self.total_preprocessing_time / self.frame_count
        self.avg_fps = self.frame_count / (self.total_inference_time + self.total_preprocessing_time)
        
        # 打印详细的性能信息
        rospy.loginfo("\n=== 模型性能统计 ===")
        rospy.loginfo("检测到目标数量: %d", num_detections)
        rospy.loginfo("置信度阈值: %.2f", self.confidence_threshold)
        rospy.loginfo("当前帧 - 预处理时间: %.4f秒", preprocessing_time)
        rospy.loginfo("当前帧 - 推理时间: %.4f秒", inference_time)
        rospy.loginfo("当前帧 - 总处理时间: %.4f秒", total_time)
        rospy.loginfo("当前帧 - FPS: %.2f", 1.0 / total_time)
        rospy.loginfo("累计统计 - 平均预处理时间: %.4f秒", self.avg_preprocessing_time)
        rospy.loginfo("累计统计 - 平均推理时间: %.4f秒", self.avg_inference_time)
        rospy.loginfo("累计统计 - 平均FPS: %.2f", self.avg_fps)
        rospy.loginfo("累计处理帧数: %d", self.frame_count)
        rospy.loginfo("===================")

        # print(boxes_lidar)
        # print(scores)
        # print(label)

        arr_bbox = BoundingBoxArray()  # 创建BoundingBoxArray消息
        for i in range(num_detections):
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            # bbox.pose.position.z = float(boxes_lidar[i][2]) + float(boxes_lidar[i][5]) / 2
            bbox.pose.position.z = float(boxes_lidar[i][2])
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]

            arr_bbox.boxes.append(bbox)
        
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        
        self.pub_bbox.publish(arr_bbox)
        
        # 发布推理时间信息（可选）
        time_header = Header()
        time_header.stamp = rospy.Time.now()
        time_header.frame_id = f"推理时间: {inference_time:.4f}s, FPS: {1.0/total_time:.2f}"
        self.pub_inference_time.publish(time_header)


if __name__ == '__main__':
    sec = Pointpillars_ROS()
    rospy.init_node('pointpillars_ros_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        # 在程序退出时打印最终统计信息
        rospy.loginfo("\n=== 最终性能统计 ===")
        rospy.loginfo("总处理帧数: %d", sec.frame_count)
        rospy.loginfo("平均预处理时间: %.4f秒", sec.avg_preprocessing_time)
        rospy.loginfo("平均推理时间: %.4f秒", sec.avg_inference_time)
        rospy.loginfo("平均FPS: %.2f", sec.avg_fps)
        rospy.loginfo("===================")
        del sec
        print("Shutting down")