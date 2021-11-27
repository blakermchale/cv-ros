#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import numpy as np
import pyrealsense2 as rs2
from tf2_ros.transform_broadcaster import TransformBroadcaster

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped


class BBTfCreator(Node):
    def __init__(self):
        super().__init__("bb_tf_creator")
        # Subscribers
        # self._sub_depth = self.create_subscription(CompressedImage, "/drone_0/realsense/aligned_depth_to_color/image_raw", self._cb_depth_image, QoSPresetProfiles.SENSOR_DATA.value)
        # self._sub_bb = self.create_subscription(BoundingBoxes, "/darknet_ros/bounding_boxes", self._cb_bb, 10)
        # self._sub_info = self.create_subscription(CameraInfo, "/drone_0/realsense/aligned_depth_to_color/camera_info", self._cb_info, 10)

        # self._sub_bb = Subscriber(self, BoundingBoxes, "bounding_boxes", self._cb_bb, 10)
        # self._sub_depth = Subscriber(self, Image, "depth/image", self._cb_depth_image, QoSPresetProfiles.SENSOR_DATA)
        self._namespace = "drone_0"

        self._fps = 30.0  # FPS assumption for time delay between matching messages

        self.bridge = CvBridge()

        # Subscribers
        self._sub_bb = Subscriber(self, BoundingBoxes, "darknet_ros/bounding_boxes", qos_profile=10)
        # self._sub_depth = Subscriber(self, CompressedImage, "/drone_0/realsense/aligned_depth_to_color/image_raw/compressedDepth", qos_profile=QoSPresetProfiles.SENSOR_DATA.value)
        self._sub_depth = Subscriber(self, Image, "realsense/aligned_depth_to_color/image_raw", qos_profile=1)
        self._sub_info = Subscriber(self, CameraInfo, "realsense/aligned_depth_to_color/camera_info", qos_profile=1)
        self._ts = ApproximateTimeSynchronizer([self._sub_bb, self._sub_depth, self._sub_info], 2, 1.0/self._fps)
        self._ts.registerCallback(self._cb_ts)

        # Publishers
        self._pub_bb_depth = self.create_publisher(Image, "bb_depth", 1)

        # TF
        self._br = TransformBroadcaster(self)

    def _cb_depth_image(self, msg: Image):
        self.get_logger().info(f"{msg.header}")
        pass

    def _cb_bb(self, msg: BoundingBoxes):
        self.get_logger().info(f"{msg.header}")
        pass

    def _cb_info(self, msg: CameraInfo):
        self.get_logger().info(f"{msg.header}")
        pass

    def _cb_ts(self, bb_msg: BoundingBoxes, depth_msg: Image, info: CameraInfo):
        im_encoding = "passthrough"
        im = self.bridge.imgmsg_to_cv2(depth_msg,im_encoding)

        intrinsics = rs2.intrinsics()
        intrinsics.width = info.width
        intrinsics.height = info.height
        intrinsics.ppx = info.k[2]
        intrinsics.ppy = info.k[5]
        intrinsics.fx = info.k[0]
        intrinsics.fy = info.k[4]
        intrinsics.model = rs2.distortion.none
        if info.d: intrinsics.coeffs = [float(i) for i in info.d]

        header = Header()
        header.stamp = self.get_clock().now().to_msg() # bb_msg.header.stamp
        header.frame_id = self._namespace
        tf_list = []

        for bb in bb_msg.bounding_boxes:
            xmin, ymin, xmax, ymax = bb.xmin, bb.ymin, bb.xmax, bb.ymax
            bbcen = np.asfarray([(xmin + xmax)/2, (ymin + ymax)/2])
            class_id = bb.class_id
            id = bb.id
            bb_im = im[ymin:ymax,xmin:xmax]
            bb_im_non_zero = bb_im[bb_im > 0]  # exclude zero extremes
            bb_im_mean = np.mean(bb_im_non_zero)
            bb_im_std = np.std(bb_im_non_zero)
            bb_im_dist = abs(bb_im - bb_im_mean)
            max_dev = 2
            bb_im_filtered = bb_im[bb_im_dist < max_dev*bb_im_std]
            avg_depth = np.mean(bb_im_filtered)

            # Get reprojection and scale from mm to m
            result = rs2.rs2_deproject_pixel_to_point(intrinsics, bbcen, avg_depth)

            bb_tf = TransformStamped()
            bb_tf.transform.translation.y = result[0]/1000.
            bb_tf.transform.translation.z = result[1]/1000.
            bb_tf.transform.translation.x = result[2]/1000.
            # detection_tf.transform.rotation =  # TODO: is there a way to get orientation of detection?
            bb_tf.child_frame_id = f"{self._namespace}/darknet/{class_id}"
            bb_tf.header = header
            tf_list.append(bb_tf)
            
            # Show excluded zone in output
            # bb_im[bb_im_dist >= max_dev*bb_im_std] = 10000
            # im[ymin:ymax,xmin:xmax] = bb_im
            # Debug mean of bounding boxes
            # self.get_logger().info(f"{np.mean(bb_im_filtered)}, {bb_im_mean}, {np.unique(bb_im)}")
            # Place rectangle in image
            # cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            # cv2.putText(im,class_id,(xmax+10,ymax),0,0.3,(0,255,0))
        self._br.sendTransform(tf_list)
        # marked_img = self.bridge.cv2_to_imgmsg(im,im_encoding)
        # marked_img.header.frame_id = "drone_0/realsense"
        # self._pub_bb_depth.publish(marked_img)


def main(args=None):
    rclpy.init(args=args) 
    exe = MultiThreadedExecutor()
    creator = BBTfCreator()
    rclpy.spin(creator, executor=exe)


if __name__=="__main__":
    # args = ["--ros-args", ]
    main()
