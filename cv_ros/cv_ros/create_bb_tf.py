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

from ros2_utils import convert_axes_from_msg, AxesFrame

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped


class BBTfCreator(Node):
    def __init__(self):
        super().__init__("bb_tf_creator")
        self._namespace = self.get_namespace().split("/")[-1]

        # Parameters
        self.declare_parameter("max_dev", 2)
        self.declare_parameter("fps", 10.0)  # FPS assumption for time delay between matching messages
        self.declare_parameter("prob_thresh", 0.7)  # Threshold to ignore bounding box

        self._fps = self.get_parameter("fps").value
        self._depth_info = None
        self._rgb_info = None

        self.bridge = CvBridge()

        # Subscribers
        self._sub_bb = Subscriber(self, BoundingBoxes, "darknet_ros/bounding_boxes", qos_profile=10)
        self._sub_depth = Subscriber(self, Image, "realsense/aligned_depth_to_color/image_raw", qos_profile=1)
        self._ts = ApproximateTimeSynchronizer([self._sub_bb, self._sub_depth], 2, 1.0/self._fps)
        self._ts.registerCallback(self._cb_ts)
        self._sub_depth_info = self.create_subscription(CameraInfo, "realsense/aligned_depth_to_color/camera_info", self._cb_depth_info, 1)
        self._sub_rgb_info = self.create_subscription(CameraInfo, "realsense/color/camera_info", self._cb_rgb_info, 1)

        # Publishers
        self._pub_bb_depth = self.create_publisher(Image, "bb_depth", 1)

        # TF
        self._br = TransformBroadcaster(self)

        self.get_logger().info("Initialized BBTFCreator!")

    def _cb_depth_info(self, msg: CameraInfo):
        self._depth_info = msg
        if not self.destroy_subscription(self._sub_depth_info):
            self.get_logger().error("Couldn't stop camera info subscription")
            return
        del self._sub_depth_info

    def _cb_rgb_info(self, msg: CameraInfo):
        self._rgb_info = msg
        if not self.destroy_subscription(self._sub_rgb_info):
            self.get_logger().error("Couldn't stop camera info subscription")
            return
        del self._sub_rgb_info

    def _cb_ts(self, bb_msg: BoundingBoxes, depth_msg: Image):
        if self._depth_info is None or self._rgb_info is None:
            self.get_logger().warn("Camera info not set yet")
            return
        # self.get_logger().info(f"{bb_msg.header.stamp}, {depth_msg.header.stamp}")
        depth_info = self._depth_info
        rgb_info = self._rgb_info
        max_dev = self.get_parameter("max_dev").value

        im_encoding = "passthrough"
        im = self.bridge.imgmsg_to_cv2(depth_msg,im_encoding)

        intrinsics = rs2.intrinsics()
        intrinsics.width = depth_info.width
        intrinsics.height = depth_info.height
        intrinsics.ppx = depth_info.k[2]
        intrinsics.ppy = depth_info.k[5]
        intrinsics.fx = depth_info.k[0]
        intrinsics.fy = depth_info.k[4]
        intrinsics.model = rs2.distortion.none
        if depth_info.d: intrinsics.coeffs = [float(i) for i in depth_info.d]

        header = Header()
        header.stamp = self.get_clock().now().to_msg() # bb_msg.header.stamp
        header.frame_id = f"{self._namespace}/realsense"
        tf_list = []

        for bb in bb_msg.bounding_boxes:
            if bb.probability <= self.prob_thresh:
                continue
            if depth_info.width != rgb_info.width and depth_info.height != rgb_info.height:
                # Scale xmin from color image bounding box to depth image
                xmin = int(np.ceil((bb.xmin/rgb_info.width)*depth_info.width))
                xmax = int(np.ceil((bb.xmax/rgb_info.width)*depth_info.width))
                ymin = int(np.ceil((bb.ymin/rgb_info.height)*depth_info.height))
                ymax = int(np.ceil((bb.ymax/rgb_info.height)*depth_info.height))
            else:
                xmin, ymin, xmax, ymax = bb.xmin, bb.ymin, bb.xmax, bb.ymax
            # self.get_logger().info(f"{ymin} {ymax} {xmin} {xmax}")
            bbcen = np.asfarray([(xmin + xmax)/2, (ymin + ymax)/2])
            class_id = bb.class_id.replace(" ", "_")
            _id = bb.id
            bb_im = im[ymin:ymax,xmin:xmax]
            bb_im_non_zero = bb_im[bb_im > 0]  # exclude zero extremes
            bb_im_mean = np.mean(bb_im_non_zero)
            bb_im_std = np.std(bb_im_non_zero)
            bb_im_dist = abs(bb_im - bb_im_mean)
            bb_im_filtered = bb_im[bb_im_dist < max_dev*bb_im_std]
            avg_depth = np.mean(bb_im_filtered)

            # Get reprojection and scale from mm to m
            result = rs2.rs2_deproject_pixel_to_point(intrinsics, bbcen, avg_depth)

            bb_tf = TransformStamped()
            bb_tf.transform.translation.y = result[0]/1000.
            bb_tf.transform.translation.z = result[1]/1000.
            bb_tf.transform.translation.x = result[2]/1000.
            # detection_tf.transform.rotation =  # TODO: is there a way to get orientation of detection?
            bb_tf.transform = convert_axes_from_msg(bb_tf.transform, AxesFrame.URHAND, AxesFrame.RHAND)
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

    @property
    def prob_thresh(self):
        return self.get_parameter("prob_thresh").value


def main(args=None):
    rclpy.init(args=args) 
    exe = MultiThreadedExecutor()
    creator = BBTfCreator()
    rclpy.spin(creator, executor=exe)


if __name__=="__main__":
    main()
