#!/usr/bin/python3

import time
import os
from collections import Counter
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import Image

from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
ALREADY_SAVED = False
qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)




# morphological operation to cleanup mask 
# could be better, but it works for now
def cleanup_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    # kernel = np.array([
    #     [0, 0, 1, 0, 0],
    #     [0, 1, 0, 1, 0],
    #     [1, 0, 0, 0, 1],
    #     [0, 1, 0, 1, 0],
    #     [0, 0, 1, 0, 0]], dtype=np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    return dilation

def predict_color(ring_img, mask=None):
    if mask is None or np.sum(mask) == 0:
        return ""
    mask = cleanup_mask(mask)


    # Create histograms folder if it doesn't exist
    hist_folder = "/home/gamma/colcon_ws/avg_hist"
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)
    
    # Ensure mask matches image dimensions
    if mask.shape[:2] != ring_img.shape[:2]:
        mask = cv2.resize(mask, (ring_img.shape[1], ring_img.shape[0]))
    
    # print("Ring image pixel:", ring_img[29][107])
    
    hsv = cv2.cvtColor(ring_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply mask to get only masked values
    h_masked = h[mask > 0]
    s_masked = s[mask > 0]
    v_masked = v[mask > 0]
    
    # print("HSV (masked):", np.mean(h_masked), np.mean(s_masked), np.mean(v_masked))
    if np.mean(v_masked) < 140:
        return "black"
    
    hist = cv2.calcHist([h], [0], mask, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten() 


    histograms = os.listdir(hist_folder)
    best_match = None
    second_best = None 

    best_score = float('inf')
    second_best = float('inf')
    for hist_file in histograms:
        if hist_file.endswith(".npy"):
            hist_path = os.path.join(hist_folder, hist_file)
            hist_data = np.load(hist_path)
            score = np.sqrt(0.5 * np.sum((np.sqrt(hist) - np.sqrt(hist_data))**2))
            color_name = hist_file.split(".")[0]
            # print(f"Comparing to {color_name}, score: {score}")
            if score < best_score:
                second_best = best_score
                best_score = score
                best_match = color_name
            elif score < second_best:
                second_best = score

    iowe_ratio = second_best / best_score if second_best != float('inf') else 1.0
    print(f"Best match: {best_match} with score {best_score}, IOWE ratio: {iowe_ratio}")
    colors = ['red', 'green', 'blue']
    for color in colors:
        if best_match.startswith(color) and iowe_ratio > 1.2:
            return color
    return ""


class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 5
        timer_period = 1/timer_frequency

        # ellipse thresholds
        self.ecc_thr = 100
        self.ratio_thr = 1.5
        self.center_thr = 10
        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()
        self.rings = None
        self.circles = None
        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.image_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, "/top_camera/rgb/preview/depth", self.depth_callback, qos_profile_sensor_data)
        self.points_sub = self.create_subscription(PointCloud2, "/top_camera/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)
        

        self.color_pub = self.create_publisher(String, "/ring_color", QoSReliabilityPolicy.BEST_EFFORT)
        self.ring_pub = self.create_publisher(MarkerArray, "/rings", QoSReliabilityPolicy.BEST_EFFORT)
        self.confirmed_ring_count = 0
        self.last_color = ""
        self.position_buffer = []
        self.color_buffer = []
        
        #cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)        

    def image_callback(self, data):
        if not hasattr(self, 'rings') or self.rings is None:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        color = predict_color(cv_image, self.rings)
        if color != "":
            self.last_color = color
            color_msg = String()
            color_msg.data = color
            print("Publishing color: ", color)
            self.color_pub.publish(color_msg)
        
        cv2.imshow("Mask", self.rings)
        cv2.imshow("cv_image ", cv_image)


    def depth_callback(self,data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        # # Save depth image only once
        # if not hasattr(self, 'depth_saved') or not self.depth_saved:
        #     # Normalize for visualization and save
        #     depth_normalized = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
        #     cv2.imwrite('/home/gamma/colcon_ws/depth_image.png', depth_normalized)
        #     print("✓ Depth image saved to /home/gamma/colcon_ws/depth_image.png")
        #     self.depth_saved = True
        self.rings = None
        depth_image[depth_image==np.inf] = 0
        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 = image_1/np.max(image_1)*255
        #image_1 = 255 - image_1
        image_copy = np.array(image_1, dtype=np.uint8)
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        self.circles = cv2.HoughCircles(image_1.astype(np.uint8),cv2.HOUGH_GRADIENT,1,20,param1=70,param2=20,minRadius=1,maxRadius=30)
        if self.circles is not None:
            self.circles = np.uint16(np.around(self.circles))
            for i in self.circles[0,:]:
                # za berljivost
                x_center = i[0]
                y_center = i[1]
                radius = i[2]

                # draw the outer circle
                cv2.circle(image_1,(x_center,y_center),radius,(0,255,0),2)
                # draw the center of the circle
                cv2.circle(image_1,center=(x_center,y_center),radius=2,color=(0,0,255),thickness=3)

                cv2.circle(mask,(x_center,y_center),radius+5,255,-1)
                cv2.circle(mask,(x_center,y_center),max(0,radius-2),0,-1)


            mask = cleanup_mask(mask)

            self.rings = mask.copy()

        #print("Circles found:", self.circles)
        image_viz = np.array(image_1, dtype= np.uint8)
        mask_viz = np.array(mask, dtype= np.uint8)
        cv2.imshow("Depth window", image_viz)
        #cv2.imshow("Mask window", mask_viz)
        cv2.waitKey(1)

    def pointcloud_callback(self, data):
        if not hasattr(self, 'rings') or self.rings is None or np.sum(self.rings)==0:
            return
        # get point cloud attributes
        point_rings = self.rings.copy()
        # print("Processing point cloud...")
        height = data.height
        width = data.width

        a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
        a = a.reshape((height,width,3))

        rings_found = []
        max_range = 255
        for x in range(point_rings.shape[1]):
            for y in range(point_rings.shape[0]):
                if point_rings[y,x]>0:
                    d = a[y,x,:]
                    #print("Ring found at (x,y,z):", d)
                    rings_found.append(d)
        #print("Total rings found:", len(rings_found))
        # print("Total rings:",len(point_rings))
        # remove outliers
        # basic outlier removal on x
        rings_found = [ring for ring in rings_found if np.abs(ring[0]) <= self.center_thr]
        if not rings_found:
            return

        rings_array = np.array(rings_found)

        # Only process rings closer than 1.5 m
        distances = np.linalg.norm(rings_array, axis=1)
        rings_array = rings_array[distances < 1.5]
        if len(rings_array) == 0:
            return

        center = rings_array.mean(axis=0)

        # Accumulate until we have 100 samples, then commit one averaged marker
        self.position_buffer.append(center)
        self.color_buffer.append(self.last_color)

        if len(self.position_buffer) < 100:
            return

        avg_center = np.mean(self.position_buffer, axis=0)
        color_counts = Counter(c for c in self.color_buffer if c != "")
        detected_color = color_counts.most_common(1)[0][0] if color_counts else ""
        self.position_buffer = []
        self.color_buffer = []

        color_map = {
            "red":   (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue":  (0.0, 0.0, 1.0),
        }
        r, g, b = color_map.get(detected_color, (1.0, 1.0, 1.0))

        self.confirmed_ring_count += 1
        ring_label = f"ring {self.confirmed_ring_count}"

        sphere = Marker()
        sphere.header.frame_id = "/base_link"
        sphere.header.stamp = data.header.stamp
        sphere.ns = "rings"
        sphere.id = self.confirmed_ring_count * 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(avg_center[0])
        sphere.pose.position.y = float(avg_center[1])
        sphere.pose.position.z = float(avg_center[2])
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
        sphere.color.r = r
        sphere.color.g = g
        sphere.color.b = b
        sphere.color.a = 1.0

        label = Marker()
        label.header.frame_id = "/base_link"
        label.header.stamp = data.header.stamp
        label.ns = "ring_labels"
        label.id = self.confirmed_ring_count * 2 + 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(avg_center[0])
        label.pose.position.y = float(avg_center[1])
        label.pose.position.z = float(avg_center[2]) + 0.2
        label.pose.orientation.w = 1.0
        label.scale.z = 0.15
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = ring_label

        marker_array = MarkerArray()
        marker_array.markers = [sphere, label]
        print(f"Confirmed {ring_label} at {avg_center}, color={detected_color}")
        self.ring_pub.publish(marker_array)
def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()