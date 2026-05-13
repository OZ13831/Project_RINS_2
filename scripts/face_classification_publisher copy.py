#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class FaceClassificationPublisher(Node):
    def __init__(self):
        super().__init__("face_classification_publisher")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topic", "/oakd/rgb/preview/image_raw"),
                ("publish_topic", "/face_class"),
                ("detector_model", "yolov8n.pt"),
                ("classifier_model", "/home/gamma/colcon_ws/face_classification/runs/classify/train/weights/best.pt"),
                ("device", ""),
            ],
        )

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.publish_topic = self.get_parameter("publish_topic").get_parameter_value().string_value
        self.detector_model_path = self.get_parameter("detector_model").get_parameter_value().string_value
        self.classifier_model_path = self.get_parameter("classifier_model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.detector = YOLO(self.detector_model_path)
        self.classifier = YOLO(self.classifier_model_path)

        self.processing = False

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.class_pub = self.create_publisher(String, self.publish_topic, 10)

    def image_callback(self, msg: Image) -> None:
        if self.processing:
            return

        self.processing = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.processing = False
            return

        try:
            det_results = self.detector.predict(
                source=cv_image, classes=[0], verbose=False, device=self.device
            )

            for box in det_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = cv_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_results = self.classifier.predict(
                    source=crop, verbose=False, device=self.device
                )
                probs = cls_results[0].probs
                if probs is None:
                    continue

                top_class_id = probs.top1
                top_class_name = cls_results[0].names[top_class_id]

                out_msg = String()
                out_msg.data = top_class_name
                self.class_pub.publish(out_msg)

                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(cv_image, (cx, cy), 4, (0, 255, 0), -1)
                label = top_class_name
                cv2.putText(
                    cv_image,
                    label,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("face_classification", cv_image)
            cv2.waitKey(1)
        finally:
            self.processing = False


def main() -> None:
    rclpy.init(args=None)
    node = FaceClassificationPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
