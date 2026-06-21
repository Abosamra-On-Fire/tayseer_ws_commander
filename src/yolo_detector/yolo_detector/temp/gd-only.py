import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDinoNode(Node):
    def __init__(self):
        super().__init__('gdino_node')

        # Parameters
        self.declare_parameter('text_prompt', 'ball .')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('input_image_topic', 'rgb')
        self.declare_parameter('output_image_topic', '/yolo/annotated_image')
        self.declare_parameter('output_detections_topic', '/yolo/detections')
        self.declare_parameter('device', 'auto')

        self.text_prompt = self.get_parameter('text_prompt').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.input_topic = self.get_parameter('input_image_topic').value
        out_img_topic = self.get_parameter('output_image_topic').value
        out_det_topic = self.get_parameter('output_detections_topic').value
        device_param = self.get_parameter('device').value

        if device_param == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_param

        self.get_logger().info(f'Using device: {self.device.upper()}')
        self.get_logger().info('Loading Grounding DINO Tiny (~200 MB)...')

        # --- Load Grounding DINO Tiny ---
        self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(self.device)

        self.get_logger().info('Model loaded. SAM removed.')

        self.bridge = CvBridge()

        # Publishers
        self.image_pub = self.create_publisher(Image, out_img_topic, 10)
        self.det_pub = self.create_publisher(Detection2DArray, out_det_topic, 10)

        # Subscriber
        self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )

        self.get_logger().info(f'Subscribed to topic: {self.input_topic}')
        self.get_logger().info(f'Prompt: "{self.text_prompt}"')

    def image_callback(self, msg):
        # Clear CUDA cache before inference (helps with fragmentation)
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Bridge error: {e}')
            return

        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # ========== Grounding DINO Inference ==========
        inputs = self.dino_processor(
            images=pil_image,
            text=self.text_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            target_sizes=[pil_image.size[::-1]]
        )[0]

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("text_labels", results.get("labels", []))

        # Prepare output messages
        det_array = Detection2DArray()
        det_array.header = msg.header
        annotated = cv_image.copy()

        if len(boxes) == 0:
            # Publish empty detection + original image
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            img_msg.header = msg.header
            self.image_pub.publish(img_msg)
            self.det_pub.publish(det_array)
            return

        # Draw boxes and populate Detection2DArray
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            conf = float(score)

            # Detection2D message
            det = Detection2D()
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(label)
            hyp.hypothesis.score = conf
            det.results.append(hyp)
            det_array.detections.append(det)

            # Draw on image (BGR: green box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} {conf:.2f}'
            cv2.putText(annotated, text, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish results
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        img_msg.header = msg.header
        self.image_pub.publish(img_msg)
        self.det_pub.publish(det_array)

        self.get_logger().debug(
            f'Published {len(det_array.detections)} detections: '
            f'{[d.results[0].hypothesis.class_id for d in det_array.detections]}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = GroundingDinoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


##################################### launch

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='detector_node',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'image_path': '/home/ubuntu/tayseer_ws/src/yolo_detector/yolo_detector/image.jpg',
                'text_prompt': 'ball . cube .',      # <-- change to whatever you want to detect
                'box_threshold': 0.3,
                'text_threshold': 0.25,
                'output_image_topic': '/yolo/annotated_image',
                'output_detections_topic': '/yolo/detections',
                'publish_rate_hz': 1.0,
                'device': 'cpu',
            }]
        )
    ])