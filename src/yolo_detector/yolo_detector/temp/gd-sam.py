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
from transformers import SamModel, SamProcessor

class GroundingDinoSamNode(Node):
    def __init__(self):
        super().__init__('gdino_sam_node')

        # Parameters
        self.declare_parameter('text_prompt', 'ball .')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('input_image_topic', 'rgb')  # New parameter for input topic
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
        self.get_logger().info('Loading Grounding DINO (first run downloads ~700 MB)...')

        # --- Load Grounding DINO ---
        self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(self.device)

        self.get_logger().info('Loading SAM (first run downloads ~375 MB)...')
        # --- Load SAM ---
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)

        self.get_logger().info('Models loaded.')

        self.bridge = CvBridge()

        # Publishers
        self.image_pub = self.create_publisher(Image, out_img_topic, 10)
        self.det_pub = self.create_publisher(Detection2DArray, out_det_topic, 10)

        # ✅ Subscriber: Listens to 'rgb' (or configured topic)
        self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10  # Queue size
        )

        self.get_logger().info(f'Subscribed to topic: {self.input_topic}')
        self.get_logger().info(f'Prompt: "{self.text_prompt}"')

    def image_callback(self, msg):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        try:
            # Convert ROS Image to OpenCV (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Bridge error: {e}')
            return

        # Convert to PIL for Transformers
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # ========== Grounding DINO ==========
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

        if len(boxes) == 0:
            return # No objects found

        # ========== SAM ==========
        boxes_np = boxes.cpu().numpy()
        sam_inputs = self.sam_processor(
            pil_image,
            input_boxes=[boxes_np.tolist()],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )[0].numpy()

        # ========== Prepare Messages & Annotate ==========
        det_array = Detection2DArray()
        det_array.header = msg.header

        annotated = cv_image.copy()

        # Apply Masks
        for mask in masks:
            if mask.ndim == 3:
                mask = mask[1] if mask.shape[0] > 1 else mask[0]
            mask = mask.astype(bool)
            
            green_mask = np.zeros_like(annotated, dtype=np.uint8)
            green_mask[mask] = [0, 255, 0]
            annotated = cv2.addWeighted(annotated, 0.6, green_mask, 0.4, 0)

        # Boxes & Labels
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            conf = float(score)

            det = Detection2D()
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(label)  # ✅ Ensure string type
            hyp.hypothesis.score = conf
            det.results.append(hyp)
            det_array.detections.append(det)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} {conf:.2f}'
            cv2.putText(annotated, text, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish results
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        img_msg.header = msg.header
        self.image_pub.publish(img_msg)
        self.det_pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    node = GroundingDinoSamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


    