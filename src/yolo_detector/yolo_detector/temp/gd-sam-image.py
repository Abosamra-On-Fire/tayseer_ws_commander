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
import os


class GroundingDinoSamNode(Node):
    def __init__(self):
        super().__init__('gdino_sam_node')

        # Parameters
        self.declare_parameter('image_path', 'image.jpg')
        self.declare_parameter('text_prompt', 'ball . cube . bottle .')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('output_image_topic', '/yolo/annotated_image')
        self.declare_parameter('output_detections_topic', '/yolo/detections')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('device', 'auto')

        image_path = self.get_parameter('image_path').value
        self.text_prompt = self.get_parameter('text_prompt').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        out_img_topic = self.get_parameter('output_image_topic').value
        out_det_topic = self.get_parameter('output_detections_topic').value
        publish_rate = self.get_parameter('publish_rate_hz').value
        device_param = self.get_parameter('device').value

        self.image_path = os.path.abspath(os.path.expanduser(image_path))

        if device_param == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_param

        self.get_logger().info(f'Using device: {self.device.upper()}')
        self.get_logger().info('Loading Grounding DINO (first run downloads ~700 MB)...')

        # --- Load Grounding DINO ---
        self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
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

        # Timer
        self.timer = self.create_timer(1.0 / publish_rate, self.process_and_publish)

        self.get_logger().info(f'Image path: {self.image_path}')
        self.get_logger().info(f'Prompt: "{self.text_prompt}"')

    def process_and_publish(self):
        if not os.path.exists(self.image_path):
            self.get_logger().error(f'Image not found: {self.image_path}')
            return

        cv_image = cv2.imread(self.image_path)
        if cv_image is None:
            self.get_logger().error(f'Failed to load image: {self.image_path}')
            return

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

        header_stamp = self.get_clock().now().to_msg()
        frame_id = 'camera'

        if len(boxes) == 0:
            self.get_logger().info('No objects detected')
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            msg.header.stamp = header_stamp
            msg.header.frame_id = frame_id
            self.image_pub.publish(msg)
            return

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
        )[0].numpy()  # shape: (N, num_masks_per_box, H, W)

        # ========== Prepare Messages & Annotate ==========
        det_array = Detection2DArray()
        det_array.header.stamp = header_stamp
        det_array.header.frame_id = frame_id

        annotated = cv_image.copy()

        # ✅ FIX: Handle SAM's 3-mask output & safely apply overlay
        for mask in masks:
            # SAM returns 3 masks per box (low, medium, high quality).
            # We pick the medium one (index 1) for best balance.
            if mask.ndim == 3:
                mask = mask[1] if mask.shape[0] > 1 else mask[0]
            mask = mask.astype(bool)
            
            # Create green overlay
            green_mask = np.zeros_like(annotated, dtype=np.uint8)
            green_mask[mask] = [0, 255, 0]  # BGR green
            
            # Blend smoothly
            annotated = cv2.addWeighted(annotated, 0.6, green_mask, 0.4, 0)

        # Boxes, labels, Detection2DArray
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            conf = float(score)

            det = Detection2D()
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = float(x2 - x1)  
            det.bbox.size_y = float(y2 - y1)
            
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = conf
            det.results.append(hyp)
            det_array.detections.append(det)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} {conf:.2f}'
            cv2.putText(annotated, text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        img_msg.header = det_array.header
        self.image_pub.publish(img_msg)
        self.det_pub.publish(det_array)

        self.get_logger().info(
            f'Detected {len(det_array.detections)} objects: '
            f'{[d.results[0].hypothesis.class_id for d in det_array.detections]}'
        )


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