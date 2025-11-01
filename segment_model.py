import torch
import torchvision
from PIL import Image
import numpy as np
import cv2
import time

class ImageSegmenter:
    def __init__(self, threshold=0.7):
        # using GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").to(self.device)
        self.model.eval()
        self.threshold = threshold
        print("[Model Loaded Successfully for Segmentation]")

    def segment_image(self, image_path):
        # it is main function for segmentation
        try:
            start_time = time.time()
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            img_tensor = torchvision.transforms.ToTensor()(image).to(self.device)

            with torch.no_grad():
                outputs = self.model([img_tensor])[0]

            masks = outputs["masks"].cpu().numpy()[:, 0, :, :]
            boxes = outputs["boxes"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()

            overlay = image_np.copy()

            for i, score in enumerate(scores):
                if score < self.threshold:
                    continue

                mask = cv2.resize(masks[i], (image_np.shape[1], image_np.shape[0])) > self.threshold
                color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                color_mask = np.zeros_like(overlay, dtype=np.uint8)
                color_mask[mask] = color

                overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
                cv2.putText(overlay, f"Object {labels[i]} ({score:.2f})", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

            print(f"[Segmentation Completed in {time.time() - start_time:.2f}s]")
            return overlay

        except Exception as e:
            print("[Error] Segmentation failed:", e)
            return np.array(Image.open(image_path))

# done by Sarthak Maddi for zidio internship
