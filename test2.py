import cv2
import numpy as np
from pathlib import Path

from ch_ppocr_det.text_detect import TextDetector

def extract_and_place_on_white(image_path, boxes):
    # Read image
    img = cv2.imread(str(image_path))

    # Get image dimensions
    img_h, img_w, _ = img.shape

    # Create a white background of the same size as the original image
    white_bg = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # Process each box
    for box in boxes:
        # Convert coordinates to integers
        box = box.astype(np.int32)

        # Get bounding rectangle of the polygon
        x, y, w, h = cv2.boundingRect(box)

        # Extract the region of interest (ROI) from the original image
        roi = img[y:y+h, x:x+w]

        # Create a mask for the polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [box - [x, y]], 255)

        # Apply the mask to the ROI
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

        # Place the masked ROI onto the white background
        white_bg[y:y+h, x:x+w] = cv2.addWeighted(white_bg[y:y+h, x:x+w], 0, roi_masked, 1, 0)

    # Save the resulting image
    output_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_processed.jpg"))
    cv2.imwrite(output_path, white_bg)
    return output_path

# Main execution
img_path = Path("test.jpg")

# Example detector
text_det = TextDetector(
    model_path="models/detect_text.onnx",
    config={
        "use_cuda": True,
        "gpu_id": 0  # Use first GPU
    }
)

# Detect boxes
dt_boxes, det_elapse = text_det(img_path)

# Generate the result with white background
output_path = extract_and_place_on_white(img_path, dt_boxes)

print(f"Detection time: {det_elapse:.3f}s")
print(f"Output with white background saved to: {output_path}")