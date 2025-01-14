import base64
import json
import os
from pathlib import Path
import time
import logging
import gradio as gr
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import jsonify
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImageAnalyzer")

# Prompt for the AI
PROMPT = """Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
Trích xuất thông tin kiện hàng trong ảnh và trả về dạng JSON.
"""

GENERATION_CONFIG = {
              "temperature": 0.01,
                "top_p": 0.1,
                "min_p": 0.1,
                "top_k": 1,
                "max_tokens": 1024,
                "repetition_penalty": 1.1,
                "best_of": 5
}

class ImageAnalyzer:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def encode_image(self, image_path: str) -> str:
        """Encode the image file to a Base64 string."""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            logger.info("Image encoded successfully.")
            return encoded_image
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_image(self, image_path: str) -> str:
        """Analyze the given image and return extracted JSON data."""
        image_base64 = self.encode_image(image_path)
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="erax-ai/EraX-VL-7B-V1.5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ],
                extra_body=GENERATION_CONFIG
            )
            print(response)
            end_time = time.time()
            
            # Create response dictionary
            response_data = {
                "content": response.choices[0].message.content,
                "metadata": {
                    "model": response.model,
                    "created": response.created,
                    "response_time": f"{end_time - start_time:.2f}",
                    "tokens": {
                        "total": response.usage.total_tokens,
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens
                    }
                }
            }
            return json.dumps(response_data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

def analyze_image_gradio(image) -> str:
    """
    Wrapper for Gradio demo that handles image analysis.

    Args:
        image: Image input from Gradio interface (can be filepath or numpy array)

    Returns:
        str: JSON response with extracted information or error message
    """
    analyzer = ImageAnalyzer(api_key="sk-empty", api_base="http://localhost:2242/v1")

    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        temp_image_path = temp_dir / "temp_uploaded_image.jpg"

        # Handle different image input types
        if isinstance(image, str):
            # If image is already a filepath
            temp_image_path = Path(image)
        else:
            # If image is numpy array from Gradio
            import cv2
            import numpy as np
            if isinstance(image, np.ndarray):
                cv2.imwrite(str(temp_image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                raise ValueError("Unsupported image format")

        # Perform analysis
        result = analyzer.analyze_image(str(temp_image_path))

        # Clean up if we created the temp file
        if temp_image_path.parent == temp_dir:
            temp_image_path.unlink(missing_ok=True)

        return result
    except Exception as e:
        logger.error(f"Error in Gradio interface: {e}", exc_info=True)
        return f"Error: {str(e)}"
    finally:
        # Cleanup temp directory if empty
        try:
            temp_dir.rmdir()
        except (OSError, FileNotFoundError):
            pass

if __name__ == "__main__":
    # Gradio Interface
    interface = gr.Interface(
        fn=analyze_image_gradio,
        inputs=gr.Image(label="Upload Image"),  # Remove type="filepath"
        outputs=gr.Textbox(),
        title="OCR for Citizen ID",
        description="Upload an image of a Citizen ID (front side) and extract structured information in JSON format."
    )

    interface.launch(server_name="0.0.0.0")



# curl -X POST https://api.openai.com/v1/completions \
#   -H "Authorization: Bearer YOUR_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "internllm2.5",
#     "prompt": "Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.\nBạn được cung cấp 1 ảnh mặt trước của 1 căn cước công dân hợp pháp, không vi phạm. Có thể có nhiều phiên bản khác nhau của căn cước công dân.\nBạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json",

#     "stop": null,
#     "extra_body": {
#     "max_tokens": 1024,
#     "temperature": 0.0,
#       "max_new_tokens": 1024,
#       "do_sample": false,
#       "num_beams": 3,
#       "repetition_penalty": 1.5
#     },
#     "image_url": "https://example.com/path/to/your/image.jpg"
#   }'


# curl -X POST http://192.168.1.136:2243/v1/completions \
#   -H "Authorization: Bearer YOUR_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "5CD-AI/Vintern-1B-v3_5",
#     "prompt": "Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.\nBạn được cung cấp 1 ảnh mặt trước của 1 căn cước công dân hợp pháp, không vi phạm. Có thể có nhiều phiên bản khác nhau của căn cước công dân.\nBạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json",
#     "image_url": "https://huggingface.co/erax-ai/EraX-VL-7B-V1.5/resolve/main/images/trinhquangduy_front.jpg",
#     "extra_body": {
#       "max_new_tokens": 1024,
#       "do_sample": false,
#       "num_beams": 3,
#       "repetition_penalty": 2.5,
#       "temperature": 0.01,
#       "top_p": 0.1,
#       "top_k": 1
#     }
#   }'
