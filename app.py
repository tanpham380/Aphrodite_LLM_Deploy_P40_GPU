import base64
import os
import time
import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
prompt =  """Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
        Bạn được cung cấp 1 ảnh mặt trước của 1 căn cước công dân hợp pháp, không vi phạm. Có thể có nhiều phiên bản khác nhau của căn cước công dân. 
        Bạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json.
        Trả lại kết quả OCR của tất cả thông tin 1 JSON duy nhất
                Return JSON with these fields:
{{
    "id_number": "",
    "fullname": "",
    "day_of_birth": "",
    "sex": "",
    "nationality": "",
    "place_of_residence": "",
    "place_of_origin": "",
    "date_of_expiration": "",
    "date_of_issue": "",
    "place_of_issue": ""
}}
"""
class ImageAnalyzer:
    def __init__(self):
        self.api_key = "sk-empty"
        self.api_base = "http://localhost:2242/v1"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def encode_image(self, image_path: str) -> str:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_image(self, image_path: str) -> str:
        image_base64 = self.encode_image(image_path)
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="erax-ai/EraX-VL-7B-V1.5",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ]
            )
            end_time = time.time()
            
            # Log response metadata
            logger.info(f"Response metadata:")
            logger.info(f"Model: {response.model}")
            logger.info(f"Created: {response.created}")
            logger.info(f"Response time: {end_time - start_time:.2f} seconds")
            logger.info(f"Total tokens: {response.usage.total_tokens}")
            logger.info(f"Prompt tokens: {response.usage.prompt_tokens}")
            logger.info(f"Completion tokens: {response.usage.completion_tokens}")
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

if __name__ == "__main__":
    analyzer = ImageAnalyzer()
    image_path = os.path.join(os.path.dirname(__file__), "test23.jpg")
    result = analyzer.analyze_image(image_path)
    print(f"Analysis result: {result}")
    
