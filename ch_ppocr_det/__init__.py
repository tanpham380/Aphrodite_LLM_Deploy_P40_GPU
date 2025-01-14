# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List, Optional, Tuple

import numpy as np
from .text_detect import TextDetector



# text_det = TextDetector()
# def auto_text_det(img: np.ndarray) -> Tuple[Optional[List[np.ndarray]], float]:
#     dt_boxes, det_elapse = text_det(img)
#     if dt_boxes is None or len(dt_boxes) < 1:
#         return None, 0.0
#     return dt_boxes, det_elapse