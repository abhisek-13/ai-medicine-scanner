from paddleocr import PaddleOCR
import tempfile
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os

def image_text_extractor(arr):
  image = Image.fromarray(arr)
  
  with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
    temp_filepath = temp_file.name
    image.save(temp_filepath, format="JPEG")
    
  ocr = PaddleOCR(lang = 'en')
  result = ocr.ocr(temp_filepath)
  print(result)
  if result != [None]:
    result_text = "\n".join(i[1][0] for i in result[0])
    return result_text
  else:
    return None


def image_analysis(arr):
  image = Image.fromarray(arr)
  with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
    temp_filepath = temp_file.name
    image.save(temp_filepath, format="JPEG")

  # Initialize OCR engine
  ocr = PaddleOCR(use_angle_cls=True, lang="en")

  slice = {'horizontal_stride': 300, 'vertical_stride': 500, 'merge_x_thres': 50, 'merge_y_thres': 35}
  results = ocr.ocr(temp_filepath, cls=True, slice=slice)

  # Load image
  image = Image.open(temp_filepath).convert("RGB")
  draw = ImageDraw.Draw(image)
  font = ImageFont.truetype("latin.ttf", size=10)  # Adjust size as needed

  # Process and draw results
  for res in results:
    for line in res:
      box = [tuple(point) for point in line[0]]
          # Finding the bounding box
      box = [(min(point[0] for point in box), min(point[1] for point in box)),
                (max(point[0] for point in box), max(point[1] for point in box))]
      txt = line[1][0]
      draw.rectangle(box, outline="red", width=2)  # Draw rectangle
      draw.text((box[0][0], box[0][1] - 25), txt, fill="yellow", font=font)  # Draw text above the box
  with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
    temp_file_path = temp_file.name
    image.save(temp_file_path, format="JPEG")
    
  ocr_image = Image.open(temp_file_path)
  image_arr = np.array(ocr_image)
    
  return image_arr