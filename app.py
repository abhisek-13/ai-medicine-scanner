import streamlit as st
import tempfile
import numpy as np
from PIL import Image
from src.ocr_recog import image_text_extractor, image_analysis
from src.text_gen import text_generator
# Streamlit app
def main():
    st.title("AI Powered Medicine Scanner")
    st.warning('Important: This web app is for educational purposes only. Always consult a doctor before using any medicine. Do not rely on the information here to make medical decisions.', icon="⚠️")
    enable = st.checkbox("Enable camera")
    if enable:
      picture = st.camera_input("Take a picture", disabled=not enable)
    
      if picture:
        
        image = Image.open(picture)
        image_arr = np.array(image)
  
        
        text = image_text_extractor(image_arr)
        if text == None:
          st.warning("The image is not recognizable. Please ensure proper lighting is present. Clear the photo and Try again.", icon="⚠️")
        else:   
          ocr_image_arr = image_analysis(image_arr)
          ocr_image = Image.fromarray(ocr_image_arr)
          final_op = text_generator(text)
        
          st.header('Image Analysis:')
          st.image(ocr_image)
          st.markdown(final_op)
          

if __name__ == "__main__":
    main()