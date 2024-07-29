#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image


# In[3]:


# Load the trained model
model = load_model('skin_cancer_detection_model.h5')


# In[4]:


# Define the size of the input images
# img_size = (224, 224)


# In[5]:


def preprocess_image(img):
    img = np.array(img)  # Convert PIL image to NumPy array
    img = cv2.resize(img, (224, 224))  # Resize the image to the input size of the model
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# In[6]:


st.title('Skin Cancer Detection')

uploaded_file = st.file_uploader('Upload an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    st.write('')
    st.write('Classifying...')
    
    img = preprocess_image(img)
    prediction = model.predict(img)
    
    if prediction[0][0] > 0.5:
        result = 'Cancer'
    else:
        result = 'Non-cancer'
    
    st.write(f'The image is classified as: {result}')


# In[7]:


# # Run the app
# if __name__ == '__main__':
#     app()


# In[ ]:




