import streamlit as st
import os
import base64
import hashlib

@st.cache_resource
def load_static_assets():
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    github_logo = img_to_base64(os.path.join(static_dir, 'github-logo.png'))
    linkedin_logo = img_to_base64(os.path.join(static_dir, 'linkedin-logo.png'))
    instagram_logo = img_to_base64(os.path.join(static_dir, 'instagram-logo.png'))
    
    return github_logo, linkedin_logo, instagram_logo

def image_to_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()


