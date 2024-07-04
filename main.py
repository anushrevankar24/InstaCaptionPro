import streamlit as st
import os
from PIL import Image
from models import load_vocab_and_models, generate_image_description
from utils import image_to_hash
import config
import json
from vocabulary import Vocabulary

st.set_page_config(**config.PAGE_CONFIG)

with st.spinner("Loading models and assets..."):
    try:
        vocab, encoder, decoder, device, model = load_vocab_and_models()
    except Exception as e:
        st.error(f"Error loading models and assets: {str(e)}")
        st.stop()

st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #FF7518;'>AI Instagram Caption Generator</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; font-size: 22    px;'>Create captivating Instagram captions in seconds</p>", unsafe_allow_html=True)


if 'description' not in st.session_state:
    st.session_state.description = ''
if 'tone' not in st.session_state:
    st.session_state.tone = 'Formal'
if 'add_hashtags' not in st.session_state:
    st.session_state.add_hashtags = True
if 'add_emojis' not in st.session_state:
    st.session_state.add_emojis= True
if 'no_of_captions' not in st.session_state:
    st.session_state.no_of_captions = 5
if 'current_image_hash' not in st.session_state:
    st.session_state.current_image_hash = None
if 'current_image_description' not in st.session_state:
    st.session_state.current_image_description = None

def resize_image(image, max_width,max_height):
    
    img_width, img_height = image.size
    aspect_ratio = img_width / img_height
    frame_ratio = max_width / max_height
    if aspect_ratio > frame_ratio:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    return image.resize((new_width, new_height), Image.LANCZOS)

MAX_WIDTH, MAX_HEIGHT = 600,450

with st.container():
    st.markdown("<div class='column'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Caption this image...", type=['jpg', 'png', 'gif'])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            resized_image = resize_image(image, MAX_WIDTH, MAX_HEIGHT)
            background = Image.new('RGB', (MAX_WIDTH, MAX_HEIGHT), (255, 255, 255))
            paste_position = ((MAX_WIDTH - resized_image.width) // 2,
                              (MAX_HEIGHT - resized_image.height) // 2)
            background.paste(resized_image, paste_position)
            st.image(background, use_column_width=False)
            image_hash = image_to_hash(image)
            if st.session_state.current_image_hash != image_hash:
                st.session_state.current_image_hash = image_hash
                st.session_state.current_image_description = None
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    st.session_state.description = st.text_area("Description (optional)", st.session_state.description)
    col1, col2 = st.columns(2)
    with col1:
       st.session_state.tone = st.selectbox("Tone", ["Friendly","Formal", "Casual", "Attitude", "Luxury", "Excited", "Inspiring"], index=["Friendly","Formal", "Casual", "Attitude", "Luxury", "Excited", "Inspiring"].index(st.session_state.tone))
    with col2:
       st.session_state.no_of_captions= st.selectbox("No of Captions", [5, 3, 1], index=[5, 3, 1].index(st.session_state.no_of_captions))
    col3, col4 = st.columns(2)

    with col3:
        st.session_state.add_hashtags= st.checkbox("Add hashtags", st.session_state.add_hashtags)
    with col4:
        st.session_state.add_emojis= st.checkbox("Add emojis", st.session_state.add_emojis)
    
  
if st.button("Generate Captions"):
    if st.session_state.current_image_hash is not None:
      try:
            with st.spinner("Generating Captions..."):
                if st.session_state.current_image_description is None:
                     st.session_state.current_image_description = generate_image_description(image, encoder, decoder, device, vocab)
                image_description = st.session_state.current_image_description
                prompt = config.generate_prompt(st.session_state)
                response = model.generate_content(prompt)
                captions = json.loads(response.text)
            for i, caption in enumerate(captions, 1):
                st.markdown(config.CAPTION_HTML.format(i=i, caption=caption), unsafe_allow_html=True)
      except Exception as e:
            st.error(f"Error generating captions: {str(e)}")
    else:
        st.error("Please upload an image to generate captions.")
        
st.markdown(config.FOOTER_HTML, unsafe_allow_html=True) 


        
        


