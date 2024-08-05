import streamlit as st
from PIL import Image
from models import load_vocab_and_models, generate_image_description
import config
import json
from vocabulary import Vocabulary
import hashlib

st.set_page_config(**config.PAGE_CONFIG)

with st.spinner("Loading models and assets..."):
    try:
        vocab, encoder, decoder, device, model = load_vocab_and_models()
    except Exception as e:
        st.error(f"Error loading models and assets: {str(e)}")
        st.stop()

st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #FF7518;'>AI Instagram Caption Generator</h1>", unsafe_allow_html=True)

if 'description' not in st.session_state:
    st.session_state.description = ''
if 'tone' not in st.session_state:
    st.session_state.tone = "😎 Casual"
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


def image_to_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

with st.container():
    st.markdown("<div class='column'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Caption this image...", type=['jpg', 'png', 'gif'])
    if uploaded_file is not None:
        try:    
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image,width=300,use_column_width=False);
            image_hash = image_to_hash(image)
            if st.session_state.current_image_hash != image_hash:
                st.session_state.current_image_hash = image_hash
                st.session_state.current_image_description = None
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    st.session_state.description = st.text_area("Description (optional)", st.session_state.description)
    col1, col2 = st.columns([2,1])
    with col1:
        tone_options = [
            "😊 Friendly", "💼 Formal", "🥳 Excited", "😎 Casual", "👔 Professional", 
            "💪 Confident", "🤗 Empathetic", "💎 Luxury", "✨ Inspiring", "😏 Sarcastic",
            "😂 Humorous", "🎉 Enthusiastic", "🥂 Sophisticated",  "😎 Cool","✈️ Wanderlust",
            "🥰 Romantic", "🚀 Motivational",  "🧠 Thoughtful"]
        st.session_state.tone = st.selectbox("Tone", tone_options, 
                                            index=tone_options.index(st.session_state.tone))
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
                st.write(image_description)
            for i, caption in enumerate(captions, 1):
                st.markdown(config.CAPTION_HTML.format(i=i, caption=caption), unsafe_allow_html=True)
      except Exception as e:
            st.error(f"Error generating captions: {str(e)}")
    else:
        st.error("Please upload an image to generate captions.")
        
st.markdown(config.FOOTER_HTML, unsafe_allow_html=True) 


        
        


