
import streamlit as st

PAGE_CONFIG = {
    "page_title": "InstaCaptionPro",
    "page_icon": "✨",
    "layout": "centered",
    "initial_sidebar_state": "collapsed",
    "menu_items": None
}

CUSTOM_CSS = """
    <style>  
     .stFileUploader > div > button {
        background-color: #FF7518;
        color: white;
    }
    .stFileUploader > div > button:hover {
        background-color: #FF5F1F;
    }
    
    /* Spinner */
    .stSpinner > div > div {
        border-top-color: #FF9900 !important;
    }
    .stButton button {
        background-color: #FF7518;
        color: white !important;
        font-size: 50px;
        border-radius: 8px;
        padding: 7px 15px;
        margin-top: 5px;
    }
    .stButton button:hover {
        background-color:  #FF5F1F	;
        color: white !important;
    }
   
    .stTextArea, .stSelectbox {
        width: 100%;
    }
    .columns {
        display: flex;
        justify-content: space-between;
    }
    .column {
        width: 20%;
    }
    .wide-column {
        width: 100%;
    }
    h1 {
         margin-top: 0px;
        margin-bottom: 3px;
    }
    p {
        margin-top: 5px;
        margin-bottom: 10px;
        
    }
     .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #FFF8F0;
        color: #6c757d;
        text-align: center;
        padding: 5px 0;
        font-size: 10px;
    }
    .footer a {
        color: #FF7518;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        color: #FF5F1F;
    }
    .footer img {
        height: 15px;
        width: 15px;
        margin-right: 5px;
        vertical-align: middle;
    }
.caption-container {
        background-color: #FFF8F0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        position: relative;
    }
    .caption-text {
        font-size: 18px;
        color: #333;
        margin-right: 30px;  /* Make space for the copy button */
    }

   
   
    </style>
"""



CAPTION_HTML = """
<div class='caption-container'>
    <p class='caption-text'><strong>Caption {i}:</strong> {caption}</p>
</div>
"""

def generate_prompt(session_state):
    prompt_template = st.secrets.PROMPT_TEMPLATE
    return prompt_template.format(
        no_of_captions=session_state.no_of_captions,
        image_description=session_state.current_image_description,
        user_context= session_state.description,
        tone=session_state.tone,
        include_hashtags="Yes" if session_state.add_hashtags else "No",
        include_emojis="Yes" if session_state.add_emojis else "No"
    )
    

FOOTER_HTML = """
<div class="footer">
    <p>
        InstaCaptionPro - Designed and developed by Anush Revankar  
        <a href="https://github.com/anushrevankar24" target="_blank"><img src="./app/static/github-logo.png" alt="GitHub" /> GitHub</a>
        <a href="https://www.linkedin.com/in/anush-revankar-0ab02225b" target="_blank"><img src="./app/static/linkedin-logo.png" alt="LinkedIn" /> LinkedIn</a>
        <a href="https://instagram.com/anushrevankar24?igshid=ZDdkNTZiNTM=" target="_blank"><img src="./app/static/instagram-logo.png" alt="Instagram" /> Instagram</a>
    </p>
</div>
"""



