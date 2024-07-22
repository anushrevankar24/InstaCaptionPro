# InstaCaptionPro

## AI Instagram Caption Generator

This project is an AI-powered Instagram caption generator built using Streamlit. Users can upload an image, optionally provide a description, select a tone, and generate multiple captions with options to include hashtags and emojis. The caption generation leverages a pre-trained model integrated with Google Gemini API.

## Features

- **Image Upload**: Upload images in JPG, PNG, or GIF format.
- **Description**: Optionally provide a description for the image.
- **Tone Selection**: Choose from a variety of tones including Friendly, Formal, Excited, Casual, Professional, Confident, Empathetic, Luxury, Inspiring, Sarcastic, Humorous, Enthusiastic, Sophisticated, Cool, Wanderlust, Romantic, Motivational, Thoughtful.
- **Number of Captions**: Select the number of captions to generate (1, 3, or 5).
- **Hashtags and Emojis**: Option to include hashtags and emojis in the generated captions.
- **Caption Generation**: Generate captivating Instagram captions in seconds using a pre-trained model and Google Gemini API.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-instagram-caption-generator.git
   cd ai-instagram-caption-generator
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
4. Obtain a Google Gemini API key:
   - Visit the Google Gemini API page.
   - Follow the instructions to create a project and enable the API.
   - Generate an API key.
5. Set up your API key and prompt in Streamlit Secrets:
 - Create a file named secrets.toml in the .streamlit folder in the root directory of the project.
 - Add your API key and prompt to the secrets.toml file:
   ```bash
    API_KEY = "your_api_key_here"
    PROMPT_TEMPLATE = "your_prompt_here"
 - Note: Currently, a secret prompt is used that cannot be made public. You will need to obtain your own prompt for the application.
 3. Run the application:
    ```bash
    streamlit run app.py

## Contributing
- Fork the repository.
- Create a new branch: git checkout -b my-feature-branch.
- Make your changes and commit them: git commit -m 'Add some feature'.
- Push to the branch: git push origin my-feature-branch.
- Submit a pull request.
## License
- This project is licensed under the MIT License.





