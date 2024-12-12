# InstaCaptionPro

## AI Instagram Caption Generator

InstaCaption Pro is an AI-powered tool designed to help users generate Instagram captions effortlessly. I created this project because I often struggled to come up with suitable captions for my posts. Searching online for ideas and figuring out which caption would best fit my images was time-consuming. To simplify this process, I decided to build InstaCaption Pro.

The main goal of this project is to create engaging and relevant captions for Instagram posts. Users can upload an image, add an optional description, choose the tone of the caption, and select the number of captions they want. They also have the option to include hashtags and emojis.

The application is built using Streamlit for an easy and interactive user experience. For the deep learning models, I utilized PyTorch and integrated Google Generative AI for generating captions. At the core of the app is an Encoder-Decoder architecture. The encoder uses a pre-trained ResNet50 model to extract features from the uploaded image, while an LSTM-based decoder generates a description based on these features. This description, along with user-selected options like tone and the number of captions, is sent to the Google Generative AI model, which creates a set of captions that are displayed to the user.

To ensure a smooth and efficient user experience, the application employs several optimization techniques. It uses Streamlit's caching mechanism to prevent the model from loading with every request, significantly improving response times. Additionally, session state management retains user inputs and generated descriptions. For instance, if a user changes the number of captions or the tone without altering the image, the app does not regenerate the image description, saving both time and computational resources. Overall, InstaCaption Pro aims to make the process of creating engaging Instagram captions quick and effortless.

## Demo of the Project
[Video Link](https://drive.google.com/file/d/1dGzSpKMzISMa1IQ7jMHh8oVxdDXxBQ18/view?usp=drive_link) 

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





