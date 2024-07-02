
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import streamlit as st
import torch
from PIL import Image
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pickle
from vocabulary import Vocabulary
load_dotenv()

@st.cache_resource
def load_vocab():
    vocab_file = "models/vocab.pkl"
    with open(vocab_file, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_models(vocab_size):
    embed_size = 256
    hidden_size = 512
    device = torch.device("cpu")
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
    
    encoder.load_state_dict(torch.load("models/encoder-3.pkl", map_location=device))
    decoder.load_state_dict(torch.load("models/decoder-3.pkl", map_location=device))
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, device

@st.cache_resource
def load_gemini_model():
    genai.configure(api_key=st.secrets.API_KEY)
    return genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

def load_vocab_and_models():
    vocab = load_vocab()
    encoder, decoder, device = load_models(len(vocab))
    gemini_model = load_gemini_model()
    return vocab, encoder, decoder, device, gemini_model

@st.cache_data
def generate_image_description(image, _encoder, _decoder, _device, _vocab):
    image = transform_image(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        features = _encoder(image).unsqueeze(1)
        output = _decoder.sample(features)
    return clean_sentence(output, _vocab.idx2word)

transform_image = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])

def clean_sentence(output, idx2word):
    sentence = ""
    for i in output:
        word = idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        if i == 18:
            sentence = sentence + word
        else:
            sentence = sentence + " " + word
    return sentence
  

# ----------- Encoder ------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # disable learning for parameters
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


# --------- Decoder ----------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args:
            embed_size: final embedding size of the CNN encoder
            hidden_size: hidden size of the LSTM
            vocab_size: size of the vocabulary
            num_layers: number of layers of the LSTM
        """
        super(DecoderRNN, self).__init__()

        # Assigning hidden dimension
        self.hidden_dim = hidden_size
        # Map each word index to a dense word embedding tensor of embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Creating LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # Initializing linear to apply at last of RNN layer for further prediction
        self.linear = nn.Linear(hidden_size, vocab_size)
        # Initializing values for hidden and cell state
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):
      
        cap_embedding = self.embed(
            captions[:, :-1]
        )  # (bs, cap_length) -> (bs, cap_length-1, embed_size)

        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)

        #  getting output i.e. score and hidden layer.
        # first value: all the hidden states throughout the sequence. second value: the most recent hidden state
        lstm_out, self.hidden = self.lstm(
            embeddings
        )  # (bs, cap_length, hidden_size), (1, bs, hidden_size)
        outputs = self.linear(lstm_out)  # (bs, cap_length, vocab_size)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
       
        res = []

        # Now we feed the LSTM output and hidden states back into itself to get the caption
        for i in range(max_len):
            lstm_out, states = self.lstm(
                inputs, states
            )  # lstm_out: (1, 1, hidden_size)
            outputs = self.linear(lstm_out.squeeze(dim=1))  # outputs: (1, vocab_size)
            _, predicted_idx = outputs.max(dim=1)  # predicted: (1, 1)
            res.append(predicted_idx.item())
            # if the predicted idx is the stop index, the loop stops
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx)  # inputs: (1, embed_size)
            # prepare input for next iteration
            inputs = inputs.unsqueeze(1)  # inputs: (1, 1, embed_size)

        return res