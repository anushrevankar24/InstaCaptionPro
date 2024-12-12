
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import streamlit as st
import torch
from PIL import Image
import os
import google.generativeai as genai
import pickle
from vocabulary import Vocabulary

@st.cache_resource
def load_vocab():
    vocab_file = "models/vocab.pkl"
    with open(vocab_file, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_models(vocab_size):
    embed_size = 256
    hidden_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
    
    encoder_checkpoint = torch.load("models/encoder-9.pkl", map_location=device)
    decoder_checkpoint = torch.load("models/decoder-9.pkl", map_location=device)
    
    if isinstance(encoder_checkpoint, dict) and 'model_state_dict' in encoder_checkpoint:
        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    else:
        raise ValueError("Encoder checkpoint is not in the expected format")
    
    if isinstance(decoder_checkpoint, dict) and 'model_state_dict' in decoder_checkpoint:
        decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    else:
        raise ValueError("Decoder checkpoint is not in the expected format")
    
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
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):
      
        cap_embedding = self.embed(
            captions[:, :-1]
        ) 
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)
        lstm_out, self.hidden = self.lstm(
            embeddings
        ) 
        outputs = self.linear(lstm_out) 
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        res = []
        for i in range(max_len):
            lstm_out, states = self.lstm(
                inputs, states
            )  
            outputs = self.linear(lstm_out.squeeze(dim=1))  
            _, predicted_idx = outputs.max(dim=1)  
            res.append(predicted_idx.item())
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx)  
            inputs = inputs.unsqueeze(1)  
        return res