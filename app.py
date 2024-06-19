import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
from transformers import pipeline
import gdown
import os

# Function to download the model from Google Drive
def download_model_from_drive(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)

# Google Drive link
model_url = 'https://drive.google.com/uc?id=1SRehRKXhavHhfCQDB0M4S1rjcjW9H6iI'
model_path = 'transformer_model.keras'

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    download_model_from_drive(model_url, model_path)

# Load the saved model and vectorizers
model = keras.models.load_model(model_path, custom_objects={
    'PositionalEmbedding': PositionalEmbedding,
    'TransformerEncoder': TransformerEncoder,
    'TransformerDecoder': TransformerDecoder,
    'MultiHeadAttention': MultiHeadAttention
})

with open('source_vectorization.pkl', 'rb') as f:
    source_vectorization = pickle.load(f)

with open('target_vectorization.pkl', 'rb') as f:
    target_vectorization = pickle.load(f)

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

# Define translation function
def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break

    decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
    return decoded_sentence

# Streamlit app setup
st.title("English to German Translation and Sentiment Analysis")

st.write("Enter an English sentence to translate it into German and analyze the sentiment of the translated sentence.")

input_sentence = st.text_input("English Sentence")

if st.button("Translate and Analyze Sentiment"):
    if input_sentence:
        translated_sentence = decode_sequence(input_sentence)
        sentiment = sentiment_pipeline(translated_sentence)[0]['label']
        
        st.write("**English Sentence:**", input_sentence)
        st.write("**German Translation:**", translated_sentence)
        st.write("**Sentiment Analysis:**", sentiment)
    else:
        st.write("Please enter a sentence.")
