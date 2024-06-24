import streamlit as st
import numpy as np
import random
import re
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from streamlit_chat import message

# Pastikan stopwords NLTK telah diunduh
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load necessary resources
model = load_model('best_lstm_model1.h5')

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(json.dumps(data))  # Convert dict to JSON string

# Load label encoder
with open('label_encoder.json', 'r') as f:
    label_encoder_data = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_data['classes'])

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Set the max_sequence_length as used during training
max_sequence_length = 8  # Update this value to match the value used during training

# Define text preprocessing functions
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_text(text):
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

def preprocess_input(text):
    text = clean_text(text)
    tokens = process_text(text)
    sequence = tokenizer.texts_to_sequences([' '.join(tokens)])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    return padded_sequence

def get_response(text):
    padded_sequence = preprocess_input(text)
    prediction = model.predict(padded_sequence)
    predicted_tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "Maaf, saya tidak mengerti pertanyaan Anda."

# Streamlit UI
st.set_page_config(page_title="Boti", page_icon="🤖")
st.markdown(
    """
    <style>
    .main-container {
        background-color: green;
        height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 0;
        margin: 0;
    }
    .chat-container {
        background-color: white;
        width: 60%;
        max-width: 800px;
        height: 80%;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("BoTi🤖")

if 'history' not in st.session_state:
    st.session_state['history'] = []

def update_chat(user_input):
    response = get_response(user_input)
    st.session_state.history.append({"message": user_input, "is_user": True})
    st.session_state.history.append({"message": response, "is_user": False})

# Display chat history
for i, chat in enumerate(st.session_state['history']):
    if chat['is_user']:
        message(chat['message'], is_user=True, key=f"user_{i}", avatar_style="big-smile")
    else:
        message(chat['message'], is_user=False, key=f"bot_{i}", avatar_style="bottts")

# Input form at the bottom with a Send button
with st.form(key='my_form', clear_on_submit=True):
    col1, col2 = st.columns([7, 1])
    with col1:
        user_input = st.text_input("Masukkan pertanyaan...", key="user_input")
    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)  # Adjust the margin as needed
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        update_chat(user_input)
        st.experimental_rerun()

# Trigger the first response
if 'initial_response' not in st.session_state:
    st.session_state['initial_response'] = True
    st.experimental_rerun()
