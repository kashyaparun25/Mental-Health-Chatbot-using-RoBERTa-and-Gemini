import os
import torch
import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
import google.generativeai as genai
import torch.nn as nn
from transformers import AutoModel

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")  # Configure the Gemini API model

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GoEmotions Dataset
dataset = load_dataset("go_emotions", split="test")
emotion_labels = dataset.features["labels"].feature.names  # Define emotion labels here

# Define the custom model architecture
class BERTModel(nn.Module):
    def __init__(self, num_labels):
        super(BERTModel, self).__init__()
        self.transformer = AutoModel.from_pretrained("roberta-base")
        self.fc1 = nn.Linear(self.transformer.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return self.sigmoid(logits)

# Load tokenizer and model
tokenizer_path = "D:/RoBERTa_Goemotion"  # Update with your tokenizer directory
model_path = "D:/RoBERTa_Goemotion/model_weights.pth"  # Update with your model path

num_labels = 28  # Number of emotion classes
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = BERTModel(num_labels=num_labels)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define Ekman Mapping
ekman_mapping = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "gratitude", "love", "optimism", "relief", "pride"],
    "sadness": ["sadness", "disappointment", "grief"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"]
}

# Function to map GoEmotions to Ekman emotions
def map_to_ekman(goemotions_probs):
    ekman_probs = {emotion: 0 for emotion in ekman_mapping.keys()}
    for ekman_emotion, goemotions_list in ekman_mapping.items():
        ekman_probs[ekman_emotion] = sum(
            [goemotions_probs.get(label, 0) for label in goemotions_list]
        )
    return ekman_probs

# Function to classify emotion
def classify_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    probabilities = outputs.cpu().detach().numpy()[0]
    goemotions_probs = {label: prob for label, prob in zip(emotion_labels, probabilities)}
    top_3_emotions = sorted(goemotions_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    ekman_probs = map_to_ekman(goemotions_probs)
    return goemotions_probs, ekman_probs, top_3_emotions

# Function to generate responses using Gemini API with solutions
def get_gemini_response_with_solutions(prompt, emotions):
    emotions_text = ", ".join([f"{emotion} ({round(prob * 100, 2)}%)" for emotion, prob in emotions])
    try:
        # Modify the prompt to explicitly request solutions
        response = gemini_model.generate_content(
            f"Behave like a mental health expert. The user is feeling these emotions: {emotions_text}. Please analyze the situation and provide a response with possible solutions to help the user. And keep the solutions to the point. Aditionally you can ask the user to for more questions if required "
        )
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Error generating response."

# Chat history management
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Streamlit App
st.title("🧑‍⚕️ Mental Health Chatbot 🧠 ")

# Display chat history
st.subheader("Chat History")
for entry in st.session_state["chat_history"]:
    st.write(f"**User Prompt:** {entry['user_input']}")
    st.write(f"**Mental Health Bot Response:** {entry['response']}")
    st.write("**Top 3 Emotions Detected:**")
    for emotion, prob in entry["top_3_emotions"]:
        st.write(f"- {emotion}: {round(prob * 100, 2)}%")
    st.write("**GoEmotions Probabilities:**")
    st.bar_chart(entry["goemotions_probs_df"])
    st.write("**Ekman Emotion Probabilities:**")
    st.bar_chart(entry["ekman_probs_df"])

# Persistent prompt bar at the bottom
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
st.markdown("---")
with st.container():
    with st.form("user_input_form", clear_on_submit=True):
        user_input = st.text_area("Enter your message:", placeholder="Type your message here...", key="user_input")
        submit_button = st.form_submit_button("Submit")

if submit_button and user_input:
    # Classify emotion
    goemotions_probs, ekman_probs, top_3_emotions = classify_emotion(user_input)

    # Prepare dataframes for visualization
    goemotions_probs_df = pd.DataFrame.from_dict(
        goemotions_probs, orient="index", columns=["Probability"]
    ).sort_values(by="Probability", ascending=False)
    ekman_probs_df = pd.DataFrame.from_dict(
        ekman_probs, orient="index", columns=["Probability"]
    ).sort_values(by="Probability", ascending=False)

    # Generate response with solutions
    gemini_response = get_gemini_response_with_solutions(user_input, top_3_emotions)

    # Append to chat history
    st.session_state["chat_history"].append(
        {
            "user_input": user_input,
            "response": gemini_response,
            "top_3_emotions": top_3_emotions,
            "goemotions_probs_df": goemotions_probs_df,
            "ekman_probs_df": ekman_probs_df,
        }
    )

    # Display current results
    st.subheader("Top 3 Emotions Detected")
    for emotion, prob in top_3_emotions:
        st.write(f"- {emotion}: {round(prob * 100, 2)}%")

    st.subheader("GoEmotions Probabilities")
    st.bar_chart(goemotions_probs_df)

    st.subheader("Ekman Emotion Probabilities")
    st.bar_chart(ekman_probs_df)

    st.subheader("Response from Gemini API (with Solutions)")
    st.write(gemini_response)
