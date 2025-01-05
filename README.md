### Mental Health Chatbot using RoBERTa and Gemini

#### Repository Structure
```
Mental_Health_Chatbot/
├── model_weights.pth       # Pre-trained model weights
├── special_tokens_map.json # Tokenizer special tokens mapping
├── tokenizer/              # Tokenizer configurations and vocab
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── vocab.json
├── app.py                  # Main application script (Streamlit)
├── README.md               # Project documentation
├── resources/              # Supporting documentation and images
│   ├── DL_Presentation.pdf
│   ├── BERT+LSTM.pdf
│   ├── RoBERTa.pdf
│   ├── chatbot_workflow.png # Workflow diagram
```

---

### Project Overview
This repository contains the implementation of a mental health chatbot designed to classify user emotions and provide solutions for emotional well-being. The chatbot leverages state-of-the-art NLP models, GoEmotions dataset, and Gemini AI for response generation.

#### Key Features:
- **Emotion Detection**: Identifies 28 emotions using fine-tuned RoBERTa and Ekman mappings.
- **Solution-Oriented Responses**: Generates actionable mental health responses via Gemini API.
- **Interactive UI**: A user-friendly chatbot interface built using Streamlit.
- **Visualization**: Bar charts showcasing emotion probabilities.

---

### Model Description

#### RoBERTa
- Pretrained model used for emotion classification.
- Integrated with a cosine learning rate scheduler for training stability.
- Binary cross-entropy loss for multi-label classification.

#### BERT + LSTM
- Combines BERT embeddings with LSTM for sequential modeling.
- Used as a baseline for model comparison.

---

### Dataset
- **GoEmotions**: A dataset by Google with 58k labeled Reddit comments.
- **Labels**: 28 fine-grained emotion categories including joy, anger, sadness, and fear.

#### Preprocessing
- Tokenization via RoBERTa tokenizer.
- Multi-hot encoding for multi-label classification.

---

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9
- HuggingFace Transformers
- Streamlit
- Google Generative AI SDK

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

### Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Mental_Health_Chatbot.git
```

2. Navigate to the project directory and start the Streamlit app:
```bash
cd Mental_Health_Chatbot
streamlit run app.py
```

3. Enter a message in the chatbot UI to receive emotion detection results and mental health responses.

---

### Workflow Diagram
![Chatbot Workflow](resources/chatbot_workflow.png)

---

### References
1. GoEmotions Dataset by Google
2. HuggingFace Transformers
3. RoBERTa and BERT Research Papers
