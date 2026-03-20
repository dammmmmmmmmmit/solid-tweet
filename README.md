# 🔥 Disaster Tweet Classifier

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Live Web App:** [https://solid-tweet-hdsqgkpbprjm8jkvpttt86.streamlit.app/](https://solid-tweet-hdsqgkpbprjm8jkvpttt86.streamlit.app/)

A real-time web application built with Streamlit that classifies tweets as real disasters or non-disasters using an ensemble of three distinct Machine Learning and Deep Learning NLP models. 

The application utilizes a custom Gruvbox-inspired dark aesthetic and evaluates user-provided incoming text concurrently across all three models to achieve a robust "Majority Verdict."

---

## 🛠️ Technology Stack

### Frontend & Web Application
- **Streamlit:** Serves the interactive user interface, handles state management, and renders data visualization metrics without requiring a dedicated front-end JavaScript codebase.
- **Matplotlib:** Renders the customized evaluation metrics, model probabilities, and training loss curves styled to precisely match the app's Gruvbox aesthetic.
- **Hugging Face Hub (`huggingface_hub`):** Dynamically downloads gigabytes of model weights onto the Streamlit Cloud container at runtime to bypass GitHub's 100MB Large File Storage (LFS) limits.

### Machine Learning & Data Processing
- **PyTorch:** The primary deep learning framework powering the CNN-LSTM neural network architecture and executing inference.
- **Hugging Face Transformers:** Handles the implementation, tokenization, and inference for the fine-tuned BERT architecture.
- **Scikit-Learn:** Powers the TF-IDF vectorization pipelines and the baseline Random Forest ensemble model.
- **Numpy & Pandas:** Utilized for array manipulations, data handling, and preprocessing.

---

## 🧠 Model Architectures & Explanation

The core philosophy of this project is to benchmark and compare three totally different paradigms of NLP: Classical ML (TF-IDF + Trees), Deep Sequence Modeling (CNN-LSTM + Pre-trained Word Vectors), and Modern Transformers (BERT).

Below is the detailed breakdown of each model actively running in the application ensemble:

### 1. Random Forest (Baseline Machine Learning)
- **Vectorization:** TF-IDF (Term Frequency - Inverse Document Frequency). Captures unigrams and bigrams from the tweets while penalizing heavily frequent stop-words.
- **Architecture:** An ensemble of randomized decision trees. It processes the sparse matrix output by the TF-IDF vectorizer.
- **Strengths:** Incredibly fast CPU inference. Creates a highly interpretable and solid statistical baseline. 

### 2. CNN-LSTM (Deep Sequence Model)
- **Embeddings:** Initialized using pre-trained **100-dimensional GloVe embeddings** (`glove.twitter.27B.100d`), which map tokens to geometric contextual vectors learned from 27 Billion real Twitter posts.
- **CNN Layer (1D):** A 1D Convolutional Neural Network layer acts as a local feature extractor. It slides over the sequential word embeddings to identify essential n-gram patterns (like disaster keywords) regardless of where they appear in the tweet.
- **LSTM Layer:** The pooled output from the CNN is passed into a Long Short-Term Memory (LSTM) recurrent layer. This allows the model to understand the sequential context and temporal relationship between the extracted features before making a final dense prediction.
- **Strengths:** Excellent at combining local keyword detection (via CNN) with broader sentence structure and context understanding (via LSTM).

### 3. BERT (Transformer)
- **Base Model:** `bert-base-uncased` fine-tuned specifically on the disaster tweet dataset.
- **Architecture:** Deep bidirectional transformer encoding. Unlike the LSTM which reads text sequentially, BERT processes all tokens in the sentence simultaneously using Multi-Head Self-Attention.
- **Strengths:** State-of-the-Art contextual understanding. BERT can distinguish between a tweet saying *"The sky is on fire tonight"* (metaphorical/safe) versus *"Forest fire spreading rapidly"* (actual disaster) with significantly higher accuracy than the other two models. It acts as the ultimate decider in consensus disputes.

---

## 🚀 Application Features

1. **Live Prediction Mode:** Type or paste any tweet to see how all three models predict its likelihood of being a disaster in real time. It visually highlights disaster-related keywords (like *wildfire*, *earthquake*, *evacuate*) and resolves model disagreements through a 2/3 majority vote.
2. **Feed Triage Monitor:** Paste a large batch of raw unstructured tweets (one per line) to instantly triage them. The BERT model scans the entire bulk feed and flags high-confidence emergency alerts.
3. **Analytics Dashboard:** Visualizes evaluation metrics (Accuracy, F1 Score, Precision, Recall, ROC-AUC) and displays the historical training/loss curves for the deep learning models across the dataset.

---

## 💻 Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/dammmmmmmmmmit/solid-tweet.git
cd solid-tweet
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit Application
```bash
streamlit run app.py
```
> **Note on Model Weights:** Due to GitHub's file size limitations, the application is configured to automatically download the necessary heavy `.pt`, `.safetensors`, and `.pkl` model artifacts directly from Hugging Face Hub during its first launch. Allow 1-2 minutes for the initial download to complete.

---

## 📊 Dataset Reference
The models were trained on the classic [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) dataset from Kaggle, containing 7,613 hand-labeled tweets analyzing whether a post reports a real catastrophic event.
