# 🔥 Disaster Tweet Classifier

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**Live Demo:** [https://solid-tweet-hdsqgkpbprjm8jkvpttt86.streamlit.app/](https://solid-tweet-hdsqgkpbprjm8jkvpttt86.streamlit.app/)

A real-time web application built with Streamlit that classifies tweets as real disasters or non-disasters using an ensemble of three Machine Learning and Deep Learning NLP models. 

## 🧠 Models Used
The application processes inputs through three different models simultaneously to provide a consensus prediction:
1. **Random Forest (Baseline):** Trained on TF-IDF features (unigrams and bigrams). Fast and lightweight.
2. **CNN-LSTM:** A hybrid deep neural network utilizing pre-trained 100-dimensional GloVe embeddings designed for Twitter data. 
3. **BERT (`bert-base-uncased`):** A fine-tuned transformer model that provides state-of-the-art contextual understanding of the tweets.

## 🚀 Features
- **Live Predict:** Type or paste any tweet to see how all three models predict its likelihood of being a disaster in real-time. Includes highlighting of disaster-related keywords.
- **Feed Monitor:** Paste a large batch of tweets (one per line) to instantly triage them using the BERT model. Flags high-confidence emergency alerts.
- **Dashboard:** Visualizes the evaluation metrics (Accuracy, F1, Precision, Recall, AUC) and training/loss curves for the models across the Kaggle NLP Disaster dataset.
- **Gruvbox UI:** Custom dark-mode aesthetic styling targeting the beloved Gruvbox color scheme.

---

## 🏗️ Architecture & Cloud Infrastructure
Due to the large size of the NLP model files (totaling over 500MB), the weights are **not** bundled directly with the application codebase. 

Instead, the Streamlit app uses `huggingface_hub` to **automatically pull the pre-trained `.pt`, `.safetensors`, and `.pkl` object files from Hugging Face** upon the first startup. 

- **GitHub Repository:** Contains the Application code (`app.py`), the UI logic, Data files (`train.csv`, `test.csv`), and the Notebooks.
- **Hugging Face Hub:** Hosts the ML model weights and the cached vocabulary. 

## 💻 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/dammmmmmmmmmit/solid-tweet.git
cd solid-tweet
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
*Note: The first time you run the app, it will take a minute or two to automatically download the model artifacts (~500MB) from Hugging Face into your local directory.*

## 📊 Dataset
The models were trained on the classic [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) dataset from Kaggle, containing 7,613 hand-labeled tweets.
