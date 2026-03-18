import re
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
MAX_LEN   = 30
EMBED_DIM = 100
DEVICE    = torch.device("cpu")

app = FastAPI(
    title="Disaster Tweet Classifier API",
    description="Classifies a tweet as disaster or not using RF, CNN-LSTM, and BERT.",
    version="1.0.0"
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text: str, vocab: dict, max_len: int) -> list:
    tokens = text.split()[:max_len]
    ids    = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens]
    ids   += [0] * (max_len - len(ids))
    return ids

# ── CNN-LSTM model class ──────────────────────────────────────────────────────
class CNNLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters,
                 kernel_size, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.conv          = nn.Conv1d(embed_dim, num_filters, kernel_size,
                                       padding=kernel_size // 2)
        self.relu          = nn.ReLU()
        self.maxpool       = nn.MaxPool1d(kernel_size=2)
        self.lstm          = nn.LSTM(num_filters, hidden_dim,
                                     batch_first=True, dropout=dropout)
        self.lstm_dropout  = nn.Dropout(dropout)
        self.fc            = nn.Linear(hidden_dim, 1)
        self.sigmoid       = nn.Sigmoid()

    def forward(self, x):
        emb      = self.embed_dropout(self.embedding(x))
        emb      = emb.permute(0, 2, 1)
        conv_out = self.relu(self.conv(emb))
        pooled   = self.maxpool(conv_out).permute(0, 2, 1)
        _, (hidden, _) = self.lstm(pooled)
        out      = self.lstm_dropout(hidden[-1])
        return self.sigmoid(self.fc(out)).squeeze(1)

# ── Load models on startup ────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global rf_model, tfidf, vocab, cnn_model, bert_tokenizer, bert_model

    with open("rf_model.pkl",  "rb") as f: rf_model = pickle.load(f)
    with open("tfidf.pkl",     "rb") as f: tfidf    = pickle.load(f)
    with open("vocab.pkl",     "rb") as f: vocab    = pickle.load(f)

    cnn_model = CNNLSTMClassifier(
        vocab_size=len(vocab), embed_dim=EMBED_DIM,
        num_filters=128, kernel_size=3, hidden_dim=64, dropout=0.3
    ).to(DEVICE)
    cnn_model.load_state_dict(torch.load("cnn_lstm.pt", map_location=DEVICE))
    cnn_model.eval()

    bert_tokenizer = BertTokenizer.from_pretrained("bert_finetuned")
    bert_model     = BertForSequenceClassification.from_pretrained(
        "bert_finetuned"
    ).to(DEVICE)
    bert_model.eval()

    print("All models loaded.")

# ── Prediction helpers ────────────────────────────────────────────────────────
def run_rf(text: str):
    cleaned = clean_tweet(text)
    vec     = tfidf.transform([cleaned])
    pred    = int(rf_model.predict(vec)[0])
    prob    = float(rf_model.predict_proba(vec)[0][1])
    return pred, prob

def run_cnn(text: str):
    cleaned = clean_tweet(text)
    seq     = text_to_sequence(cleaned, vocab, MAX_LEN)
    tensor  = torch.tensor([seq], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        prob = cnn_model(tensor).item()
    return int(prob >= 0.5), round(prob, 4)

def run_bert(text: str):
    cleaned = clean_tweet(text)
    enc     = bert_tokenizer(
        cleaned, return_tensors="pt",
        max_length=64, truncation=True, padding="max_length"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = bert_model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred  = int(torch.argmax(probs).item())
    return pred, round(probs[1].item(), 4)

# ── Request / Response schemas ────────────────────────────────────────────────
class TweetRequest(BaseModel):
    tweet: str

    class Config:
        json_schema_extra = {
            "example": {"tweet": "Wildfire spreading near highway 40, residents evacuating"}
        }

class ModelResult(BaseModel):
    prediction:  str
    label:       int
    confidence:  float

class ClassifyResponse(BaseModel):
    tweet:        str
    cleaned:      str
    rf:           ModelResult
    cnn_lstm:     ModelResult
    bert:         ModelResult
    majority:     str
    agreement:    str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":    "Disaster Tweet Classifier API",
        "version": "1.0.0",
        "routes":  ["/classify", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyResponse)
def classify(body: TweetRequest):
    tweet = body.tweet.strip()
    if not tweet:
        raise HTTPException(status_code=422, detail="Tweet cannot be empty.")

    cleaned = clean_tweet(tweet)

    p_rf,   prob_rf   = run_rf(tweet)
    p_cnn,  prob_cnn  = run_cnn(tweet)
    p_bert, prob_bert = run_bert(tweet)

    votes     = [p_rf, p_cnn, p_bert]
    n_dis     = sum(votes)
    majority  = "disaster" if n_dis >= 2 else "not_disaster"
    agreement = f"{n_dis}/3 models" if majority == "disaster" else f"{3 - n_dis}/3 models"

    return ClassifyResponse(
        tweet    = tweet,
        cleaned  = cleaned,
        rf       = ModelResult(
            prediction = "disaster" if p_rf   == 1 else "not_disaster",
            label      = p_rf,
            confidence = prob_rf
        ),
        cnn_lstm = ModelResult(
            prediction = "disaster" if p_cnn  == 1 else "not_disaster",
            label      = p_cnn,
            confidence = prob_cnn
        ),
        bert     = ModelResult(
            prediction = "disaster" if p_bert == 1 else "not_disaster",
            label      = p_bert,
            confidence = prob_bert
        ),
        majority  = majority,
        agreement = agreement,
    )
