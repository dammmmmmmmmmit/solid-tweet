import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import roc_curve
from transformers import BertTokenizer, BertForSequenceClassification

GRV = {
    "bg":      "#282828",
    "bg1":     "#3c3836",
    "bg2":     "#504945",
    "bg3":     "#665c54",
    "fg":      "#ebdbb2",
    "fg2":     "#d5c4a1",
    "yellow":  "#fabd2f",
    "orange":  "#fe8019",
    "red":     "#fb4934",
    "green":   "#b8bb26",
    "aqua":    "#8ec07c",
    "blue":    "#83a598",
    "purple":  "#d3869b",
    "gray":    "#928374",
}

st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] {{
      background-color: {GRV['bg']} !important;
      color: {GRV['fg']} !important;
      font-family: 'IBM Plex Sans', sans-serif;
  }}
  section[data-testid="stSidebar"] {{
      background-color: {GRV['bg1']} !important;
      border-right: 1px solid {GRV['bg2']};
  }}
  section[data-testid="stSidebar"] * {{ color: {GRV['fg']} !important; }}
  .stButton > button {{
      background-color: {GRV['bg2']} !important;
      color: {GRV['yellow']} !important;
      border: 1px solid {GRV['yellow']} !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-weight: 600;
      letter-spacing: 0.05em;
      border-radius: 2px !important;
      transition: all 0.15s ease;
  }}
  .stButton > button:hover {{
      background-color: {GRV['yellow']} !important;
      color: {GRV['bg']} !important;
  }}
  .stTextArea textarea, .stTextInput input {{
      background-color: {GRV['bg1']} !important;
      color: {GRV['fg']} !important;
      border: 1px solid {GRV['bg3']} !important;
      border-radius: 2px !important;
      font-family: 'JetBrains Mono', monospace !important;
  }}
  .stTextArea textarea:focus, .stTextInput input:focus {{
      border-color: {GRV['yellow']} !important;
      box-shadow: 0 0 0 1px {GRV['yellow']}44 !important;
  }}
  .stSelectbox > div > div {{
      background-color: {GRV['bg1']} !important;
      color: {GRV['fg']} !important;
      border: 1px solid {GRV['bg3']} !important;
      border-radius: 2px !important;
  }}
  [data-testid="metric-container"] {{
      background-color: {GRV['bg1']} !important;
      border: 1px solid {GRV['bg2']} !important;
      border-radius: 2px !important;
      padding: 12px 16px !important;
  }}
  [data-testid="metric-container"] label {{
      color: {GRV['gray']} !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 0.7rem !important;
      text-transform: uppercase;
      letter-spacing: 0.1em;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
      color: {GRV['yellow']} !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 1.6rem !important;
      font-weight: 700;
  }}
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {GRV['bg1']} !important;
      gap: 0px;
      border-bottom: 1px solid {GRV['bg2']};
  }}
  .stTabs [data-baseweb="tab"] {{
      background-color: transparent !important;
      color: {GRV['gray']} !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 0.75rem !important;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border-radius: 0 !important;
      padding: 10px 20px !important;
  }}
  .stTabs [aria-selected="true"] {{
      color: {GRV['yellow']} !important;
      border-bottom: 2px solid {GRV['yellow']} !important;
      background-color: {GRV['bg']} !important;
  }}
  .stDataFrame {{ border: 1px solid {GRV['bg2']} !important; }}
  hr {{ border-color: {GRV['bg2']} !important; }}
  h1, h2, h3 {{
      font-family: 'JetBrains Mono', monospace !important;
      color: {GRV['yellow']} !important;
      font-weight: 700;
      letter-spacing: -0.02em;
  }}
  h4, h5, h6 {{
      font-family: 'IBM Plex Sans', sans-serif !important;
      color: {GRV['fg2']} !important;
  }}
  .stRadio label {{ color: {GRV['fg']} !important; }}
  .stRadio [data-baseweb="radio"] div {{ border-color: {GRV['gray']} !important; }}
  .stSpinner > div {{ border-top-color: {GRV['yellow']} !important; }}
  .grv-card {{
      background: {GRV['bg1']};
      border: 1px solid {GRV['bg2']};
      border-radius: 2px;
      padding: 20px 24px;
      margin-bottom: 12px;
  }}
  .grv-label {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: {GRV['gray']};
      margin-bottom: 4px;
  }}
  .grv-mono {{ font-family: 'JetBrains Mono', monospace; color: {GRV['aqua']}; }}
  .prob-bar-wrap {{
      background: {GRV['bg2']};
      border-radius: 1px;
      height: 6px;
      width: 100%;
      margin-top: 6px;
  }}
  .prob-bar-fill-disaster {{ background: {GRV['red']}; height: 6px; border-radius: 1px; }}
  .prob-bar-fill-safe {{ background: {GRV['green']}; height: 6px; border-radius: 1px; }}
</style>
""", unsafe_allow_html=True)


def gruvbox_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(GRV["bg1"])
    ax.set_facecolor(GRV["bg"])
    ax.tick_params(colors=GRV["fg2"], labelsize=8)
    ax.xaxis.label.set_color(GRV["fg2"])
    ax.yaxis.label.set_color(GRV["fg2"])
    ax.title.set_color(GRV["yellow"])
    for spine in ax.spines.values():
        spine.set_edgecolor(GRV["bg3"])
    ax.grid(color=GRV["bg2"], linestyle="--", linewidth=0.6, alpha=0.8)
    return fig, ax


MAX_LEN   = 30
EMBED_DIM = 100

# ── Top disaster keywords (from TF-IDF feature importance proxy) ──────────────
DISASTER_KEYWORDS = {
    "fire","wildfire","flood","earthquake","tsunami","tornado","hurricane",
    "cyclone","storm","disaster","emergency","evacuate","evacuation","rescue",
    "collapse","explosion","crash","accident","killed","dead","deaths","injured",
    "trapped","damage","destruction","destroyed","burning","flames","wreckage",
    "victims","casualty","casualties","shelter","warning","alert","outbreak",
    "blaze","sinkhole","landslide","avalanche","volcano","eruption","debris",
    "fatalities","missing","survivors","rubble","devastation","hazard","threat",
}

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text, vocab, max_len):
    tokens = text.split()[:max_len]
    ids    = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens]
    ids   += [0] * (max_len - len(ids))
    return ids

def highlight_keywords(text):
    """Wrap disaster keywords in highlighted spans."""
    words  = text.split()
    result = []
    for w in words:
        clean = re.sub(r'[^a-z]', '', w.lower())
        if clean in DISASTER_KEYWORDS:
            result.append(
                f"<span style='background:{GRV['red']}33;color:{GRV['red']};"
                f"border-radius:2px;padding:1px 3px;font-weight:700'>{w}</span>"
            )
        else:
            result.append(f"<span style='color:{GRV['fg2']}'>{w}</span>")
    return " ".join(result)

def conf_distribution_chart(probs, preds):
    """Small histogram of confidence scores split by class."""
    fig, ax = gruvbox_fig(6, 2.8)
    dis_probs  = [p for p, pred in zip(probs, preds) if pred == 1]
    safe_probs = [p for p, pred in zip(probs, preds) if pred == 0]
    bins = np.linspace(0, 1, 11)
    if dis_probs:
        ax.hist(dis_probs,  bins=bins, color=GRV["red"],   alpha=0.75,
                label=f"Disaster ({len(dis_probs)})", edgecolor=GRV["bg"])
    if safe_probs:
        ax.hist(safe_probs, bins=bins, color=GRV["green"], alpha=0.75,
                label=f"Safe ({len(safe_probs)})",    edgecolor=GRV["bg"])
    ax.set_xlabel("Confidence score", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Confidence distribution", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, facecolor=GRV["bg1"], edgecolor=GRV["bg3"],
              labelcolor=GRV["fg"])
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


class CNNLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_size, hidden_dim, dropout=0.3):
        super(CNNLSTMClassifier, self).__init__()
        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.conv          = nn.Conv1d(embed_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.relu          = nn.ReLU()
        self.maxpool       = nn.MaxPool1d(kernel_size=2)
        self.lstm          = nn.LSTM(num_filters, hidden_dim, batch_first=True, dropout=dropout)
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


@st.cache_resource
def load_all_models():
    device = torch.device("cpu")
    with open("rf_model.pkl",  "rb") as f: rf    = pickle.load(f)
    with open("tfidf.pkl",     "rb") as f: tfidf = pickle.load(f)
    with open("vocab.pkl",     "rb") as f: vocab = pickle.load(f)
    cnn = CNNLSTMClassifier(
        vocab_size=len(vocab), embed_dim=EMBED_DIM,
        num_filters=128, kernel_size=3, hidden_dim=64, dropout=0.3
    ).to(device)
    cnn.load_state_dict(torch.load("cnn_lstm.pt", map_location=device))
    cnn.eval()
    bert_tok   = BertTokenizer.from_pretrained("bert_finetuned")
    bert_model = BertForSequenceClassification.from_pretrained("bert_finetuned").to(device)
    bert_model.eval()
    with open("dashboard_data.json") as f:
        dash = json.load(f)
    return rf, tfidf, vocab, cnn, bert_tok, bert_model, dash, device


def predict_rf(text, rf, tfidf):
    cleaned = clean_tweet(text)
    vec     = tfidf.transform([cleaned])
    pred    = rf.predict(vec)[0]
    prob    = rf.predict_proba(vec)[0][1]
    return int(pred), float(prob)

def predict_cnn(text, cnn, vocab, device):
    cleaned = clean_tweet(text)
    seq     = text_to_sequence(cleaned, vocab, MAX_LEN)
    tensor  = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = cnn(tensor).item()
    return int(prob >= 0.5), prob

def predict_bert(text, bert_tok, bert_model, device):
    cleaned = clean_tweet(text)
    enc     = bert_tok(cleaned, return_tensors="pt",
                       max_length=64, truncation=True, padding="max_length")
    enc     = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = bert_model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred  = torch.argmax(probs).item()
    return pred, probs[1].item()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='margin-bottom:24px'>
      <div style='font-family:JetBrains Mono,monospace;font-size:1.1rem;
                  color:{GRV["yellow"]};font-weight:700;letter-spacing:-0.01em'>
        🔥 disaster.clf
      </div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                  color:{GRV["gray"]};letter-spacing:0.1em;text-transform:uppercase;
                  margin-top:4px'>
        tweet classification · v3
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "navigate",
        ["⚡  Live Predict", "📡  Feed Monitor", "📊  Dashboard", "🧠  Architecture"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                color:{GRV["gray"]};line-height:1.8'>
      <div style='color:{GRV["fg2"]};margin-bottom:6px;text-transform:uppercase;
                  letter-spacing:0.1em'>models</div>
      <div>RF · TF-IDF + bigrams</div>
      <div>CNN-LSTM · GloVe-100</div>
      <div>BERT · bert-base-uncased</div>
      <br>
      <div style='color:{GRV["fg2"]};margin-bottom:6px;text-transform:uppercase;
                  letter-spacing:0.1em'>dataset</div>
      <div>Kaggle NLP Disaster</div>
      <div>7,613 train tweets</div>
    </div>
    """, unsafe_allow_html=True)


with st.spinner("Loading models..."):
    try:
        rf, tfidf, vocab, cnn, bert_tok, bert_model, dash, device = load_all_models()
        models_loaded = True
    except Exception as e:
        models_loaded = False
        load_error    = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "Live Predict" in page:
    import requests as req
    import json as _json

    st.markdown("## ⚡ Live Prediction")

    # ── Mode toggle ────────────────────────────────────────────────────────────
    col_title, col_toggle = st.columns([3, 1])
    with col_title:
        st.markdown(f"<div style='color:{GRV['gray']};font-size:0.85rem;margin-bottom:8px'>"
                    "Type any tweet. All three models run simultaneously.</div>",
                    unsafe_allow_html=True)
    with col_toggle:
        api_mode = st.toggle("Use API", value=False)
        if api_mode:
            api_url = st.text_input("API URL", value="http://localhost:8000",
                                    label_visibility="collapsed")
        else:
            api_url = None

    if api_mode:
        st.markdown(f"""
        <div style='background:{GRV["bg1"]};border:1px solid {GRV["aqua"]};
                    border-radius:2px;padding:8px 14px;margin-bottom:16px;
                    font-family:JetBrains Mono,monospace;font-size:0.7rem;
                    color:{GRV["aqua"]}'>
          API mode — predictions routed via
          <span style='color:{GRV["yellow"]}'>{api_url}/classify</span>
        </div>
        """, unsafe_allow_html=True)

    if not models_loaded and not api_mode:
        st.error(f"Could not load models: {load_error}")
        st.stop()

    tweet_input = st.text_area(
        "tweet_input", height=100,
        placeholder="e.g. Wildfire spreading near residential area...",
        label_visibility="collapsed"
    )

    run = st.button("▶  RUN ALL MODELS", use_container_width=False)

    def render_results(p_rf, p_cnn, p_bert):
        """Shared rendering for both direct and API mode."""
        st.markdown("---")

        highlighted = highlight_keywords(tweet_input)
        st.markdown(f"""
        <div class='grv-label' style='margin-bottom:4px'>Keyword analysis</div>
        <div style='font-family:IBM Plex Sans,sans-serif;font-size:0.9rem;
                    background:{GRV["bg1"]};padding:12px 16px;
                    border-radius:2px;border:1px solid {GRV["bg2"]};
                    margin-bottom:20px;line-height:1.8'>{highlighted}</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                    color:{GRV["gray"]};margin-bottom:20px'>
          Highlighted words = known disaster signal words
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for col, name, (p, pb), accent in zip(
            [c1, c2, c3],
            ["Random Forest", "CNN-LSTM", "BERT"],
            [p_rf, p_cnn, p_bert],
            [GRV["green"], GRV["blue"], GRV["purple"]],
        ):
            result_color = GRV["red"] if p == 1 else GRV["green"]
            bar_class    = "disaster" if p == 1 else "safe"
            label        = "DISASTER" if p == 1 else "NOT DISASTER"
            col.markdown(f"""
            <div style='background:{GRV["bg1"]};border:1px solid {accent};
                        border-radius:2px;padding:20px 22px;height:100%'>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                          text-transform:uppercase;letter-spacing:0.12em;
                          color:{accent};margin-bottom:10px'>{name}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:1.4rem;
                          font-weight:700;color:{result_color};margin-bottom:8px'>
                {label}
              </div>
              <div style='font-family:JetBrains Mono,monospace;font-size:1.8rem;
                          font-weight:700;color:{result_color}'>{pb*100:.1f}%</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                          color:{GRV["gray"]};margin-bottom:8px'>confidence</div>
              <div class='prob-bar-wrap'>
                <div class='prob-bar-fill-{bar_class}' style='width:{pb*100:.1f}%'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        votes      = [p_rf[0], p_cnn[0], p_bert[0]]
        n_disaster = sum(votes)
        verdict    = "DISASTER" if n_disaster >= 2 else "NOT DISASTER"
        v_color    = GRV["red"] if n_disaster >= 2 else GRV["green"]
        disagree   = len(set(votes)) > 1 and n_disaster in [1, 2]

        st.markdown(f"""
        <div style='margin-top:20px;background:{GRV["bg1"]};
                    border-left:3px solid {v_color};padding:14px 20px;
                    border-radius:2px;margin-bottom:8px'>
          <span style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                       text-transform:uppercase;letter-spacing:0.12em;color:{GRV["gray"]}'>
            Majority verdict ({n_disaster}/3 models agree)
          </span><br>
          <span style='font-family:JetBrains Mono,monospace;font-size:1.2rem;
                       font-weight:700;color:{v_color}'>{verdict}</span>
        </div>
        """, unsafe_allow_html=True)

        if disagree:
            st.markdown(f"""
            <div style='background:{GRV["yellow"]}18;border:1px solid {GRV["yellow"]};
                        border-radius:2px;padding:10px 16px;margin-bottom:16px'>
              <span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                           color:{GRV["yellow"]};font-weight:700'>MODELS DISAGREE</span>
              <span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                           color:{GRV["fg2"]};margin-left:8px'>
                RF={("DIS" if p_rf[0]==1 else "SAFE")} &nbsp;
                CNN={("DIS" if p_cnn[0]==1 else "SAFE")} &nbsp;
                BERT={("DIS" if p_bert[0]==1 else "SAFE")} —
                this tweet is ambiguous. BERT result is most reliable.
              </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"<div class='grv-label' style='margin-top:16px;margin-bottom:8px'>Confidence across models</div>",
                    unsafe_allow_html=True)
        fig, ax = gruvbox_fig(7, 2.2)
        model_names = ["Random Forest", "CNN-LSTM", "BERT"]
        probs_all   = [p_rf[1], p_cnn[1], p_bert[1]]
        colors_bar  = [GRV["red"] if p >= 0.5 else GRV["green"] for p in probs_all]
        bars = ax.barh(model_names, probs_all, color=colors_bar, height=0.4, edgecolor=GRV["bg"])
        ax.axvline(0.5, color=GRV["gray"], lw=1, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(disaster)", fontsize=8)
        ax.set_title("Disaster probability per model", fontsize=9, fontweight="bold")
        for bar, prob in zip(bars, probs_all):
            ax.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                    f"{prob*100:.1f}%", va="center", fontsize=8,
                    color=GRV["fg2"], fontfamily="monospace")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if run and tweet_input.strip():

        # ── API MODE ───────────────────────────────────────────────────────────
        if api_mode:
            try:
                with st.spinner("Calling API..."):
                    response = req.post(
                        f"{api_url.rstrip('/')}/classify",
                        json={"tweet": tweet_input},
                        timeout=30
                    )
                if response.status_code == 200:
                    data = response.json()

                    # Parse response into (pred, prob) tuples
                    p_rf   = (data["rf"]["label"],       data["rf"]["confidence"])
                    p_cnn  = (data["cnn_lstm"]["label"],  data["cnn_lstm"]["confidence"])
                    p_bert = (data["bert"]["label"],      data["bert"]["confidence"])

                    # ── Raw API response viewer ────────────────────────────────
                    with st.expander("API response (raw JSON)", expanded=False):
                        st.markdown(f"""
                        <div style='background:{GRV["bg"]};border:1px solid {GRV["bg2"]};
                                    border-radius:2px;padding:14px 16px;
                                    font-family:JetBrains Mono,monospace;font-size:0.75rem;
                                    color:{GRV["aqua"]};line-height:1.8;white-space:pre-wrap'>
POST {api_url}/classify
Status: {response.status_code} OK
Latency: {response.elapsed.total_seconds()*1000:.0f} ms

{_json.dumps(data, indent=2)}
                        </div>
                        """, unsafe_allow_html=True)

                    render_results(p_rf, p_cnn, p_bert)

                else:
                    st.error(f"API returned {response.status_code}: {response.text}")

            except req.exceptions.ConnectionError:
                st.error(f"Cannot connect to API at `{api_url}`. Make sure `uvicorn api:app --port 8000` is running.")
            except req.exceptions.Timeout:
                st.error("API request timed out after 30 seconds.")
            except Exception as e:
                st.error(f"API error: {e}")

        # ── DIRECT MODE ────────────────────────────────────────────────────────
        else:
            with st.spinner("Running all 3 models..."):
                p_rf   = predict_rf(tweet_input, rf, tfidf)
                p_cnn  = predict_cnn(tweet_input, cnn, vocab, device)
                p_bert = predict_bert(tweet_input, bert_tok, bert_model, device)

            render_results(p_rf, p_cnn, p_bert)

    elif run:
        st.warning("Please enter a tweet first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FEED MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Feed Monitor" in page:
    st.markdown("## 📡 Feed Monitor")
    st.markdown(f"<div style='color:{GRV['gray']};font-size:0.85rem;margin-bottom:20px'>"
                "Paste multiple tweets (one per line). BERT triages them in one pass. "
                "Adjust the alert threshold to control sensitivity.</div>",
                unsafe_allow_html=True)

    if not models_loaded:
        st.error(f"Could not load models: {load_error}")
        st.stop()

    SAMPLE = """Massive earthquake hits coastal region, tsunami warning issued
Just had the best pizza of my life downtown
Wildfire spreading rapidly, residents urged to evacuate immediately
lol my cat knocked over my coffee again
Flash flood warning in effect for the next 6 hours
The new season of that show is actually pretty good
Building collapse reported downtown, rescue teams on site
Traffic is insane this morning, going to be late
Tornado spotted near highway 40, seek shelter now
Anyone else excited for the weekend?"""

    col_left, col_right = st.columns([3, 1])
    with col_left:
        feed_input = st.text_area(
            "feed_input", height=220,
            placeholder="Paste tweets here, one per line...",
            label_visibility="collapsed"
        )
    with col_right:
        st.markdown(f"<div class='grv-label' style='margin-bottom:6px'>Alert threshold</div>",
                    unsafe_allow_html=True)
        threshold = st.slider("threshold", 0.5, 0.99, 0.75, 0.01,
                              label_visibility="collapsed")
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:1.4rem;
                    font-weight:700;color:{GRV["yellow"]};text-align:center'>
            {int(threshold*100)}%
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                    color:{GRV["gray"]};text-align:center;text-transform:uppercase;
                    letter-spacing:0.1em'>min confidence</div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load sample tweets", use_container_width=True):
            st.session_state["feed_sample"] = SAMPLE

    if "feed_sample" in st.session_state and not feed_input.strip():
        feed_input = st.session_state["feed_sample"]

    run_feed = st.button("▶  TRIAGE FEED", use_container_width=False)

    if run_feed and feed_input.strip():
        tweets = [t.strip() for t in feed_input.strip().split("\n") if t.strip()]

        with st.spinner(f"Running BERT on {len(tweets)} tweets..."):
            results = []
            for tw in tweets:
                pred, prob = predict_bert(tw, bert_tok, bert_model, device)
                results.append({
                    "tweet": tw,
                    "pred":  pred,
                    "prob":  prob,
                    "alert": pred == 1 and prob >= threshold,
                })

        n_disaster = sum(1 for r in results if r["pred"] == 1)
        n_alert    = sum(1 for r in results if r["alert"])
        n_safe     = len(results) - n_disaster

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total tweets",     len(results))
        s2.metric("Disaster flagged", n_disaster)
        s3.metric("High-conf alerts", n_alert)
        s4.metric("Safe / unclear",   n_safe)

        st.markdown("---")

        # ── Confidence distribution chart ──────────────────────────────────────
        all_probs = [r["prob"] for r in results]
        all_preds = [r["pred"] for r in results]
        if len(all_probs) > 1:
            fig = conf_distribution_chart(all_probs, all_preds)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("---")

        # ── Triage results ─────────────────────────────────────────────────────
        st.markdown(f"<div class='grv-label' style='margin-bottom:10px'>Triage results</div>",
                    unsafe_allow_html=True)

        for r in sorted(results, key=lambda x: -x["prob"]):
            if r["alert"]:
                border = GRV["red"]
                bg     = GRV["red"] + "18"
                badge  = f"<span style='background:{GRV['red']}22;color:{GRV['red']};border:1px solid {GRV['red']};font-family:JetBrains Mono,monospace;font-size:0.65rem;font-weight:700;letter-spacing:0.1em;padding:2px 8px;border-radius:2px'>ALERT</span>"
            elif r["pred"] == 1:
                border = GRV["orange"]
                bg     = GRV["bg1"]
                badge  = f"<span style='background:{GRV['orange']}22;color:{GRV['orange']};border:1px solid {GRV['orange']};font-family:JetBrains Mono,monospace;font-size:0.65rem;font-weight:700;letter-spacing:0.1em;padding:2px 8px;border-radius:2px'>DISASTER</span>"
            else:
                border = GRV["bg2"]
                bg     = GRV["bg1"]
                badge  = f"<span style='background:{GRV['green']}22;color:{GRV['green']};border:1px solid {GRV['green']};font-family:JetBrains Mono,monospace;font-size:0.65rem;font-weight:700;letter-spacing:0.1em;padding:2px 8px;border-radius:2px'>SAFE</span>"

            bar_w   = int(r["prob"] * 100)
            bar_col = GRV["red"] if r["pred"] == 1 else GRV["green"]
            highlighted = highlight_keywords(r["tweet"])

            st.markdown(f"""
            <div style='background:{bg};border:1px solid {border};border-radius:2px;
                        padding:12px 16px;margin-bottom:8px'>
              <div style='display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:6px'>
                <div style='font-family:IBM Plex Sans,sans-serif;font-size:0.85rem;
                            flex:1;margin-right:16px;line-height:1.6'>{highlighted}</div>
                <div style='display:flex;align-items:center;gap:10px;white-space:nowrap'>
                  {badge}
                  <span style='font-family:JetBrains Mono,monospace;font-size:0.9rem;
                               font-weight:700;color:{bar_col}'>{r["prob"]*100:.1f}%</span>
                </div>
              </div>
              <div style='background:{GRV["bg2"]};border-radius:1px;height:4px'>
                <div style='background:{bar_col};width:{bar_w}%;height:4px;border-radius:1px'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    elif run_feed:
        st.warning("Please paste at least one tweet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("## 📊 Model Dashboard")

    if not models_loaded:
        st.error(f"Could not load models: {load_error}")
        st.stop()

    metrics    = dash["metrics"]
    colors_map = {
        "Random Forest (Baseline)": GRV["green"],
        "CNN-LSTM":                 GRV["blue"],
        "BERT (bert-base-uncased)": GRV["purple"],
    }

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown(f"<div class='grv-label' style='margin-bottom:12px'>Performance metrics</div>",
                unsafe_allow_html=True)
    metric_keys = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"]
    for row in metrics:
        model_name = row.get("Model") or row.get("index", "")
        color      = colors_map.get(model_name, GRV["yellow"])
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                    color:{color};text-transform:uppercase;letter-spacing:0.1em;
                    margin-top:16px;margin-bottom:6px'>— {model_name}</div>
        """, unsafe_allow_html=True)
        cols = st.columns(len(metric_keys))
        for col, key in zip(cols, metric_keys):
            val = row.get(key, "N/A")
            col.metric(label=key, value=f"{float(val):.4f}" if val != "N/A" else "N/A")

    st.markdown("---")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown(f"<div class='grv-label' style='margin-bottom:8px'>All models comparison</div>",
                unsafe_allow_html=True)
    import numpy as np
    model_labels  = ["Random Forest", "CNN-LSTM", "BERT"]
    metric_colors = [GRV["blue"], GRV["orange"], GRV["green"], GRV["yellow"], GRV["purple"]]
    x     = np.arange(len(model_labels))
    width = 0.15
    fig_bar, ax_bar = gruvbox_fig(10, 4)
    for i, key in enumerate(metric_keys):
        vals = [float(row.get(key, 0)) for row in metrics]
        bars = ax_bar.bar(x + i * width, vals, width,
                          label=key, color=metric_colors[i],
                          alpha=0.85, edgecolor=GRV["bg"])
        for bar in bars:
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.004,
                        f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=7,
                        color=GRV["fg2"], fontfamily="monospace")
    ax_bar.set_xticks(x + width * 2)
    ax_bar.set_xticklabels(model_labels, fontsize=9, color=GRV["fg2"])
    ax_bar.set_ylim(0.5, 1.02)
    ax_bar.set_ylabel("Score", fontsize=8)
    ax_bar.set_title("All Models — RF vs CNN-LSTM vs BERT", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7, facecolor=GRV["bg1"], edgecolor=GRV["bg3"],
                  labelcolor=GRV["fg"], ncol=5)
    fig_bar.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    st.markdown("---")

    # ── Training curves ───────────────────────────────────────────────────────
    st.markdown(f"<div class='grv-label' style='margin-bottom:8px'>Training curves</div>",
                unsafe_allow_html=True)

    h_cnn  = dash.get("history_cnn")
    h_bert = dash.get("history_bert")

    if h_cnn or h_bert:
        curve_cols = st.columns(2)

        if h_cnn:
            with curve_cols[0]:
                st.markdown(f"""
                <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                            color:{GRV["blue"]};text-transform:uppercase;
                            letter-spacing:0.1em;margin-bottom:6px'>CNN-LSTM</div>
                """, unsafe_allow_html=True)
                fig_c, axes_c = plt.subplots(1, 2, figsize=(8, 3))
                fig_c.patch.set_facecolor(GRV["bg1"])
                for ax in axes_c:
                    ax.set_facecolor(GRV["bg"])
                    ax.tick_params(colors=GRV["fg2"], labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(GRV["bg3"])
                    ax.grid(color=GRV["bg2"], linestyle="--", linewidth=0.5, alpha=0.7)
                ep = range(1, len(h_cnn["train_loss"]) + 1)
                axes_c[0].plot(ep, h_cnn["train_loss"], color=GRV["blue"],   lw=1.5, marker="o", ms=3, label="Train")
                axes_c[0].plot(ep, h_cnn["val_loss"],   color=GRV["orange"], lw=1.5, marker="o", ms=3, label="Val")
                axes_c[0].set_title("Loss", fontsize=8, fontweight="bold", color=GRV["yellow"])
                axes_c[0].set_xlabel("Epoch", fontsize=7, color=GRV["fg2"])
                axes_c[0].legend(fontsize=6, facecolor=GRV["bg1"], edgecolor=GRV["bg3"], labelcolor=GRV["fg"])
                axes_c[1].plot(ep, h_cnn["train_acc"], color=GRV["blue"],   lw=1.5, marker="o", ms=3, label="Train")
                axes_c[1].plot(ep, h_cnn["val_acc"],   color=GRV["orange"], lw=1.5, marker="o", ms=3, label="Val")
                axes_c[1].set_title("Accuracy", fontsize=8, fontweight="bold", color=GRV["yellow"])
                axes_c[1].set_xlabel("Epoch", fontsize=7, color=GRV["fg2"])
                axes_c[1].legend(fontsize=6, facecolor=GRV["bg1"], edgecolor=GRV["bg3"], labelcolor=GRV["fg"])
                fig_c.tight_layout()
                st.pyplot(fig_c, use_container_width=True)
                plt.close(fig_c)

        if h_bert:
            with curve_cols[1]:
                st.markdown(f"""
                <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                            color:{GRV["purple"]};text-transform:uppercase;
                            letter-spacing:0.1em;margin-bottom:6px'>BERT</div>
                """, unsafe_allow_html=True)
                fig_b, axes_b = plt.subplots(1, 2, figsize=(8, 3))
                fig_b.patch.set_facecolor(GRV["bg1"])
                for ax in axes_b:
                    ax.set_facecolor(GRV["bg"])
                    ax.tick_params(colors=GRV["fg2"], labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(GRV["bg3"])
                    ax.grid(color=GRV["bg2"], linestyle="--", linewidth=0.5, alpha=0.7)
                ep_b = range(1, len(h_bert["train_loss"]) + 1)
                axes_b[0].plot(ep_b, h_bert["train_loss"], color=GRV["purple"], lw=1.5, marker="o", ms=3, label="Train")
                axes_b[0].plot(ep_b, h_bert["val_loss"],   color=GRV["orange"], lw=1.5, marker="o", ms=3, label="Val")
                axes_b[0].set_title("Loss", fontsize=8, fontweight="bold", color=GRV["yellow"])
                axes_b[0].set_xlabel("Epoch", fontsize=7, color=GRV["fg2"])
                axes_b[0].legend(fontsize=6, facecolor=GRV["bg1"], edgecolor=GRV["bg3"], labelcolor=GRV["fg"])
                axes_b[1].plot(ep_b, h_bert["train_acc"], color=GRV["purple"], lw=1.5, marker="o", ms=3, label="Train")
                axes_b[1].plot(ep_b, h_bert["val_acc"],   color=GRV["orange"], lw=1.5, marker="o", ms=3, label="Val")
                axes_b[1].set_title("Accuracy", fontsize=8, fontweight="bold", color=GRV["yellow"])
                axes_b[1].set_xlabel("Epoch", fontsize=7, color=GRV["fg2"])
                axes_b[1].legend(fontsize=6, facecolor=GRV["bg1"], edgecolor=GRV["bg3"], labelcolor=GRV["fg"])
                fig_b.tight_layout()
                st.pyplot(fig_b, use_container_width=True)
                plt.close(fig_b)
    else:
        st.markdown(f"""
        <div style='background:{GRV["bg1"]};border:1px solid {GRV["bg2"]};
                    border-radius:2px;padding:14px 18px;
                    font-family:JetBrains Mono,monospace;font-size:0.75rem;
                    color:{GRV["gray"]}'>
          Training curves not found in dashboard_data.json.<br>
          Re-run the save artifacts cell in the notebook to include them.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── ROC + Confusion matrix ────────────────────────────────────────────────
    col_roc, col_cm = st.columns([1.2, 1])

    with col_roc:
        st.markdown(f"<div class='grv-label' style='margin-bottom:8px'>ROC curves</div>",
                    unsafe_allow_html=True)
        fig_r, ax_r = gruvbox_fig(6, 4)
        for key, name, color in [
            ("rf",   "Random Forest", GRV["green"]),
            ("cnn",  "CNN-LSTM",      GRV["blue"]),
            ("bert", "BERT",          GRV["purple"]),
        ]:
            d       = dash[key]
            roc_auc = next((r.get("ROC-AUC") for r in metrics
                            if key in (r.get("Model","") or "").lower()), None)
            label   = f"{name} (AUC={float(roc_auc):.3f})" if roc_auc else name
            ax_r.plot(d["fpr"], d["tpr"], color=color, lw=2, label=label)
        ax_r.plot([0,1],[0,1], color=GRV["bg3"], lw=1, linestyle="--", label="Random")
        ax_r.set_xlabel("False Positive Rate", fontsize=8)
        ax_r.set_ylabel("True Positive Rate",  fontsize=8)
        ax_r.set_title("ROC curves", fontsize=10, fontweight="bold")
        ax_r.legend(fontsize=7, facecolor=GRV["bg1"], edgecolor=GRV["bg3"],
                    labelcolor=GRV["fg"])
        ax_r.set_xlim([0,1]); ax_r.set_ylim([0,1.02])
        fig_r.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    with col_cm:
        st.markdown(f"<div class='grv-label' style='margin-bottom:8px'>Confusion matrices</div>",
                    unsafe_allow_html=True)
        cm_select = st.selectbox("cm_model", ["Random Forest", "CNN-LSTM", "BERT"],
                                 label_visibility="collapsed")
        cm_key  = {"Random Forest": "rf", "CNN-LSTM": "cnn", "BERT": "bert"}[cm_select]
        cm_data = np.array(dash["cm"][cm_key])
        fig_m, ax_m = gruvbox_fig(4.5, 3.8)
        ax_m.imshow(cm_data, cmap="YlOrBr", aspect="auto")
        ax_m.set_xticks([0,1]); ax_m.set_yticks([0,1])
        ax_m.set_xticklabels(["Not Disaster","Disaster"], fontsize=7, color=GRV["fg2"])
        ax_m.set_yticklabels(["Not Disaster","Disaster"], fontsize=7, color=GRV["fg2"])
        ax_m.set_xlabel("Predicted", fontsize=8); ax_m.set_ylabel("Actual", fontsize=8)
        ax_m.set_title(f"{cm_select}", fontsize=9, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax_m.text(j, i, str(cm_data[i,j]),
                          ha="center", va="center",
                          color=GRV["bg"] if cm_data[i,j] > cm_data.max()/2 else GRV["fg"],
                          fontsize=13, fontweight="bold", fontfamily="JetBrains Mono")
        fig_m.tight_layout()
        st.pyplot(fig_m, use_container_width=True)
        plt.close(fig_m)

    st.markdown("---")
    st.markdown(f"<div class='grv-label' style='margin-bottom:8px'>Model complexity trade-off</div>",
                unsafe_allow_html=True)
    import pandas as pd
    cmp_df = pd.DataFrame({
        "Model":         ["Random Forest", "CNN-LSTM", "BERT"],
        "Params":        ["N/A (trees)",   "~750K",    "~110M"],
        "Size":          ["< 1 MB",        "~3 MB",    "~440 MB"],
        "Train Speed":   ["< 5 sec",       "~30 sec",  "~5 min"],
        "Accuracy":      [row["Accuracy"] for row in metrics],
        "ROC-AUC":       [row.get("ROC-AUC","—") for row in metrics],
    })
    st.dataframe(cmp_df.set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif "Architecture" in page:
    st.markdown("## 🧠 Architecture & Math")

    tab1, tab2, tab3 = st.tabs(["  Random Forest  ", "  CNN-LSTM  ", "  BERT  "])

    # ── RF ────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f"""
        <svg width="100%" viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg"
             style="max-height:280px">
          <defs>
            <marker id="a1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M2 1L8 5L2 9" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </marker>
          </defs>
          <rect x="215" y="20" width="250" height="44" rx="8" fill="{GRV['bg2']}" stroke="{GRV['bg3']}" stroke-width="0.5"/>
          <text x="340" y="42" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg']}" font-size="13" font-family="JetBrains Mono,monospace">Raw tweet</text>
          <line x1="340" y1="64" x2="340" y2="88" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a1)"/>
          <rect x="190" y="90" width="300" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['aqua']}" stroke-width="0.8"/>
          <text x="340" y="111" text-anchor="middle" dominant-baseline="central" fill="{GRV['aqua']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">TF-IDF vectoriser</text>
          <text x="340" y="131" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">10,000-dim sparse vector</text>
          <path d="M260 146 L260 176 L120 176 L120 194" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a1)"/>
          <path d="M340 146 L340 194" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a1)"/>
          <path d="M420 146 L420 176 L560 176 L560 194" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a1)"/>
          <rect x="50" y="196" width="140" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['green']}" stroke-width="0.8"/>
          <text x="120" y="217" text-anchor="middle" dominant-baseline="central" fill="{GRV['green']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">Tree 1</text>
          <text x="120" y="237" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">Gini splits</text>
          <rect x="270" y="196" width="140" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['green']}" stroke-width="0.8"/>
          <text x="340" y="217" text-anchor="middle" dominant-baseline="central" fill="{GRV['green']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">Tree 2 … 100</text>
          <text x="340" y="237" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">Bootstrap sample</text>
          <rect x="490" y="196" width="140" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['green']}" stroke-width="0.8"/>
          <text x="560" y="217" text-anchor="middle" dominant-baseline="central" fill="{GRV['green']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">Tree 200</text>
          <text x="560" y="237" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">Gini splits</text>
          <path d="M120 252 L120 276 L280 276 L280 288" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a1)"/>
          <path d="M340 252 L340 288" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a1)"/>
          <path d="M560 252 L560 276 L400 276 L400 288" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a1)"/>
          <rect x="215" y="290" width="250" height="30" rx="8" fill="{GRV['bg2']}" stroke="{GRV['yellow']}" stroke-width="0.8"/>
          <text x="340" y="305" text-anchor="middle" dominant-baseline="central" fill="{GRV['yellow']}" font-size="12" font-weight="500" font-family="JetBrains Mono,monospace">Majority vote → P(disaster)</text>
        </svg>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:{GRV["bg1"]};border:1px solid {GRV["bg2"]};border-radius:2px;
                    padding:20px 24px;margin-top:12px'>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;
                      font-size:0.82rem;line-height:1.8;color:{GRV["fg2"]}'>
            <div>
              <div style='color:{GRV["aqua"]};font-family:JetBrains Mono,monospace;
                          font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;
                          margin-bottom:6px'>How it works</div>
              Each tweet is converted to a 10,000-dim TF-IDF vector capturing word and bigram frequencies.
              200 decision trees each trained on a random bootstrap sample vote on the label.
              Final probability = fraction of trees voting "disaster".<br>
              No neural network — pure statistical learning.
            </div>
            <div>
              <div style='color:{GRV["aqua"]};font-family:JetBrains Mono,monospace;
                          font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;
                          margin-bottom:6px'>Key formulas</div>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                tfidf(t,d) = tf(t,d) × log(N / df(t))
              </span><br>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                Gini = 1 − (p² + (1−p)²)
              </span><br>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                P(disaster) = votes_disaster / 200
              </span><br><br>
              Strong baseline because tweets are short (~15 tokens) and
              TF-IDF bigrams directly capture discriminative phrases like
              "evacuation order" or "rescue team".
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── CNN-LSTM ──────────────────────────────────────────────────────────────
    with tab2:
        st.markdown(f"""
        <svg width="100%" viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg"
             style="max-height:320px">
          <defs>
            <marker id="a2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M2 1L8 5L2 9" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </marker>
          </defs>
          <rect x="215" y="20" width="250" height="44" rx="8" fill="{GRV['bg2']}" stroke="{GRV['bg3']}" stroke-width="0.5"/>
          <text x="340" y="42" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg']}" font-size="13" font-family="JetBrains Mono,monospace">Integer token sequence</text>
          <text x="492" y="42" dominant-baseline="central" fill="{GRV['gray']}" font-size="11" font-family="JetBrains Mono,monospace">(batch, 30)</text>
          <line x1="340" y1="64" x2="340" y2="88" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a2)"/>
          <rect x="215" y="90" width="250" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['purple']}" stroke-width="0.8"/>
          <text x="340" y="111" text-anchor="middle" dominant-baseline="central" fill="{GRV['purple']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">Embedding layer</text>
          <text x="340" y="131" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">GloVe-100 pretrained</text>
          <text x="492" y="118" dominant-baseline="central" fill="{GRV['gray']}" font-size="11" font-family="JetBrains Mono,monospace">(batch, 30, 100)</text>
          <line x1="340" y1="146" x2="340" y2="170" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a2)"/>
          <rect x="215" y="172" width="250" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['purple']}" stroke-width="0.8"/>
          <text x="340" y="193" text-anchor="middle" dominant-baseline="central" fill="{GRV['purple']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">Conv1D + MaxPool</text>
          <text x="340" y="213" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">128 filters, kernel=3</text>
          <text x="492" y="200" dominant-baseline="central" fill="{GRV['gray']}" font-size="11" font-family="JetBrains Mono,monospace">(batch, 128, 15)</text>
          <line x1="340" y1="228" x2="340" y2="252" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a2)"/>
          <rect x="215" y="254" width="250" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['purple']}" stroke-width="0.8"/>
          <text x="340" y="275" text-anchor="middle" dominant-baseline="central" fill="{GRV['purple']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">LSTM</text>
          <text x="340" y="295" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">hidden dim=64, final h_T</text>
          <text x="492" y="282" dominant-baseline="central" fill="{GRV['gray']}" font-size="11" font-family="JetBrains Mono,monospace">(batch, 64)</text>
          <line x1="340" y1="310" x2="340" y2="334" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a2)"/>
          <rect x="215" y="336" width="250" height="44" rx="8" fill="{GRV['bg2']}" stroke="{GRV['yellow']}" stroke-width="0.8"/>
          <text x="340" y="358" text-anchor="middle" dominant-baseline="central" fill="{GRV['yellow']}" font-size="12" font-weight="500" font-family="JetBrains Mono,monospace">Linear + sigmoid → P(disaster)</text>
          <text x="492" y="358" dominant-baseline="central" fill="{GRV['gray']}" font-size="11" font-family="JetBrains Mono,monospace">(batch, 1)</text>
        </svg>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:{GRV["bg1"]};border:1px solid {GRV["bg2"]};border-radius:2px;
                    padding:20px 24px;margin-top:12px'>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;
                      font-size:0.82rem;line-height:1.8;color:{GRV["fg2"]}'>
            <div>
              <div style='color:{GRV["aqua"]};font-family:JetBrains Mono,monospace;
                          font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;
                          margin-bottom:6px'>How it works</div>
              Tokens are looked up in a 100-dim GloVe embedding table pretrained on 27B tweets.
              Conv1D extracts local n-gram patterns (each filter = one 3-word pattern detector).
              MaxPool halves the sequence. LSTM reads left-to-right capturing order and context.
              Final hidden state h_T is the tweet's summary vector fed to the classifier.
            </div>
            <div>
              <div style='color:{GRV["aqua"]};font-family:JetBrains Mono,monospace;
                          font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;
                          margin-bottom:6px'>Key formulas</div>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                conv(x) = ReLU(W * x + b)
              </span><br>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                c_t = f_t ⊙ c_(t-1) + i_t ⊙ tanh(W_c·[h_(t-1), x_t])
              </span><br>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                BCE = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
              </span><br><br>
              Trained with Adam lr=1e-3, L2 weight decay, gradient clipping,
              ReduceLROnPlateau, and EarlyStopping (patience=4).
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── BERT ──────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown(f"""
        <svg width="100%" viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg"
             style="max-height:340px">
          <defs>
            <marker id="a3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M2 1L8 5L2 9" fill="none" stroke="{GRV['gray']}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </marker>
          </defs>
          <rect x="215" y="20" width="250" height="44" rx="8" fill="{GRV['bg2']}" stroke="{GRV['bg3']}" stroke-width="0.5"/>
          <text x="340" y="42" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg']}" font-size="13" font-family="JetBrains Mono,monospace">Raw tweet</text>
          <line x1="340" y1="64" x2="340" y2="88" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a3)"/>
          <rect x="215" y="90" width="250" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['orange']}" stroke-width="0.8"/>
          <text x="340" y="111" text-anchor="middle" dominant-baseline="central" fill="{GRV['orange']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">WordPiece tokeniser</text>
          <text x="340" y="131" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">[CLS] tokens [SEP], pad to 64</text>
          <line x1="340" y1="146" x2="340" y2="170" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a3)"/>
          <rect x="215" y="172" width="250" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['orange']}" stroke-width="0.8"/>
          <text x="340" y="193" text-anchor="middle" dominant-baseline="central" fill="{GRV['orange']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">12 transformer layers</text>
          <text x="340" y="213" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">self-attention + FFN × 12</text>
          <line x1="340" y1="228" x2="340" y2="252" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a3)"/>
          <rect x="215" y="254" width="250" height="56" rx="8" fill="{GRV['bg2']}" stroke="{GRV['orange']}" stroke-width="0.8"/>
          <text x="340" y="275" text-anchor="middle" dominant-baseline="central" fill="{GRV['orange']}" font-size="13" font-weight="500" font-family="JetBrains Mono,monospace">[CLS] hidden state</text>
          <text x="340" y="295" text-anchor="middle" dominant-baseline="central" fill="{GRV['fg2']}" font-size="11" font-family="JetBrains Mono,monospace">768-dim tweet representation</text>
          <text x="492" y="282" dominant-baseline="central" fill="{GRV['gray']}" font-size="11" font-family="JetBrains Mono,monospace">(batch, 768)</text>
          <line x1="340" y1="310" x2="340" y2="334" stroke="{GRV['gray']}" stroke-width="1.5" marker-end="url(#a3)"/>
          <rect x="215" y="336" width="250" height="44" rx="8" fill="{GRV['bg2']}" stroke="{GRV['yellow']}" stroke-width="0.8"/>
          <text x="340" y="358" text-anchor="middle" dominant-baseline="central" fill="{GRV['yellow']}" font-size="12" font-weight="500" font-family="JetBrains Mono,monospace">Linear + softmax → class</text>
          <text x="492" y="358" dominant-baseline="central" fill="{GRV['gray']}" font-size="11" font-family="JetBrains Mono,monospace">(batch, 2)</text>
          <rect x="56" y="168" width="76" height="64" rx="6" fill="none" stroke="{GRV['yellow']}" stroke-width="0.8" stroke-dasharray="4 3"/>
          <text x="94" y="193" text-anchor="middle" dominant-baseline="central" fill="{GRV['yellow']}" font-size="11" font-family="JetBrains Mono,monospace">110M</text>
          <text x="94" y="213" text-anchor="middle" dominant-baseline="central" fill="{GRV['gray']}" font-size="11" font-family="JetBrains Mono,monospace">params</text>
        </svg>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:{GRV["bg1"]};border:1px solid {GRV["bg2"]};border-radius:2px;
                    padding:20px 24px;margin-top:12px'>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;
                      font-size:0.82rem;line-height:1.8;color:{GRV["fg2"]}'>
            <div>
              <div style='color:{GRV["aqua"]};font-family:JetBrains Mono,monospace;
                          font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;
                          margin-bottom:6px'>How it works</div>
              WordPiece tokenises the tweet and prepends [CLS]. Each of 12 transformer
              layers runs multi-head self-attention — every token attends to every other
              token simultaneously. The [CLS] token aggregates the full context.
              A linear head maps its 768-dim vector to 2 class scores.
            </div>
            <div>
              <div style='color:{GRV["aqua"]};font-family:JetBrains Mono,monospace;
                          font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;
                          margin-bottom:6px'>Key formulas</div>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V
              </span><br>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                logits = W · h[CLS] + b
              </span><br>
              <span style='font-family:JetBrains Mono,monospace;color:{GRV["yellow"]}'>
                p = softmax(logits)
              </span><br><br>
              Fine-tuned with AdamW lr=2e-5, linear warmup (10% of steps),
              linear decay, 4 epochs. Small LR prevents catastrophic forgetting
              of the 3.3B-word pretraining.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)