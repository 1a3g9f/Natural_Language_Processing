# 🎬 Movie Review Intelligence

An interactive Streamlit dashboard covering **10 NLP concepts** on 2,000 movie reviews:
text exploration · preprocessing · POS & NER · TF-IDF/n-grams · ML classification ·
VADER · word embeddings · LDA topics · cosine similarity · interactive frontend.

**Live demo:** _add your Streamlit Community Cloud link here_
**Screenshots:** _add 2–3 images here (Sentiment Lab, Model Performance, Word Explorer)_

## Run locally
```bash
pip install -r requirements.txt
python train.py          # optional: pre-builds models/ (~1 min). The app does this itself if missing.
streamlit run app.py
```

## Structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (7 tabs) |
| `nlp_utils.py` | `preprocess()`, NER, explanations, word highlighting, shared by training and app |
| `train.py` | Runs the notebook pipeline once, saves to `models/` |
| `models/` | Saved vectorizers, classifier, LDA, word embeddings |
| `.streamlit/config.toml` | Theme |

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub → share.streamlit.io → New app → pick `app.py`. NLTK data downloads automatically.

## Limitations
Sarcasm fools every model here. Next step: a transformer (DistilBERT) compare-models toggle.
