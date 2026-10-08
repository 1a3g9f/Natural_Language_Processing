"""Shared NLP helpers. Used by BOTH train.py and app.py so the app
preprocesses text exactly like the notebook / training pipeline."""
import html
import re
from functools import lru_cache

import nltk

NLTK_PACKAGES = [
    "movie_reviews", "stopwords", "wordnet", "omw-1.4", "vader_lexicon",
    "averaged_perceptron_tagger_eng", "maxent_ne_chunker_tab", "words",
]


@lru_cache(maxsize=1)
def ensure_nltk():
    """Download NLTK data once (works on a fresh machine / Streamlit Cloud)."""
    for pkg in NLTK_PACKAGES:
        nltk.download(pkg, quiet=True)
    return True


ensure_nltk()

from nltk.corpus import stopwords  # noqa: E402  (needs data first)
from nltk.stem import PorterStemmer, WordNetLemmatizer  # noqa: E402
from nltk.tokenize import RegexpTokenizer  # noqa: E402

tokenizer = RegexpTokenizer(r"[a-z]+")
# Keep negations: removing "not" would turn "not good" into "good"
stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """lowercase -> tokenize -> drop stopwords -> lemmatize (same as notebook)."""
    tokens = tokenizer.tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)


def entities_of(text: str):
    """Named entities as (label, text) pairs using NLTK's ne_chunk."""
    toks = re.findall(r"[A-Za-z]+", text)
    return [
        (c.label(), " ".join(w for w, _ in c))
        for c in nltk.ne_chunk(nltk.pos_tag(toks))
        if hasattr(c, "label")
    ]


def pos_tags(text: str):
    return nltk.pos_tag(re.findall(r"[A-Za-z]+", text))


def explain(text, tfidf, clf, top_k=6):
    """Top words/bigrams that pushed the model positive/negative.
    contribution = TF-IDF value x logistic-regression weight (as in the notebook)."""
    vec = tfidf.transform([preprocess(text)])
    if vec.nnz == 0:
        return [], []
    feats = tfidf.get_feature_names_out()
    contrib = vec.data * clf.coef_[0][vec.indices]
    ranked = sorted(zip(feats[vec.indices], contrib), key=lambda x: x[1])
    neg = [(w, c) for w, c in ranked if c < 0][:top_k]
    pos = [(w, c) for w, c in ranked[::-1] if c > 0][:top_k]
    return pos, neg


def highlight_html(text, tfidf, clf):
    """Return the review as HTML with each word shaded green/red by its model weight."""
    feats = tfidf.get_feature_names_out()
    weight = dict(zip(feats, clf.coef_[0]))
    scale = max(abs(clf.coef_[0])) or 1.0
    out = []
    for token in re.findall(r"[A-Za-z']+|[^A-Za-z']+", text):
        if not re.match(r"[A-Za-z']", token):
            out.append(html.escape(token))
            continue
        key = lemmatizer.lemmatize(token.lower())
        w = weight.get(key, 0.0)
        if abs(w) / scale < 0.04:
            out.append(html.escape(token))
            continue
        alpha = min(0.15 + 0.85 * abs(w) / (0.6 * scale), 0.95)
        rgb = "46,204,113" if w > 0 else "231,76,60"
        out.append(
            f'<span class="hl" style="background:rgba({rgb},{alpha:.2f})" '
            f'title="weight {w:+.2f}">{html.escape(token)}</span>'
        )
    return "".join(out)


def chip(text, color="#555"):
    return f'<span class="chip" style="background:{color}">{html.escape(str(text))}</span>'
