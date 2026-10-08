"""Runs the notebook's full pipeline once and saves everything to models/.
    python train.py
The app calls train_all() automatically if models/ is empty."""
import re
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from embeddings import WordVectors
from nltk.corpus import movie_reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from nlp_utils import preprocess, stop_words, tokenizer

MODELS_DIR = Path(__file__).parent / "models"
ARTIFACTS = MODELS_DIR / "artifacts.joblib"


def train_all(progress=lambda msg: None):
    MODELS_DIR.mkdir(exist_ok=True)

    progress("Loading 2,000 movie reviews...")
    texts, labels = [], []
    for category in movie_reviews.categories():
        for fid in movie_reviews.fileids(category):
            texts.append(movie_reviews.raw(fid))
            labels.append(1 if category == "pos" else 0)
    labels = np.array(labels)

    progress("Exploring text...")
    all_tokens = [t for txt in texts for t in tokenizer.tokenize(txt.lower())]
    content_tokens = [t for t in all_tokens if t not in stop_words and len(t) > 2]
    stats = {
        "n_reviews": len(texts),
        "n_pos": int(labels.sum()),
        "n_neg": int(len(labels) - labels.sum()),
        "total_words": len(all_tokens),
        "vocab_size": len(set(all_tokens)),
        "zipf": sorted(Counter(all_tokens).values(), reverse=True)[:300],
        "top_words": Counter(content_tokens).most_common(15),
        "top_bigrams": [
            (" ".join(b), c)
            for b, c in Counter(zip(content_tokens, content_tokens[1:])).most_common(12)
        ],
    }

    progress("Preprocessing (lemmatize, keep 'not')...")
    clean_texts = [preprocess(t) for t in texts]

    progress("Training 4 classifiers (80/20 split)...")
    idx = np.arange(len(texts))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    X_train = [clean_texts[i] for i in tr]
    X_test = [clean_texts[i] for i in te]
    y_train, y_test = labels[tr], labels[te]

    bow = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr_bow, Xte_bow = bow.fit_transform(X_train), bow.transform(X_test)
    Xtr_tf, Xte_tf = tfidf.fit_transform(X_train), tfidf.transform(X_test)

    setups = {
        "Naive Bayes + BoW": (MultinomialNB(), Xtr_bow, Xte_bow),
        "Naive Bayes + TF-IDF": (MultinomialNB(), Xtr_tf, Xte_tf),
        "LogReg + BoW": (LogisticRegression(max_iter=1000), Xtr_bow, Xte_bow),
        "LogReg + TF-IDF": (LogisticRegression(max_iter=1000), Xtr_tf, Xte_tf),
    }
    results = {}
    for name, (model, Xtr, Xte) in setups.items():
        model.fit(Xtr, y_train)
        results[name] = float(accuracy_score(y_test, model.predict(Xte)))

    clf = setups["LogReg + TF-IDF"][0]
    y_pred = clf.predict(Xte_tf)
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    progress("Scoring VADER on the same test set...")
    vader = SentimentIntensityAnalyzer()
    vader_pred = [1 if vader.polarity_scores(texts[i])["compound"] > 0 else 0 for i in te]
    results["VADER (no training)"] = float(accuracy_score(y_test, vader_pred))

    progress("Top adjectives per class (POS tagging)...")
    def top_adjectives(label, n_docs=150, k=12):
        ids = np.where(labels == label)[0][:n_docs]
        adjs = []
        for i in ids:
            toks = re.findall(r"[a-z]+", texts[i].lower())[:250]
            adjs += [w for w, tag in nltk.pos_tag(toks)
                     if tag == "JJ" and w not in stop_words and len(w) > 2]
        return Counter(adjs).most_common(k)

    adjectives = {"neg": top_adjectives(0), "pos": top_adjectives(1)}

    progress("Training word embeddings...")
    sentences = [tokenizer.tokenize(s.lower()) for s in texts]
    w2v = WordVectors(sentences, vector_size=100, window=5, min_count=5, seed=42)

    progress("Fitting LDA topics...")
    topic_vec = CountVectorizer(max_df=0.4, min_df=10, max_features=3000)
    topic_matrix = topic_vec.fit_transform(clean_texts)
    lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=15).fit(topic_matrix)
    vocab = np.array(topic_vec.get_feature_names_out())
    topic_words = [list(vocab[c.argsort()[-10:][::-1]]) for c in lda.components_]

    progress("Building similarity index...")
    sim_vec = TfidfVectorizer(max_features=5000, stop_words="english")
    all_tfidf = sim_vec.fit_transform(clean_texts)

    artifacts = dict(
        texts=texts, labels=labels, stats=stats, w2v=w2v, tfidf=tfidf, clf=clf,
        results=results, report=report, cm=cm, adjectives=adjectives,
        topic_vec=topic_vec, lda=lda, topic_words=topic_words,
        sim_vec=sim_vec, all_tfidf=all_tfidf, trained_at=time.strftime("%Y-%m-%d %H:%M"),
    )
    joblib.dump(artifacts, ARTIFACTS, compress=3)
    progress("Done.")
    return artifacts


def load_all():
    return joblib.load(ARTIFACTS)


def models_exist():
    return ARTIFACTS.exists()


if __name__ == "__main__":
    t0 = time.time()
    train_all(progress=lambda m: print("•", m))
    print(f"Saved to {MODELS_DIR} in {time.time() - t0:.0f}s")
    for k, v in load_all()["results"].items():
        print(f"  {k:<24} {v:.3f}")
