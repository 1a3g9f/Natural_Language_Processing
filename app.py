"""🎬 Movie Review Intelligence — Streamlit frontend for the NLP notebook.
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import train
from nlp_utils import (chip, entities_of, explain, highlight_html, lemmatizer,
                       pos_tags, preprocess, stemmer, stop_words, tokenizer)

st.set_page_config(page_title="Movie Review Intelligence", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

# ───────────────────────── Design tokens ─────────────────────────
GREEN, RED, BLUE, AMBER, PURPLE = "#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"
TEAL, PINK = "#16a085", "#e84393"
INK, MUTED = "#E9EDF6", "#93A0B8"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.markdown("""
<style>
  /* ── shell ─────────────────────────────────────────── */
  .stApp {
    background:
      radial-gradient(1100px 520px at 10% -12%, rgba(46,204,113,.11), transparent 62%),
      radial-gradient(900px 480px at 102% -4%, rgba(52,152,219,.11), transparent 58%),
      radial-gradient(700px 420px at 60% 110%, rgba(155,89,182,.09), transparent 60%),
      #0B0F16;
  }
  .block-container {padding-top: 1.5rem; padding-bottom: 2.6rem; max-width: 1240px;}
  h1, h2, h3, h4 {letter-spacing: -.2px;}
  code {background: rgba(255,255,255,.07); border-radius: 6px; padding: 1px 6px; color: #9be7bd;}

  /* ── hero ──────────────────────────────────────────── */
  .hero {border:1px solid rgba(255,255,255,.10); border-radius:22px; padding:24px 28px;
         background:linear-gradient(135deg, rgba(46,204,113,.14), rgba(52,152,219,.10) 48%, rgba(155,89,182,.14));
         box-shadow:0 18px 50px -30px rgba(0,0,0,.9); margin-bottom:16px;}
  .hero h1 {font-size:2.35rem; font-weight:800; margin:0; line-height:1.15;
            background:linear-gradient(92deg,#ffffff,#9be7bd 56%,#8fc9ff);
            -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
  .hero p {opacity:.76; margin:.45rem 0 0 0; font-size:1.02rem;}
  .hero .badges {margin-top:14px;}
  .badge {display:inline-block; font-size:.74rem; letter-spacing:.03em; padding:4px 12px; margin:0 6px 6px 0;
          border-radius:999px; border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.055); color:#c9d4e6;}

  /* ── KPI cards ─────────────────────────────────────── */
  .kpi {border:1px solid rgba(255,255,255,.09); border-radius:16px; padding:15px 17px; height:100%;
        background:linear-gradient(180deg, rgba(255,255,255,.062), rgba(255,255,255,.018));
        box-shadow:0 14px 34px -26px rgba(0,0,0,.95);}
  .kpi .v {font-size:1.72rem; font-weight:750; line-height:1.15;}
  .kpi .l {font-size:.75rem; text-transform:uppercase; letter-spacing:.09em; opacity:.62; margin-top:3px;}
  .kpi .bar {height:3px; border-radius:3px; margin-top:11px;
             background:linear-gradient(90deg, var(--c), rgba(255,255,255,0));}

  /* ── generic blocks ────────────────────────────────── */
  .card {border:1px solid rgba(255,255,255,.09); background:rgba(255,255,255,.035);
         border-radius:14px; padding:14px 18px; margin:8px 0;}
  .card h4 {margin:0 0 8px 0; font-size:.97rem;}
  .chip {color:#fff; padding:3px 11px; border-radius:999px; margin:3px 4px 3px 0;
         display:inline-block; font-size:13px; font-weight:500;}
  .hl {padding:1px 3px; border-radius:4px;}
  .review-box {line-height:2.05; font-size:1.05rem; padding:16px 20px; border-radius:14px;
               border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.035);}
  .concept {border:1px solid rgba(255,255,255,.09); border-radius:14px; padding:12px 15px;
            margin-bottom:9px; background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.018));
            transition:transform .16s ease, border-color .16s ease; height:100%;}
  .concept:hover {transform:translateY(-2px); border-color:rgba(46,204,113,.42);}
  .concept b {font-size:.93rem; display:block;}
  .concept span {opacity:.68; font-size:.82rem;}
  .concept .num {display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.08em;
                 color:#9be7bd; margin-bottom:5px;}
  .match-score {float:right; opacity:.75; font-size:.86rem;}

  /* ── section headers ───────────────────────────────── */
  .sect {margin:6px 0 12px 0;}
  .sect h3 {margin:0; font-size:1.22rem; font-weight:700;}
  .sect p {margin:.25rem 0 0 0; opacity:.66; font-size:.92rem;}
  .gauge-title {text-align:center; font-weight:650; font-size:.98rem; opacity:.94; margin-bottom:-4px;}

  /* ── pills / banners ───────────────────────────────── */
  .pill-wrap {text-align:center; margin-top:-6px;}
  .pill {display:inline-flex; align-items:center; gap:9px; padding:8px 16px; border-radius:999px;
         font-weight:600; font-size:.94rem; border:1px solid rgba(255,255,255,.14);
         background:rgba(255,255,255,.055);}
  .pill small {opacity:.68; font-weight:500;}
  .banner {border-radius:14px; padding:13px 18px; margin:14px 0 6px 0; font-size:.95rem;
           border:1px solid rgba(255,255,255,.12);}
  .banner.ok {background:linear-gradient(90deg, rgba(46,204,113,.16), rgba(46,204,113,.04));
              border-color:rgba(46,204,113,.38);}
  .banner.warn {background:linear-gradient(90deg, rgba(243,156,18,.18), rgba(243,156,18,.04));
                border-color:rgba(243,156,18,.42);}
  .banner b {font-weight:700;}

  /* ── step cards (preprocessing) ────────────────────── */
  .step {border:1px solid rgba(255,255,255,.09); border-radius:14px; padding:14px 17px; margin-bottom:10px;
         background:rgba(255,255,255,.032);}
  .step .hd {display:flex; align-items:center; gap:10px; margin-bottom:8px;}
  .step .n {width:24px; height:24px; border-radius:8px; display:grid; place-items:center;
            font-size:.76rem; font-weight:700; background:rgba(52,152,219,.22);
            border:1px solid rgba(52,152,219,.38); color:#bfe0ff; flex:0 0 auto;}
  .step .t {font-weight:650; font-size:.95rem;}
  .step .note {opacity:.62; font-size:.82rem; margin-top:7px;}

  /* ── widgets ───────────────────────────────────────── */
  [data-baseweb="tab-list"] {gap:6px; background:rgba(255,255,255,.04); padding:6px; border-radius:14px;
                             border:1px solid rgba(255,255,255,.07); flex-wrap:wrap;}
  [data-baseweb="tab"] {border-radius:10px; padding:7px 14px; height:auto;}
  [data-baseweb="tab"][aria-selected="true"] {background:rgba(255,255,255,.10);}
  [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display:none;}
  .stButton>button {border-radius:10px; border:1px solid rgba(255,255,255,.14);
                    background:rgba(255,255,255,.05); transition:all .16s ease; font-weight:550;}
  .stButton>button:hover {border-color:rgba(46,204,113,.55); background:rgba(46,204,113,.13);
                          transform:translateY(-1px); color:#fff;}
  .stTextArea textarea, .stTextInput input {border-radius:11px !important;}
  [data-testid="stSidebar"] {background:linear-gradient(180deg,#0E141D,#0A0E15);
                             border-right:1px solid rgba(255,255,255,.06);}
  [data-testid="stSidebar"] .kpi {background:rgba(255,255,255,.04);}
  div[data-testid="stMetricValue"] {font-size:1.45rem; font-weight:700;}
  [data-testid="stDataFrame"] {border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,.07);}
  ::-webkit-scrollbar {width:9px; height:9px;}
  ::-webkit-scrollbar-thumb {background:rgba(255,255,255,.16); border-radius:9px;}
  ::-webkit-scrollbar-track {background:transparent;}
</style>
""", unsafe_allow_html=True)


# ───────────────────────── Small UI helpers ─────────────────────────
def kpi(value, label, accent=BLUE):
    """Compact KPI card (replaces st.metric for a denser, branded look)."""
    return (f'<div class="kpi" style="--c:{accent}">'
            f'<div class="v" style="color:{accent}">{value}</div>'
            f'<div class="l">{label}</div><div class="bar"></div></div>')


def section(title, subtitle=""):
    sub = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="sect"><h3>{title}</h3>{sub}</div>', unsafe_allow_html=True)


def pill(text, sub="", color=GREEN, emoji=""):
    icon = f"{emoji} " if emoji else ""
    extra = f" <small>{sub}</small>" if sub else ""
    return (f'<div class="pill-wrap"><span class="pill" style="border-color:{color}66;'
            f'background:{color}1f">{icon}<span style="color:{color}">{text}</span>{extra}</span></div>')


def style_fig(fig, height=340, title=None, **kwargs):
    """One shared look for every Plotly figure: transparent, dark, clean axes."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK),
        margin=dict(l=10, r=14, t=54, b=12),
        title=dict(text=title if title is not None else (fig.layout.title.text or ""),
                   x=0, xanchor="left", font=dict(size=15.5, color=INK)),
        hoverlabel=dict(bgcolor="#111826", bordercolor="rgba(255,255,255,.18)",
                        font=dict(color=INK, family=FONT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.10)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.10)")
    if kwargs:
        fig.update_layout(**kwargs)
    return fig


# ───────────────────────── The gauge (fix) ─────────────────────────
# The old version used go.Indicator("gauge+number"), which auto-sizes the centre
# number to ~20% of the figure height and cannot be repositioned. In a short,
# half-width column that made the value grow into the coloured band and clip at
# the plot edge. This version draws the semicircle from explicit geometry on a
# square-scaled axis pair, so the text and the arc can never collide.
GAUGE_R_OUT, GAUGE_R_IN = 0.470, 0.315     # coloured steps band
GAUGE_P_OUT, GAUGE_P_IN = 0.285, 0.238     # inner progress ring


def _wedge(a0, a1, r_out, r_in, n=90, cx=0.5, cy=0.0):
    """Annular wedge polygon between two angles (degrees, 180° = left, 0° = right)."""
    a = np.radians(np.linspace(a0, a1, n))
    xo, yo = cx + r_out * np.cos(a), cy + r_out * np.sin(a)
    xi, yi = cx + r_in * np.cos(a[::-1]), cy + r_in * np.sin(a[::-1])
    return (np.concatenate([xo, xi, xo[:1]]), np.concatenate([yo, yi, yo[:1]]))


def half_gauge(value, lo, hi, steps, value_text, accent=GREEN, ticks=(), height=290):
    """180° gauge: coloured steps band + inner progress ring + centred value.

    steps     : [(from, to, colour), ...] painted around the arc
    ticks     : [(value, label), ...] placed just outside the band
    value_text: pre-formatted string, e.g. "76%" / "+0.16"
    """
    def angle(v):
        frac = 0.0 if hi == lo else (float(v) - lo) / (hi - lo)
        return 180.0 * (1.0 - min(max(frac, 0.0), 1.0))

    fig = go.Figure()

    for a, b, color in steps:
        x, y = _wedge(angle(a), angle(b), GAUGE_R_OUT, GAUGE_R_IN)
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", mode="lines",
                                 line=dict(width=0, color="rgba(0,0,0,0)"), fillcolor=color,
                                 hoverinfo="skip", showlegend=False))

    v_ang = angle(value)
    if abs(v_ang - 180.0) > 0.4:                      # progress ring (never degenerate)
        x, y = _wedge(180.0, v_ang, GAUGE_P_OUT, GAUGE_P_IN)
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", mode="lines",
                                 line=dict(width=0, color="rgba(0,0,0,0)"), fillcolor=accent,
                                 hoverinfo="skip", showlegend=False))

    r_mid = (GAUGE_R_OUT + GAUGE_R_IN) / 2.0          # value marker on the band
    a = np.radians(v_ang)
    fig.add_trace(go.Scatter(x=[0.5 + r_mid * np.cos(a)], y=[r_mid * np.sin(a)], mode="markers",
                             marker=dict(size=14, color="#0B0F16", line=dict(width=3, color=accent)),
                             hoverinfo="skip", showlegend=False))

    for tv, tl in ticks:
        ta = np.radians(angle(tv))
        ca = np.cos(ta)
        xanchor = "left" if ca > .25 else ("right" if ca < -.25 else "center")
        fig.add_annotation(x=0.5 + (GAUGE_R_OUT + 0.055) * ca,
                           y=(GAUGE_R_OUT + 0.055) * np.sin(ta), text=tl, showarrow=False,
                           xanchor=xanchor, yanchor="middle",
                           font=dict(size=11.5, color=MUTED, family=FONT))

    fig.add_annotation(x=0.5, y=0.055, text=value_text, showarrow=False,
                       xanchor="center", yanchor="bottom",
                       font=dict(size=42 if len(value_text) <= 4 else 37, color=INK, family=FONT))

    fig.update_layout(
        height=height, showlegend=False, hovermode=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=6, r=6, t=10, b=6),
        font=dict(family=FONT, color=MUTED),
        xaxis=dict(range=[-0.10, 1.10], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.075, 0.565], visible=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
    )
    return fig


# ───────────────────────── Loading (cached) ─────────────────────────
@st.cache_resource(show_spinner="First run: training the models (about 1 minute)…")
def get_artifacts():
    if not train.models_exist():
        train.train_all()
    return train.load_all()


@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()


A = get_artifacts()
vader = get_vader()
tfidf, clf, w2v, lda = A["tfidf"], A["clf"], A["w2v"], A["lda"]
texts, labels, stats, results = A["texts"], A["labels"], A["stats"], A["results"]
topic_words = A["topic_words"]
TOPIC_COLORS = [RED, BLUE, AMBER, PURPLE, TEAL]

SAMPLES = {
    "😊 Positive": "Tom Hanks is brilliant and the story is wonderful. A moving, beautifully acted film that I loved from start to finish.",
    "😞 Negative": "A boring, predictable mess. The acting was terrible, the plot made no sense and I regretted watching it.",
    "🤔 Mixed": "Tom Hanks is brilliant and the story is wonderful, but the ending was not good.",
    "🙃 Sarcastic": "Great... if you enjoy wasting two hours of your life. What a masterpiece of boredom.",
}
st.session_state.setdefault("review_text", SAMPLES["🤔 Mixed"])
st.session_state.setdefault("analyzed", SAMPLES["🤔 Mixed"])


def use_sample(text):
    st.session_state.review_text = text
    st.session_state.analyzed = text


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Review Intelligence")
    st.caption("One dataset → ten NLP concepts → one dashboard")
    st.divider()
    st.markdown(f'<div class="kpi" style="--c:{GREEN}"><div class="v" style="color:{GREEN}">'
                f'{stats["n_reviews"]:,}</div><div class="l">Reviews</div><div class="bar"></div></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="kpi" style="--c:{BLUE}; margin-top:10px"><div class="v" style="color:{BLUE}">'
                f'{stats["n_pos"]:,} / {stats["n_neg"]:,}</div><div class="l">Positive / negative</div>'
                f'<div class="bar"></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Main model**  \nTF-IDF (1–2 grams) + Logistic Regression")
    st.metric("Test accuracy (80/20 split)", f"{results['LogReg + TF-IDF']:.1%}")
    st.divider()
    st.caption(f"Models trained: {A['trained_at']}")
    st.caption("Built with scikit-learn · NLTK · Streamlit")

# ───────────────────────── Header ─────────────────────────
st.markdown(
    '<div class="hero"><h1>🎬 Movie Review Intelligence</h1>'
    '<p>Type a movie review and watch ten classic NLP techniques explain it.</p>'
    '<div class="badges"><span class="badge">2,000 reviews</span>'
    '<span class="badge">TF-IDF + LogReg</span><span class="badge">VADER</span>'
    '<span class="badge">Word embeddings</span><span class="badge">LDA topics</span>'
    '<span class="badge">Cosine similarity</span></div></div>',
    unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "😊 Sentiment Lab", "🧹 Preprocessing", "📊 Model Performance",
                "🧠 Word Explorer", "🗂️ Topics", "🔎 Similar Reviews"])

# ───────────────────────── 1. Overview ─────────────────────────
with tabs[0]:
    c = st.columns(4)
    c[0].markdown(kpi(f"{stats['n_reviews']:,}", "Reviews", BLUE), unsafe_allow_html=True)
    c[1].markdown(kpi(f"{stats['vocab_size']:,}", "Vocabulary (words)", PURPLE), unsafe_allow_html=True)
    c[2].markdown(kpi(f"{results['LogReg + TF-IDF']:.1%}", "ML accuracy", GREEN), unsafe_allow_html=True)
    c[3].markdown(kpi(f"{results['VADER (no training)']:.1%}", "VADER accuracy", AMBER), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("The 10 concepts in this project", "Each tab below is one classic NLP technique.")
    concepts = [
        ("1 · Text exploration", "Zipf's law, top words & bigrams"),
        ("2 · Preprocessing", "Stopwords, stemming vs lemmatization"),
        ("3 · POS tagging & NER", "Grammar roles and named entities"),
        ("4 · Feature extraction", "BoW, TF-IDF, n-grams"),
        ("5 · ML classification", "Naive Bayes & Logistic Regression"),
        ("6 · Lexicon sentiment", "VADER, no training needed"),
        ("7 · Word embeddings", "Nearest words by meaning"),
        ("8 · Topic modeling", "LDA discovers themes"),
        ("9 · Document similarity", "Cosine similarity search"),
        ("10 · Interactive dashboard", "You are here 👋"),
    ]
    cols = st.columns(5)
    for i, (t, d) in enumerate(concepts):
        num, _, head = t.partition(" · ")
        cols[i % 5].markdown(
            f'<div class="concept"><span class="num">{num}</span><b>{head}</b><span>{d}</span></div>',
            unsafe_allow_html=True)

    section("Exploring the text", "How the raw corpus looks before any model touches it.")
    a, b = st.columns(2)
    with a:
        fig = px.line(x=list(range(1, len(stats["zipf"]) + 1)), y=stats["zipf"],
                      labels={"x": "Word rank", "y": "Frequency"})
        fig.update_traces(line_color=AMBER, line_width=2.4, fill="tozeroy",
                          fillcolor="rgba(243,156,18,.14)")
        st.plotly_chart(style_fig(fig, 330, "Zipf's law: a few words dominate"),
                        width="stretch", config=PLOT_CONFIG)
    with b:
        df = pd.DataFrame(stats["top_words"], columns=["word", "count"]).iloc[::-1]
        fig = px.bar(df, x="count", y="word", orientation="h")
        fig.update_traces(marker_color=BLUE, marker_line_width=0)
        fig.update_layout(xaxis_title="count", yaxis_title=None)
        st.plotly_chart(style_fig(fig, 330, "Top words (after stopword removal)"),
                        width="stretch", config=PLOT_CONFIG)
    a, b = st.columns(2)
    with a:
        df = pd.DataFrame(stats["top_bigrams"], columns=["bigram", "count"]).iloc[::-1]
        fig = px.bar(df, x="count", y="bigram", orientation="h")
        fig.update_traces(marker_color=GREEN, marker_line_width=0)
        fig.update_layout(xaxis_title="count", yaxis_title=None)
        st.plotly_chart(style_fig(fig, 360, "Top bigrams"), width="stretch", config=PLOT_CONFIG)
    with b:
        st.markdown('<div class="step"><div class="hd"><span class="n">POS</span>'
                    '<span class="t">Top adjectives by class</span></div>'
                    '<div style="opacity:.62;font-size:.84rem;margin-bottom:8px">Adjectives (tagged JJ) '
                    'split by the review\'s true label — the emotional vocabulary differs sharply.</div>',
                    unsafe_allow_html=True)
        st.markdown("**In negative reviews**")
        st.markdown("".join(chip(f"{w} · {n}", RED) for w, n in A["adjectives"]["neg"]), unsafe_allow_html=True)
        st.markdown("**In positive reviews**")
        st.markdown("".join(chip(f"{w} · {n}", GREEN) for w, n in A["adjectives"]["pos"]) + "</div>",
                    unsafe_allow_html=True)

# ───────────────────────── 2. Sentiment Lab ─────────────────────────
with tabs[1]:
    section("Sentiment Lab", "Two independent readers score your review: a trained ML model and a lexicon.")
    st.markdown("**Try an example or write your own review:**")
    ex = st.columns(len(SAMPLES))
    for col, (name, text) in zip(ex, SAMPLES.items()):
        col.button(name, on_click=use_sample, args=(text,), width="stretch", key=f"ex_{name}")

    with st.form("sentiment_form"):
        st.text_area("Your review", key="review_text", height=110, label_visibility="collapsed")
        submitted = st.form_submit_button("🔍 Analyze", type="primary")
    if submitted:
        st.session_state.analyzed = st.session_state.review_text

    review = st.session_state.analyzed.strip()
    if not review:
        st.info("Write a review above and press **Analyze**.")
    else:
        p = float(clf.predict_proba(tfidf.transform([preprocess(review)]))[0][1])
        v = vader.polarity_scores(review)["compound"]
        ml_pos, v_pos = p > .5, v > .05
        ml_label = "POSITIVE 😊" if ml_pos else "NEGATIVE 😞"
        v_label = "POSITIVE 😊" if v_pos else ("NEGATIVE 😞" if v < -.05 else "NEUTRAL 😐")
        ml_color = GREEN if ml_pos else RED
        v_color = GREEN if v_pos else (RED if v < -.05 else AMBER)

        st.markdown('<div class="sect" style="margin-top:14px"><h3>The two verdicts</h3>'
                    '<p>Left: a trained classifier. Right: a rule-based lexicon. '
                    'When they disagree, read the shaded words below.</p></div>', unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        ml_steps = [(0, 50, "rgba(231,76,60,.55)"), (50, 100, "rgba(46,204,113,.55)")]
        v_steps = [(-1, -.05, "rgba(231,76,60,.55)"), (-.05, .05, "rgba(148,160,180,.45)"),
                   (.05, 1, "rgba(46,204,113,.55)")]
        with g1:
            st.markdown('<div class="gauge-title">ML model · P(positive)</div>', unsafe_allow_html=True)
            st.plotly_chart(
                half_gauge(p * 100, 0, 100, ml_steps, f"{p * 100:.0f}%", accent=ml_color,
                           ticks=[(0, "0"), (20, "20"), (40, "40"), (60, "60"), (80, "80"), (100, "100")]),
                width="stretch", config=PLOT_CONFIG)
            st.markdown(pill(ml_label, f"{max(p, 1 - p):.0%} confident", ml_color), unsafe_allow_html=True)
        with g2:
            st.markdown('<div class="gauge-title">VADER · compound score</div>', unsafe_allow_html=True)
            st.plotly_chart(
                half_gauge(v, -1, 1, v_steps, f"{v:+.2f}", accent=v_color,
                           ticks=[(-1, "-1"), (-.5, "-0.5"), (0, "0"), (.5, "0.5"), (1, "1")]),
                width="stretch", config=PLOT_CONFIG)
            st.markdown(pill(v_label, f"compound {v:+.2f}", v_color), unsafe_allow_html=True)

        if ml_pos != v_pos and abs(v) > .05:
            st.markdown('<div class="banner warn"><b>The two models disagree.</b> Compare the highlighted '
                        'words below to see why. Sarcasm and mixed reviews often fool both approaches.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner ok"><b>✅ Both models agree.</b> They can still both be wrong.'
                        '</div>', unsafe_allow_html=True)

        section("🔍 Why? Shaded words", "Each word is tinted by how much it pushed the ML model.")
        st.markdown(f'<div class="review-box">{highlight_html(review, tfidf, clf)}</div>',
                    unsafe_allow_html=True)
        st.caption("🟩 pushes positive · 🟥 pushes negative · darker = stronger. Hover a word to see its weight.")

        pos, neg = explain(review, tfidf, clf)
        a, b = st.columns(2)
        a.markdown("**Top positive features**")
        a.markdown("".join(chip(f"{w} ({c:+.2f})", GREEN) for w, c in pos) or "_none_", unsafe_allow_html=True)
        b.markdown("**Top negative features**")
        b.markdown("".join(chip(f"{w} ({c:+.2f})", RED) for w, c in neg) or "_none_", unsafe_allow_html=True)
        st.caption("Features include bigrams such as “not good”, because the model uses 1–2 word n-grams.")

        ents = entities_of(review)
        st.markdown("##### 🏷️ Named entities")
        st.markdown("".join(chip(f"{t}: {e}", BLUE) for t, e in ents) or "_No entities found._",
                    unsafe_allow_html=True)

# ───────────────────────── 3. Preprocessing ─────────────────────────
with tabs[2]:
    section("Preprocessing pipeline", "Lowercase → tokenize → drop stopwords → lemmatize.")
    pre_text = st.text_area("Text to process", "The actors were running through studies of better movies, NOT boring ones!",
                            height=80, key="pre_text")
    if not pre_text.strip():
        st.info("Type some text to see the pipeline.")
    else:
        toks = tokenizer.tokenize(pre_text.lower())
        kept = [t for t in toks if t not in stop_words and len(t) > 2]
        removed = [t for t in toks if t not in kept]
        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<div class="step"><div class="hd"><span class="n">1</span>'
                        '<span class="t">Tokens</span></div>'
                        + "".join(chip(t, "#7f8c8d") for t in toks) + "</div>", unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="step"><div class="hd"><span class="n">2</span>'
                        '<span class="t">After stopword removal</span></div>'
                        + "".join(chip(t, PURPLE) for t in kept)
                        + '<div class="note">removed: ' + (", ".join(removed) or "nothing") + "</div></div>",
                        unsafe_allow_html=True)
        st.markdown('<div class="step"><div class="hd"><span class="n">3</span>'
                    '<span class="t">Stemming vs lemmatization</span></div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"word": kept, "stem (Porter)": [stemmer.stem(t) for t in kept],
                                   "lemma (WordNet)": [lemmatizer.lemmatize(t) for t in kept]}),
                     width="stretch", hide_index=True)
        st.caption("Stemming chops endings by rule (studies → studi). Lemmatization uses a dictionary (studies → study).")
        tag_colors = {"NN": BLUE, "JJ": GREEN, "VB": AMBER, "RB": PURPLE, "NNP": PINK}
        tags = pos_tags(pre_text)
        st.markdown('<div class="step"><div class="hd"><span class="n">4</span>'
                    '<span class="t">Part-of-speech tags</span></div>' + "".join(
                        chip(f"{w}/{t}", tag_colors.get(t[:2] if t[:2] in tag_colors else t, "#7f8c8d"))
                        for w, t in tags) + '<div class="note">NN noun · NNP proper noun · JJ adjective · '
                    'VB verb · RB adverb</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="step"><div class="hd"><span class="n">5</span>'
                    '<span class="t">Final cleaned text (what the model actually sees)</span></div><code>'
                    + (preprocess(pre_text) or "(empty)") + "</code></div>", unsafe_allow_html=True)
        st.info("💡 “not” is deliberately **kept**. Removing it would turn “not good” into “good”.")

# ───────────────────────── 4. Model Performance ─────────────────────────
with tabs[3]:
    section("Model performance", "Five approaches compared on the same 400 held-out reviews.")
    acc = pd.DataFrame({"setup": list(results), "accuracy": list(results.values())}).sort_values("accuracy")
    fig = px.bar(acc, x="accuracy", y="setup", orientation="h", text=acc["accuracy"].map("{:.1%}".format),
                 color="accuracy", color_continuous_scale="Tealgrn")
    fig.update_traces(textposition="outside", textfont=dict(color=INK, size=12), marker_line_width=0)
    fig.update_layout(xaxis_range=[0.5, 0.98], coloraxis_showscale=False, yaxis_title=None,
                      xaxis_title="accuracy", xaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 330, "Accuracy by approach"), width="stretch", config=PLOT_CONFIG)

    a, b = st.columns(2)
    with a:
        cm = A["cm"]
        fig = px.imshow(cm, text_auto=True, x=["Neg", "Pos"], y=["Neg", "Pos"],
                        color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
        fig.update_traces(textfont=dict(size=17, color=INK))
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Actual", xaxis_title="Predicted")
        st.plotly_chart(style_fig(fig, 380, "Confusion matrix (LogReg + TF-IDF)"),
                        width="stretch", config=PLOT_CONFIG)
    with b:
        st.markdown('<div class="sect" style="margin-top:2px"><h3>Classification report</h3>'
                    '<p>Per-class precision, recall and F1 on the held-out set.</p></div>',
                    unsafe_allow_html=True)
        rep = pd.DataFrame(A["report"]).T.drop(index=["accuracy", "macro avg"], errors="ignore")
        rep["support"] = rep["support"].astype(int)
        st.dataframe(rep.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}),
                     width="stretch")
        st.caption("**Precision**: of reviews called positive, how many were? **Recall**: of truly positive "
                   "reviews, how many were found? **F1**: the balance of both.")

    feats = np.array(tfidf.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    top = pd.DataFrame({"feature": np.r_[feats[order[:12]], feats[order[-12:]]],
                        "weight": np.r_[clf.coef_[0][order[:12]], clf.coef_[0][order[-12:]]]})
    fig = px.bar(top, x="weight", y="feature", orientation="h", title="Most influential features (model weights)",
                 color=top["weight"] > 0, color_discrete_map={True: GREEN, False: RED})
    fig.update_traces(marker_line_width=0)
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="coefficient", height=560)
    st.plotly_chart(style_fig(fig, 560, "Most influential features (model weights)"),
                    width="stretch", config=PLOT_CONFIG)
    st.markdown('<div class="banner ok"><b>Takeaway:</b> ML learns domain words (“mess”, “dull”, “wonderful”). '
                'VADER needs no training but is weaker on film reviews. Transformers (BERT / DistilBERT) are '
                'the natural next step.</div>', unsafe_allow_html=True)

# ───────────────────────── 5. Word Explorer ─────────────────────────
with tabs[4]:
    section("Word Explorer", "Word embeddings place words used in similar contexts close together.")
    c1, c2, c3 = st.columns([2, 2, 1])
    w1 = c1.text_input("Word", "boring").strip().lower()
    w2 = c2.text_input("Compare to (optional)", "dull").strip().lower()
    topn = c3.slider("Neighbors", 3, 12, 8)
    if not w1:
        st.info("Enter a word.")
    elif w1 not in w2v.wv:
        st.warning(f"'{w1}' is not in the vocabulary (words need to appear at least 5 times). Try another word.")
    else:
        sims = w2v.wv.most_similar(w1, topn=topn)
        a, b = st.columns([1, 1])
        with a:
            df = pd.DataFrame(sims, columns=["word", "similarity"]).iloc[::-1]
            fig = px.bar(df, x="similarity", y="word", orientation="h")
            fig.update_traces(marker_color=BLUE, marker_line_width=0)
            fig.update_layout(xaxis_title="cosine similarity", yaxis_title=None)
            st.plotly_chart(style_fig(fig, 360, f"Closest to “{w1}”"), width="stretch", config=PLOT_CONFIG)
            if w2:
                if w2 in w2v.wv:
                    st.metric(f"“{w1}” ↔ “{w2}”", f"{w2v.wv.similarity(w1, w2):.2f}",
                              help="1 = identical, 0 = unrelated")
                else:
                    st.warning(f"'{w2}' is not in the vocabulary.")
        with b:
            base = ["good", "great", "excellent", "wonderful", "bad", "awful", "terrible", "boring",
                    "actor", "actress", "director", "film", "movie", "story", "plot"]
            group = {**{w: "positive" for w in base[:4]}, **{w: "negative" for w in base[4:8]},
                     **{w: "film words" for w in base[8:]}}
            group.update({w: "neighbors" for w, _ in sims})
            group[w1] = "your word"
            if w2 in w2v.wv:
                group[w2] = "compare"
            words = [w for w in group if w in w2v.wv]
            pts = PCA(n_components=2).fit_transform([w2v.wv[w] for w in words])
            dfp = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "word": words, "group": [group[w] for w in words]})
            fig = px.scatter(dfp, x="x", y="y", text="word", color="group",
                             color_discrete_map={"positive": GREEN, "negative": RED, "film words": BLUE,
                                                 "neighbors": AMBER, "your word": PINK, "compare": PURPLE})
            fig.update_traces(textposition="top center", marker_size=11,
                              marker_line_width=0.6, marker_line_color="rgba(255,255,255,.35)")
            fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_showticklabels=False,
                              yaxis_showticklabels=False)
            st.plotly_chart(style_fig(fig, 360, "2-D map (PCA of 100 dimensions)"),
                            width="stretch", config=PLOT_CONFIG)

# ───────────────────────── 6. Topics ─────────────────────────
with tabs[5]:
    section("Topics (LDA)", "Themes discovered with no labels — each review is a mixture of topics.")
    for i, col in enumerate(st.columns(5)):
        with col:
            st.markdown(f'<div class="step" style="border-color:{TOPIC_COLORS[i]}44">'
                        f'<div class="hd"><span class="n" style="background:{TOPIC_COLORS[i]}33;'
                        f'border-color:{TOPIC_COLORS[i]}66;color:#fff">{i + 1}</span>'
                        f'<span class="t">Topic {i + 1}</span></div>'
                        + "".join(chip(w, TOPIC_COLORS[i]) for w in topic_words[i][:8]) + "</div>",
                        unsafe_allow_html=True)
    st.caption("Topics need human interpretation. Can you name each one?")
    topic_text = st.text_area("Which topics does *your* text belong to?",
                              "Aliens attack earth in this action movie with amazing special effects.", height=80)
    if topic_text.strip():
        dist = lda.transform(A["topic_vec"].transform([preprocess(topic_text)]))[0]
        df = pd.DataFrame({"topic": [f"Topic {i + 1}: {', '.join(topic_words[i][:3])}" for i in range(5)],
                           "share": dist})
        fig = px.bar(df, x="share", y="topic", orientation="h", color="topic",
                     color_discrete_sequence=TOPIC_COLORS)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", yaxis_title=None, xaxis_range=[0, 1])
        st.plotly_chart(style_fig(fig, 330, "Topic mix of your text"), width="stretch", config=PLOT_CONFIG)
    else:
        st.info("Paste some text to see its topic mix.")

# ───────────────────────── 7. Similar Reviews ─────────────────────────
with tabs[6]:
    section("Similar reviews", "Search all 2,000 reviews by describing what you want (cosine similarity on TF-IDF).")
    c1, c2 = st.columns([4, 1])
    q = c1.text_input("Describe a movie", "alien invasion with special effects and a big space battle")
    k = c2.slider("Results", 1, 8, 4)
    if q.strip():
        qv = A["sim_vec"].transform([preprocess(q)])
        if qv.nnz == 0:
            st.warning("None of those words are in the vocabulary. Try different words.")
        else:
            sc = cosine_similarity(qv, A["all_tfidf"]).ravel()
            for rank, b in enumerate(sc.argsort()[-k:][::-1], 1):
                tag = chip("positive", GREEN) if labels[b] else chip("negative", RED)
                snippet = " ".join(texts[b].split())[:320]
                st.markdown(f'<div class="card"><h4>{tag}'
                            f'<span class="match-score">#{rank} · match {sc[b]:.2f}</span></h4>'
                            f'<small style="opacity:.8">{snippet}…</small></div>', unsafe_allow_html=True)
    else:
        st.info("Describe a movie to search.")

st.divider()
st.caption("Movie Review Intelligence · scikit-learn · NLTK · Streamlit · Sarcasm still fools every model here.")
