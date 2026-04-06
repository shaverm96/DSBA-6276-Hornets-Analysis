import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utils import (
    build_reviews_frame,
    discover_repo_assets,
    format_float,
    format_int,
    generate_gemini_insight,
    get_api_key_from_sources,
)

st.set_page_config(page_title="Reviews Text Analysis", page_icon="💬", layout="wide")

assets = discover_repo_assets()
out = build_reviews_frame(assets)
reviews = out["reviews"]
theme_summary = out["theme_summary"]
recommendations = out["recommendations"]
exec_summary = out["exec_summary"]
rep_pos = out["rep_pos"]
rep_neg = out["rep_neg"]
topic_samples = out["topic_samples"]
mismatch = out["mismatch"]

st.title("Google Reviews Text Analysis")
st.caption("Notebook-aligned customer sentiment and theme intelligence for operational decision support.")

with st.sidebar:
    st.header("Page Controls")

    sentiment_options = ["positive", "neutral", "negative"]
    selected_sentiment = st.multiselect("Sentiment Labels", options=sentiment_options, default=sentiment_options)

# API key is sourced from st.secrets or environment (no sidebar input).
api_key = get_api_key_from_sources("")

view = reviews.copy()
if not view.empty:
    if selected_sentiment:
        view = view[view["sentiment_label"].isin(selected_sentiment)]

st.markdown("### Review KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Reviews", format_int(float(len(view)) if not view.empty else np.nan))
k2.metric("Average Rating", format_float(float(view["rating"].mean()) if not view.empty else np.nan, 2))

if not view.empty and "sentiment_label" in view.columns:
    pos_pct = 100 * (view["sentiment_label"].eq("positive").mean())
    neg_pct = 100 * (view["sentiment_label"].eq("negative").mean())
else:
    pos_pct, neg_pct = np.nan, np.nan

k3.metric("Positive Sentiment %", format_float(pos_pct, 1))
k4.metric("Negative Sentiment %", format_float(neg_pct, 1))

top_complaint = "N/A"
if not theme_summary.empty and "low_minus_high_gap" in theme_summary.columns:
    ts = theme_summary.sort_values("low_minus_high_gap", ascending=False)
    if len(ts) > 0 and "theme" in ts.columns:
        top_complaint = str(ts.iloc[0]["theme"])
k5.metric("Top Complaint Theme", top_complaint)

st.markdown("### Core Review Visuals")

c1, c2 = st.columns(2)
with c1:
    if not view.empty and "rating" in view.columns:
        fig = px.histogram(view, x="rating", nbins=5, title="Rating Distribution")
        fig.update_layout(xaxis_title="Star Rating", yaxis_title="Number of Reviews")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Rating distribution unavailable.")

with c2:
    if not view.empty and "sentiment_label" in view.columns:
        sent = view["sentiment_label"].value_counts().reset_index()
        sent.columns = ["sentiment_label", "count"]
        sent["sentiment_label"] = sent["sentiment_label"].astype(str).str.title()
        fig = px.pie(sent, values="count", names="sentiment_label", title="Sentiment Distribution")
        fig.update_layout(legend_title_text="Sentiment Label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sentiment distribution unavailable.")

c3, c4 = st.columns(2)
with c3:
    if not view.empty and {"rating", "sentiment_score"}.issubset(set(view.columns)):
        fig = px.box(view, x="rating", y="sentiment_score", title="Sentiment Score by Star Rating")
        fig.update_layout(xaxis_title="Star Rating", yaxis_title="Sentiment Score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sentiment vs rating chart unavailable.")

with c4:
    if not view.empty and "word_count" in view.columns:
        fig = px.box(view, x="rating", y="word_count", title="Review Length by Rating")
        fig.update_layout(xaxis_title="Star Rating", yaxis_title="Review Length (Words)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Review length chart unavailable.")

st.markdown("### Top Terms and N-grams")


def top_terms(df: pd.DataFrame, ngram=(1, 1), top_n=20) -> pd.DataFrame:
    if df.empty or "clean_text" not in df.columns:
        return pd.DataFrame(columns=["term", "count"])
    vec = CountVectorizer(stop_words="english", ngram_range=ngram, min_df=1, max_features=500)
    X = vec.fit_transform(df["clean_text"].fillna(""))
    counts = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    out_df = pd.DataFrame({"term": terms, "count": counts}).sort_values("count", ascending=False).head(top_n)
    return out_df

pos = view[view["rating"] >= 4] if (not view.empty and "rating" in view.columns) else pd.DataFrame()
neg = view[view["rating"] <= 2] if (not view.empty and "rating" in view.columns) else pd.DataFrame()

terms_all = top_terms(view, (1, 1), 20)
bigrams_pos = top_terms(pos, (2, 2), 12)
bigrams_neg = top_terms(neg, (2, 2), 12)

cc1, cc2 = st.columns(2)
with cc1:
    if not terms_all.empty:
        top_terms_plot = terms_all.head(15).copy()
        term_order = top_terms_plot["term"].tolist()
        fig = px.bar(
            top_terms_plot,
            x="count",
            y="term",
            orientation="h",
            title="Top Terms (Filtered View)",
            category_orders={"term": term_order},
        )
        fig.update_layout(xaxis_title="Mention Count", yaxis_title="Term")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Top terms unavailable.")

with cc2:
    compare = pd.concat([
        top_terms(pos, (1, 1), 12).assign(group="Positive"),
        top_terms(neg, (1, 1), 12).assign(group="Negative"),
    ])
    if not compare.empty:
        term_order = (
            compare.groupby("term", as_index=False)["count"]
            .max()
            .sort_values("count", ascending=False)["term"]
            .tolist()
        )
        fig = px.bar(
            compare,
            x="count",
            y="term",
            color="group",
            barmode="group",
            orientation="h",
            title="Positive vs Negative Terms",
            category_orders={"term": term_order},
        )
        fig.update_layout(
            xaxis_title="Mention Count",
            yaxis_title="Term",
            legend_title_text="Review Segment",
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Comparison terms unavailable.")

st.markdown("### Word Cloud")
if not view.empty and "clean_text" in view.columns:
    text_blob = " ".join(view["clean_text"].fillna(""))
    if text_blob.strip():
        wc = WordCloud(width=1200, height=500, background_color="white", colormap="plasma").generate(text_blob)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud (Filtered Reviews)")
        st.pyplot(fig)
    else:
        st.info("Insufficient text for word cloud.")

st.markdown("### Theme and Recommendation Outputs (Notebook Artifacts)")

r1, r2 = st.columns(2)
with r1:
    if not theme_summary.empty:
        st.dataframe(theme_summary.sort_values("overall_mention_rate", ascending=False), use_container_width=True)
    else:
        st.info("Theme summary artifact not found.")

with r2:
    if not recommendations.empty:
        st.dataframe(recommendations, use_container_width=True)
    else:
        st.info("Recommendations artifact not found.")

st.markdown("### Representative Reviews")

rr1, rr2 = st.columns(2)
with rr1:
    st.subheader("Most Positive Examples")
    if not rep_pos.empty:
        st.dataframe(rep_pos.head(10), use_container_width=True, height=320)
    else:
        fallback = view.sort_values("sentiment_score", ascending=False).head(10)[["rating", "sentiment_score", "review_text_raw"]] if not view.empty else pd.DataFrame()
        st.dataframe(fallback, use_container_width=True, height=320)

with rr2:
    st.subheader("Most Negative Examples")
    if not rep_neg.empty:
        st.dataframe(rep_neg.head(10), use_container_width=True, height=320)
    else:
        fallback = view.sort_values("sentiment_score", ascending=True).head(10)[["rating", "sentiment_score", "review_text_raw"]] if not view.empty else pd.DataFrame()
        st.dataframe(fallback, use_container_width=True, height=320)

if not mismatch.empty:
    with st.expander("Sentiment-Rating Mismatch Samples"):
        st.dataframe(mismatch.head(30), use_container_width=True)

if not topic_samples.empty:
    with st.expander("Topic Sample Reviews"):
        st.dataframe(topic_samples, use_container_width=True)

if not exec_summary.empty:
    st.markdown("### Executive Summary Table")
    st.dataframe(exec_summary, use_container_width=True)

st.markdown("### LLM Review Insights (Gemini 2.5 Flash)")
if st.button("Generate Review Executive Summary", type="primary"):
    theme_records = theme_summary.to_dict(orient="records")[:8] if not theme_summary.empty else []
    recommendation_records = recommendations.to_dict(orient="records")[:8] if not recommendations.empty else []

    payload = {
        "filters": {
            "sentiment": selected_sentiment,
        },
        "kpis": {
            "review_count": len(view),
            "avg_rating": float(view["rating"].mean()) if not view.empty else np.nan,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
        },
        "top_terms": terms_all.head(12).to_dict(orient="records") if not terms_all.empty else [],
        "themes": theme_records,
        "recommendations": recommendation_records,
    }

    prompt = (
        "You are a senior customer-insights advisor preparing executive commentary for a presentation.\n"
        "Use ONLY the provided data context. Do not invent values, themes, or recommendations.\n"
        "Translate NLP outputs into concrete business implications.\n\n"
        "Return EXACTLY these sections:\n"
        "1) Executive Summary (3 bullets):\n"
        "   - Overall customer sentiment and experience story in this filtered view.\n"
        "2) What Customers Love (3 bullets):\n"
        "   - Must reference terms/themes in the provided context.\n"
        "3) What Frustrates Customers (3 bullets):\n"
        "   - Must reference terms/themes in the provided context.\n"
        "4) Quantified Findings (3-5 bullets):\n"
        "   - Each bullet should include at least one metric or frequency from the data context.\n"
        "5) Priority Actions (3 bullets):\n"
        "   - Specific operational actions tied to the findings.\n"
        "6) 60-Second Slide Script (max 100 words):\n"
        "   - A concise script for the presenter.\n\n"
        "Style rules:\n"
        "- Be concise, concrete, and decision-oriented.\n"
        "- If data is missing, say 'not available in current view'.\n"
        "- Avoid generic advice not supported by context.\n\n"
        f"Data context:\n{payload}"
    )

    with st.spinner("Calling Gemini 2.5 Flash..."):
        text = generate_gemini_insight(api_key, prompt)
    st.write(text)

with st.expander("Methodology Notes"):
    st.write(
        "This page reuses notebook-exported review outputs whenever available (theme summary, recommendations, executive summary, "
        "and representative examples). Lightweight preprocessing and sentiment scoring are recalculated only for interactive filtering."
    )
