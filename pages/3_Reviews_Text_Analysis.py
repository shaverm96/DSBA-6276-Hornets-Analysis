import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

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
    sentiment_scope = st.selectbox(
        "Sentiment Labels",
        options=["All Sentiment", "Custom Selection"],
        index=0,
        help="Use All Sentiment for full coverage, or switch to Custom Selection to pick specific labels.",
    )
    if sentiment_scope == "Custom Selection":
        selected_sentiment = st.multiselect(
            "Choose Sentiment Labels",
            options=sentiment_options,
            default=sentiment_options,
        )
    else:
        selected_sentiment = sentiment_options

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
        rating_counts = (
            pd.to_numeric(view["rating"], errors="coerce")
            .dropna()
            .round()
            .astype(int)
            .clip(lower=1, upper=5)
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .rename_axis("rating")
            .reset_index(name="count")
        )
        total_reviews = max(int(rating_counts["count"].sum()), 1)
        rating_counts["pct"] = (100 * rating_counts["count"] / total_reviews).round(1)
        rating_counts["star_label"] = rating_counts["rating"].astype(str) + "★"

        fig = px.bar(
            rating_counts,
            x="star_label",
            y="count",
            text="count",
            title="Rating Distribution",
            color_discrete_sequence=["#7ec0ee"],
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="Rating=%{x}<br>Reviews=%{y:,}<br>Share=%{customdata[0]}%<extra></extra>",
            customdata=rating_counts[["pct"]].to_numpy(),
        )
        fig.update_layout(
            xaxis_title="Star Rating",
            yaxis_title="Number of Reviews",
            showlegend=False,
            bargap=0.2,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Rating distribution unavailable.")

with c2:
    if not view.empty and "sentiment_label" in view.columns:
        sent = view["sentiment_label"].value_counts().reset_index()
        sent.columns = ["sentiment_label", "count"]
        sent["sentiment_label"] = sent["sentiment_label"].astype(str).str.title()
        fig = px.pie(
            sent,
            values="count",
            names="sentiment_label",
            title="Sentiment Distribution",
            color="sentiment_label",
            color_discrete_map={"Positive": "#7ec0ee", "Neutral": "#2d7dd2", "Negative": "#f2a3a3"},
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(legend_title_text="Sentiment Label", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sentiment distribution unavailable.")

c3, c4 = st.columns(2)
with c3:
    if not view.empty and {"rating", "sentiment_score"}.issubset(set(view.columns)):
        score_view = view.copy()
        score_view["rating"] = pd.to_numeric(score_view["rating"], errors="coerce").round().clip(1, 5)
        score_view = score_view.dropna(subset=["rating", "sentiment_score"]).copy()
        score_view["star_label"] = score_view["rating"].astype(int).astype(str) + "★"

        fig = px.box(
            score_view,
            x="star_label",
            y="sentiment_score",
            title="Sentiment Score by Star Rating",
            category_orders={"star_label": ["1★", "2★", "3★", "4★", "5★"]},
        )
        fig.update_traces(boxpoints=False)
        fig.update_layout(
            xaxis_title="Star Rating",
            yaxis_title="Sentiment Score",
            showlegend=False,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sentiment vs rating chart unavailable.")

with c4:
    if not view.empty and "word_count" in view.columns:
        length_view = view.copy()
        length_view["rating"] = pd.to_numeric(length_view["rating"], errors="coerce").round().clip(1, 5)
        length_view = length_view.dropna(subset=["rating", "word_count"]).copy()
        length_view["star_label"] = length_view["rating"].astype(int).astype(str) + "★"

        fig = px.box(
            length_view,
            x="star_label",
            y="word_count",
            title="Review Length by Rating",
            category_orders={"star_label": ["1★", "2★", "3★", "4★", "5★"]},
        )
        fig.update_traces(boxpoints=False)
        fig.update_layout(
            xaxis_title="Star Rating",
            yaxis_title="Review Length (Words)",
            showlegend=False,
        )
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
        top_terms_plot = top_terms_plot.sort_values("count", ascending=False)
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
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Top terms unavailable.")

with cc2:
    pos_terms = top_terms(pos, (1, 1), 20).rename(columns={"count": "positive_count"})
    neg_terms = top_terms(neg, (1, 1), 20).rename(columns={"count": "negative_count"})
    compare_wide = pos_terms.merge(neg_terms, on="term", how="outer").fillna(0)
    compare_wide["positive_count"] = pd.to_numeric(compare_wide["positive_count"], errors="coerce").fillna(0)
    compare_wide["negative_count"] = pd.to_numeric(compare_wide["negative_count"], errors="coerce").fillna(0)
    compare_wide["total_count"] = compare_wide["positive_count"] + compare_wide["negative_count"]
    compare_wide = compare_wide.sort_values("total_count", ascending=False).head(12)

    if not compare_wide.empty:
        term_order = compare_wide["term"].tolist()
        compare = compare_wide[["term", "positive_count", "negative_count"]].melt(
            id_vars="term",
            value_vars=["positive_count", "negative_count"],
            var_name="group",
            value_name="count",
        )
        compare["group"] = compare["group"].replace(
            {"positive_count": "Positive", "negative_count": "Negative"}
        )
        fig = px.bar(
            compare,
            x="count",
            y="term",
            color="group",
            barmode="stack",
            orientation="h",
            title="Positive vs Negative Terms (Stacked)",
            category_orders={"term": term_order},
            color_discrete_map={"Positive": "#8ec9f3", "Negative": "#2d7dd2"},
        )
        fig.update_layout(
            xaxis_title="Mention Count",
            yaxis_title="Term",
            legend_title_text="Review Segment",
        )
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


def pretty_label(text: str) -> str:
    return str(text).replace("_", " ").strip().title()

r1, r2 = st.columns(2)
with r1:
    if not theme_summary.empty:
        theme_tbl = theme_summary.copy()
        theme_tbl.columns = [str(c).strip().lower() for c in theme_tbl.columns]

        if "theme" in theme_tbl.columns:
            theme_tbl["theme"] = theme_tbl["theme"].map(pretty_label)

        for rate_col in ["overall_mention_rate", "low_rating_mention_rate", "high_rating_mention_rate", "low_minus_high_gap"]:
            if rate_col in theme_tbl.columns:
                theme_tbl[rate_col] = pd.to_numeric(theme_tbl[rate_col], errors="coerce")

        if "overall_mention_rate" in theme_tbl.columns:
            theme_tbl = theme_tbl.sort_values("overall_mention_rate", ascending=False)

        for rate_col in ["overall_mention_rate", "low_rating_mention_rate", "high_rating_mention_rate", "low_minus_high_gap"]:
            if rate_col in theme_tbl.columns:
                theme_tbl[rate_col] = theme_tbl[rate_col].map(lambda v: f"{(100 * v):.1f}%" if pd.notna(v) else "N/A")

        theme_tbl = theme_tbl.rename(
            columns={
                "theme": "Theme",
                "overall_mention_rate": "Overall Mention Rate",
                "low_rating_mention_rate": "Low-Rating Mention Rate",
                "high_rating_mention_rate": "High-Rating Mention Rate",
                "low_minus_high_gap": "Low-High Mention Gap",
            }
        )

        st.dataframe(theme_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("Theme summary artifact not found.")

with r2:
    if not recommendations.empty:
        rec_tbl = recommendations.copy()
        rec_tbl.columns = [str(c).strip().lower() for c in rec_tbl.columns]

        if "focus_area" in rec_tbl.columns:
            rec_tbl["focus_area"] = rec_tbl["focus_area"].map(pretty_label)

        if "priority" in rec_tbl.columns:
            priority_order = pd.CategoricalDtype(categories=["High", "Medium", "Low"], ordered=True)
            rec_tbl["priority"] = rec_tbl["priority"].astype(str).str.title().astype(priority_order)
            rec_tbl = rec_tbl.sort_values(["priority", "focus_area"] if "focus_area" in rec_tbl.columns else ["priority"])
            rec_tbl["priority"] = rec_tbl["priority"].astype(str)

        rec_tbl = rec_tbl.rename(
            columns={
                "priority": "Priority",
                "focus_area": "Focus Area",
                "evidence": "Evidence",
                "recommended_action": "Recommended Action",
            }
        )

        st.dataframe(
            rec_tbl,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priority": st.column_config.TextColumn(width="small"),
                "Focus Area": st.column_config.TextColumn(width="medium"),
                "Evidence": st.column_config.TextColumn(width="medium"),
                "Recommended Action": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("Recommendations artifact not found.")

st.markdown("### Representative Reviews")


def format_review_samples_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out_df = df.copy()
    out_df.columns = [str(c).strip().lower() for c in out_df.columns]

    text_col = next((c for c in ["review_text_raw", "review", "clean_text"] if c in out_df.columns), None)
    if text_col is not None:
        out_df[text_col] = (
            out_df[text_col]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.slice(0, 220)
        )

    if "rating" in out_df.columns:
        out_df["rating"] = pd.to_numeric(out_df["rating"], errors="coerce").map(
            lambda v: f"{int(v)}★" if pd.notna(v) else "N/A"
        )

    if "sentiment_score" in out_df.columns:
        out_df["sentiment_score"] = pd.to_numeric(out_df["sentiment_score"], errors="coerce").map(
            lambda v: f"{v:+.3f}" if pd.notna(v) else "N/A"
        )

    keep_cols = [
        c
        for c in ["reviewer", "reviewer_name", "rating", "sentiment_score", text_col]
        if c is not None and c in out_df.columns
    ]
    if keep_cols:
        out_df = out_df[keep_cols].copy()

    out_df = out_df.rename(
        columns={
            "reviewer": "Reviewer",
            "reviewer_name": "Reviewer",
            "rating": "Rating",
            "sentiment_score": "Sentiment Score",
            "review_text_raw": "Review Excerpt",
            "review": "Review Excerpt",
            "clean_text": "Review Excerpt",
        }
    )
    return out_df

rr1, rr2 = st.columns(2)
with rr1:
    st.subheader("Most Positive Examples")
    if not rep_pos.empty:
        pos_tbl = format_review_samples_table(rep_pos.head(10))
        st.dataframe(pos_tbl, use_container_width=True, height=320, hide_index=True)
    else:
        fallback = view.sort_values("sentiment_score", ascending=False).head(10)[["rating", "sentiment_score", "review_text_raw"]] if not view.empty else pd.DataFrame()
        fallback_tbl = format_review_samples_table(fallback)
        st.dataframe(fallback_tbl, use_container_width=True, height=320, hide_index=True)

with rr2:
    st.subheader("Most Negative Examples")
    if not rep_neg.empty:
        neg_tbl = format_review_samples_table(rep_neg.head(10))
        st.dataframe(neg_tbl, use_container_width=True, height=320, hide_index=True)
    else:
        fallback = view.sort_values("sentiment_score", ascending=True).head(10)[["rating", "sentiment_score", "review_text_raw"]] if not view.empty else pd.DataFrame()
        fallback_tbl = format_review_samples_table(fallback)
        st.dataframe(fallback_tbl, use_container_width=True, height=320, hide_index=True)

if not mismatch.empty:
    with st.expander("Sentiment-Rating Mismatch Samples"):
        mismatch_tbl = mismatch.copy()
        mismatch_tbl.columns = [str(c).strip().lower() for c in mismatch_tbl.columns]

        mismatch_text_col = next((c for c in ["review_text_raw", "review", "clean_text"] if c in mismatch_tbl.columns), None)
        mismatch_keep = [
            c
            for c in ["reviewer", "reviewer_name", "rating", "sentiment_score", "date", "review_date", mismatch_text_col]
            if c is not None and c in mismatch_tbl.columns
        ]
        if mismatch_keep:
            mismatch_tbl = mismatch_tbl[mismatch_keep].copy()

        mismatch_tbl = format_review_samples_table(mismatch_tbl)
        if "Date" not in mismatch_tbl.columns:
            if "date" in mismatch.columns:
                mismatch_tbl["Date"] = pd.to_datetime(mismatch["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            elif "review_date" in mismatch.columns:
                mismatch_tbl["Date"] = pd.to_datetime(mismatch["review_date"], errors="coerce").dt.strftime("%Y-%m-%d")

        display_cols = [c for c in ["Reviewer", "Rating", "Sentiment Score", "Date", "Review Excerpt"] if c in mismatch_tbl.columns]
        if display_cols:
            mismatch_tbl = mismatch_tbl[display_cols]

        st.dataframe(mismatch_tbl.head(30), use_container_width=True, hide_index=True)

if not topic_samples.empty:
    with st.expander("Topic Sample Reviews"):
        st.dataframe(topic_samples, use_container_width=True)

if not exec_summary.empty:
    st.markdown("### Executive Summary Table")
    st.dataframe(exec_summary, use_container_width=True)

st.markdown("### LLM Review Insights")


def render_executive_sections(raw_text: str) -> None:
    """Render model output as clean, presentation-friendly sections."""
    if not raw_text or not isinstance(raw_text, str):
        st.write(raw_text)
        return

    text = raw_text.replace("\r\n", "\n").strip()
    text = re.sub(r"([)\]])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])\(", r"\1 (", text)

    section_pattern = re.compile(r"^\s*(\d)\)\s+([^\n]+)", re.MULTILINE)
    matches = list(section_pattern.finditer(text))

    if not matches:
        st.markdown(text)
        return

    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_num = m.group(1)
        sec_title = m.group(2).strip()
        sec_body = text[start:end].strip()
        sections.append((sec_num, sec_title, sec_body))

    st.markdown("#### Executive Briefing")
    for sec_num, sec_title, sec_body in sections:
        with st.container(border=True):
            st.markdown(f"**{sec_num}) {sec_title}**")
            st.markdown(sec_body if sec_body else "No content generated for this section.")


if st.button("Generate Review Executive Summary", type="primary"):
    theme_records = theme_summary.to_dict(orient="records")[:8] if not theme_summary.empty else []
    recommendation_records = recommendations.to_dict(orient="records")[:8] if not recommendations.empty else []

    payload = {
        "filters": {
            "sentiment": "All Sentiment" if sentiment_scope == "All Sentiment" else selected_sentiment,
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
        "Translate NLP outputs into concrete business implications.\n"
        "Pull specific evidence from KPIs, sentiment split, top terms, themes, and recommendations shown on the page.\n\n"
        "Do NOT discuss model quality, NLP accuracy, or technical performance; focus on customer and business impacts.\n\n"
        "Return EXACTLY these sections:\n"
        "1) Executive Summary (3 bullets):\n"
        "   - Overall customer sentiment and experience story in this filtered view.\n"
        "2) What Customers Love (4 bullets):\n"
        "   - Must reference specific positive terms/themes and why they matter operationally.\n"
        "3) What Frustrates Customers (4 bullets):\n"
        "   - Must reference specific negative terms/themes and likely root causes from context.\n"
        "4) Quantified Findings (4-6 bullets):\n"
        "   - Each bullet should include at least one metric, percentage, or mention count from the data context.\n"
        "5) Priority Actions and Owners (3 bullets):\n"
        "   - Concrete actions tied to findings, including who should own each action area (Ops, Service, Marketing, etc.).\n"
        "6) 60-Second Slide Script (max 110 words):\n"
        "   - A concise script for the presenter.\n\n"
        "Style rules:\n"
        "- Be concise, concrete, and decision-oriented.\n"
        "- Prioritize evidence that is visible in this page context.\n"
        "- If data is missing, say 'not available in current view'.\n"
        "- Avoid generic advice not supported by context.\n\n"
        f"Data context:\n{payload}"
    )

    with st.spinner("Calling Gemini..."):
        text = generate_gemini_insight(api_key, prompt)
    render_executive_sections(text)

with st.expander("Methodology Notes"):
    st.write(
        "This page reuses notebook-exported review outputs whenever available (theme summary, recommendations, executive summary, "
        "and representative examples). Lightweight preprocessing and sentiment scoring are recalculated only for interactive filtering."
    )
