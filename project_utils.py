import json
import os
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent


def _existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def discover_repo_assets() -> Dict[str, Optional[Path]]:
    """Discover core datasets, notebooks, and notebook-exported artifacts."""
    assets = {
        "hornets_games": _existing_path([
            ROOT / "data" / "hornets_games_2015-2026.csv",
            ROOT / "hornets_games_2015-2026.csv",
        ]),
        "all_stars": _existing_path([
            ROOT / "data" / "nba_all_stars.csv",
            ROOT / "nba_all_stars.csv",
        ]),
        "weather": _existing_path([
            ROOT / "data" / "weather_data.csv",
            ROOT / "weather_data.csv",
        ]),
        "google_reviews": _existing_path([
            ROOT / "data" / "google_review_data.csv",
            ROOT / "google_review_data.csv",
        ]),
        "notebook_hornets": _existing_path([
            ROOT / "hornets_attendance_xgboost.ipynb",
        ]),
        "notebook_reviews": _existing_path([
            ROOT / "google_reviews_nlp_analysis.ipynb",
        ]),
        "high_risk": _existing_path([
            ROOT / "data" / "Hornets Analysis Output" / "high_risk_game_flags_for_ops.csv",
            ROOT / "high_risk_game_flags_for_ops.csv",
        ]),
        "revenue": _existing_path([
            ROOT / "data" / "Hornets Analysis Output" / "revenue_scenarios_for_presentation.csv",
            ROOT / "revenue_scenarios_for_presentation.csv",
        ]),
        "review_theme_summary": _existing_path([
            ROOT / "data" / "Google Review Output" / "google_reviews_theme_summary.csv",
            ROOT / "google_reviews_theme_summary.csv",
        ]),
        "review_recommendations": _existing_path([
            ROOT / "data" / "Google Review Output" / "google_reviews_recommendations.csv",
            ROOT / "google_reviews_recommendations.csv",
        ]),
        "review_exec_summary": _existing_path([
            ROOT / "data" / "Google Review Output" / "google_reviews_executive_summary.csv",
            ROOT / "google_reviews_executive_summary.csv",
        ]),
        "review_pos": _existing_path([
            ROOT / "data" / "Google Review Output" / "google_reviews_representative_positive.csv",
            ROOT / "google_reviews_representative_positive.csv",
        ]),
        "review_neg": _existing_path([
            ROOT / "data" / "Google Review Output" / "google_reviews_representative_negative.csv",
            ROOT / "google_reviews_representative_negative.csv",
        ]),
        "review_topic_samples": _existing_path([
            ROOT / "data" / "Google Review Output" / "google_reviews_topic_representative_reviews.csv",
            ROOT / "google_reviews_topic_representative_reviews.csv",
        ]),
        "review_mismatch": _existing_path([
            ROOT / "data" / "Google Review Output" / "google_reviews_sentiment_rating_mismatch.csv",
            ROOT / "google_reviews_sentiment_rating_mismatch.csv",
        ]),
    }
    return assets


@st.cache_data(show_spinner=False)
def load_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def _parse_shap_table_from_text(text_data: str) -> pd.DataFrame:
    """Parse SHAP callout table from text/plain notebook output as a fallback path."""
    if not text_data:
        return pd.DataFrame()

    lines = [ln.rstrip("\n") for ln in str(text_data).splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()

    header_idx = None
    for i, ln in enumerate(lines):
        l = ln.lower()
        if "rank" in l and "feature" in l and "mean_abs_shap" in l:
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    block = [lines[header_idx]]
    for ln in lines[header_idx + 1 :]:
        if re.match(r"^\s*\d+\s+\d+\s+", ln):
            block.append(ln)
        else:
            break

    if len(block) <= 1:
        return pd.DataFrame()

    try:
        parsed = pd.read_fwf(StringIO("\n".join(block)))
    except Exception:
        return pd.DataFrame()

    parsed.columns = [str(c).strip().lower() for c in parsed.columns]
    if "mean_abs_shap" not in parsed.columns or "feature" not in parsed.columns:
        return pd.DataFrame()

    parsed["mean_abs_shap"] = pd.to_numeric(
        parsed["mean_abs_shap"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    if "rank" not in parsed.columns:
        parsed = parsed.reset_index(drop=True)
        parsed["rank"] = parsed.index + 1

    return parsed[[c for c in ["rank", "feature", "mean_abs_shap"] if c in parsed.columns]].dropna(subset=["feature", "mean_abs_shap"])


def calc_regression_metrics(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    df = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if df.empty:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}

    y_true = df["actual"].astype(float)
    y_pred = df["pred"].astype(float)

    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100)

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape,
    }


@st.cache_data(show_spinner=False)
def extract_tables_from_notebook(ipynb_path: Optional[Path]) -> Dict[str, pd.DataFrame]:
    """Extract dataframe-like HTML outputs from executed notebook cells."""
    out: Dict[str, pd.DataFrame] = {
        "metrics": pd.DataFrame(),
        "feature_importance": pd.DataFrame(),
        "shap_callouts": pd.DataFrame(),
        "presentation_numbers": pd.DataFrame(),
        "insight_table": pd.DataFrame(),
    }

    if ipynb_path is None or not ipynb_path.exists():
        return out

    try:
        nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    except Exception:
        return out

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        cell_source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])

        for output in outputs:
            html = None
            data = output.get("data", {}) if isinstance(output, dict) else {}
            if "text/html" in data:
                html_data = data["text/html"]
                html = "".join(html_data) if isinstance(html_data, list) else str(html_data)

            tables = []
            if html:
                try:
                    tables = pd.read_html(html)
                except Exception:
                    tables = []

            for t in tables:
                cols = [str(c).lower() for c in t.columns]
                if {"metric", "value"}.issubset(set(cols)):
                    out["metrics"] = t.copy()
                elif {"feature", "importance"}.issubset(set(cols)):
                    if out["feature_importance"].empty or len(t) > len(out["feature_importance"]):
                        out["feature_importance"] = t.copy()
                elif {"feature", "mean_abs_shap"}.issubset(set(cols)):
                    out["shap_callouts"] = t.copy()
                elif {"rank", "feature", "mean_abs_shap"}.issubset(set(cols)):
                    out["shap_callouts"] = t.copy()
                elif {"item", "value"}.issubset(set(cols)):
                    out["presentation_numbers"] = t.copy()
                elif {"metric", "value"}.issubset(set(cols)) and "business insight extraction" in cell_source.lower():
                    out["insight_table"] = t.copy()

            # Fallback: parse SHAP callout table from text/plain when HTML parsing is unavailable.
            if out["shap_callouts"].empty and "text/plain" in data:
                text_data = data["text/plain"]
                text_blob = "".join(text_data) if isinstance(text_data, list) else str(text_data)
                parsed_shap = _parse_shap_table_from_text(text_blob)
                if not parsed_shap.empty:
                    out["shap_callouts"] = parsed_shap.copy()

    return out


@st.cache_data(show_spinner=False)
def build_hornets_frame(assets: Dict[str, Optional[Path]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    high_risk = load_csv(assets.get("high_risk"))
    revenue = load_csv(assets.get("revenue"))
    games = load_csv(assets.get("hornets_games"))
    weather = load_csv(assets.get("weather"))

    # Prefer notebook artifact outputs when available
    if not high_risk.empty:
        if "game_date" in high_risk.columns:
            high_risk["game_date"] = _safe_to_datetime(high_risk["game_date"])

        # Remove rows that appear to be placeholder/missing actuals
        if "actual_attendance" in high_risk.columns:
            high_risk = high_risk[high_risk["actual_attendance"].fillna(0) > 0].copy()

    # Fallback if artifact missing
    if high_risk.empty and not games.empty:
        g = games.copy()
        date_col = "date" if "date" in g.columns else None
        loc_col = "location" if "location" in g.columns else None
        if date_col is not None:
            g["game_date"] = _safe_to_datetime(g[date_col])
        if loc_col is not None:
            g = g[g[loc_col].astype(str).str.lower().str.contains("home", na=False)].copy()
        if "attendance" in g.columns:
            g["actual_attendance"] = pd.to_numeric(g["attendance"], errors="coerce")
            g["predicted_attendance"] = g["actual_attendance"]
        if "opponent" not in g.columns:
            g["opponent"] = "Unknown"
        if "day_of_week" not in g.columns and "game_date" in g.columns:
            g["day_of_week"] = g["game_date"].dt.day_name()
        g["demand_tier"] = "Medium"
        g["low_demand_risk_flag"] = 0
        g["key_explanatory_factors"] = "Not available"
        high_risk = g[[
            c for c in [
                "game_date",
                "opponent",
                "day_of_week",
                "predicted_attendance",
                "actual_attendance",
                "demand_tier",
                "low_demand_risk_flag",
                "key_explanatory_factors",
            ]
            if c in g.columns
        ]].copy()

    # Weather merge for optional weather impacts
    weather_merged = high_risk.copy()
    if not weather.empty and not weather_merged.empty:
        w = weather.copy()
        if "date" in w.columns:
            w["date"] = _safe_to_datetime(w["date"])

        if "game_date" in weather_merged.columns and "date" in w.columns:
            weather_merged = weather_merged.merge(
                w[[c for c in ["date", "temp_high", "temp_low", "precipitation", "wind_speed", "weather_description"] if c in w.columns]],
                left_on="game_date",
                right_on="date",
                how="left",
            )

    # KPI computation
    metrics = {
        "total_games": float(len(high_risk)) if not high_risk.empty else np.nan,
        "avg_actual_attendance": float(high_risk["actual_attendance"].mean()) if "actual_attendance" in high_risk.columns and not high_risk.empty else np.nan,
        "avg_predicted_attendance": float(high_risk["predicted_attendance"].mean()) if "predicted_attendance" in high_risk.columns and not high_risk.empty else np.nan,
        "low_demand_games": float(high_risk["low_demand_risk_flag"].sum()) if "low_demand_risk_flag" in high_risk.columns and not high_risk.empty else np.nan,
    }

    if not high_risk.empty and {"actual_attendance", "predicted_attendance"}.issubset(set(high_risk.columns)):
        reg = calc_regression_metrics(
            high_risk["actual_attendance"],
            high_risk["predicted_attendance"],
        )
        metrics.update(reg)

    return high_risk, revenue, weather_merged, metrics


@st.cache_data(show_spinner=False)
def preprocess_reviews_for_app(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def normalize_col(col: str) -> str:
        col = str(col).strip().lower()
        col = re.sub(r"[^a-z0-9]+", "_", col)
        col = re.sub(r"_+", "_", col).strip("_")
        return col

    norm_map = {normalize_col(c): c for c in df.columns}

    def pick(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if normalize_col(c) in norm_map:
                return norm_map[normalize_col(c)]
        for k, v in norm_map.items():
            for c in candidates:
                if normalize_col(c) in k:
                    return v
        return None

    text_col = pick(["review", "review_text", "text", "comment", "content"])
    rating_col = pick(["stars", "rating", "score"])
    date_col = pick(["date", "review_date", "created_at"])
    reviewer_col = pick(["reviewer", "author", "name", "user"])
    local_guide_col = pick(["local guide", "local_guide", "guide"])

    out = df.copy()
    out["review_text_raw"] = out[text_col].fillna("").astype(str).str.strip() if text_col else ""
    out["rating"] = pd.to_numeric(out[rating_col], errors="coerce") if rating_col else np.nan
    out["review_date"] = _safe_to_datetime(out[date_col]) if date_col else pd.NaT
    out["reviewer_name"] = out[reviewer_col].astype(str).str.strip() if reviewer_col else "Unknown"

    if local_guide_col:
        out["is_local_guide"] = out[local_guide_col].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        out["is_local_guide"] = False

    out = out.drop_duplicates()
    out = out.drop_duplicates(subset=["review_text_raw", "reviewer_name", "review_date"], keep="first")
    out["review_text_raw"] = out["review_text_raw"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out[out["review_text_raw"].str.len() >= 5].copy()

    # Lightweight preprocessing for n-grams and word-level charts
    text = out["review_text_raw"].str.lower()
    text = text.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    text = text.str.replace(r"[^a-z\s']", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    out["clean_text"] = text
    out["word_count"] = out["clean_text"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Sentiment
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        out["sentiment_score"] = out["review_text_raw"].fillna("").apply(lambda s: sia.polarity_scores(s)["compound"])
    except Exception:
        try:
            from textblob import TextBlob

            out["sentiment_score"] = out["review_text_raw"].fillna("").apply(lambda s: TextBlob(s).sentiment.polarity)
        except Exception:
            out["sentiment_score"] = 0.0

    out["sentiment_label"] = out["sentiment_score"].apply(
        lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
    )

    return out


@st.cache_data(show_spinner=False)
def build_reviews_frame(assets: Dict[str, Optional[Path]]) -> Dict[str, pd.DataFrame]:
    raw = load_csv(assets.get("google_reviews"))
    review_df = preprocess_reviews_for_app(raw)

    outputs = {
        "raw": raw,
        "reviews": review_df,
        "theme_summary": load_csv(assets.get("review_theme_summary")),
        "recommendations": load_csv(assets.get("review_recommendations")),
        "exec_summary": load_csv(assets.get("review_exec_summary")),
        "rep_pos": load_csv(assets.get("review_pos")),
        "rep_neg": load_csv(assets.get("review_neg")),
        "topic_samples": load_csv(assets.get("review_topic_samples")),
        "mismatch": load_csv(assets.get("review_mismatch")),
    }

    return outputs


def get_api_key_from_sources(sidebar_key: str) -> str:
    secret_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""

    env_key = os.getenv("GEMINI_API_KEY", "")

    return sidebar_key.strip() or secret_key.strip() or env_key.strip()


def generate_gemini_insight(api_key: str, prompt: str) -> str:
    if not api_key:
        return "Gemini API key is missing. Provide it in the sidebar, st.secrets, or GEMINI_API_KEY environment variable."

    try:
        import google.generativeai as genai
    except Exception:
        return "google-generativeai is not installed. Install it with: pip install google-generativeai"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or "No response text returned by Gemini."
        return text
    except Exception as e:
        return f"Gemini call failed: {e}"


def format_int(v: float) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{int(round(v)):,}"


def format_float(v: float, decimals: int = 2) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:,.{decimals}f}"
