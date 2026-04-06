import streamlit as st
from pathlib import Path

from utils import (
    build_hornets_frame,
    build_reviews_frame,
    discover_repo_assets,
    format_float,
    format_int,
)

st.set_page_config(
    page_title="Hornets Analytics Dashboard",
    page_icon="🏀",
    layout="wide",
)


def render_home() -> None:
    st.markdown(
        """
        <style>
        .kpi-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: white;
            padding: 16px 18px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 6px 18px rgba(0,0,0,0.22);
            margin-bottom: 10px;
        }
        .kpi-title {font-size: 0.9rem; opacity: 0.85;}
        .kpi-value {font-size: 1.65rem; font-weight: 700; line-height: 1.2;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    assets = discover_repo_assets()
    _, _, _, hornets_metrics = build_hornets_frame(assets)
    reviews_outputs = build_reviews_frame(assets)
    reviews_df = reviews_outputs["reviews"]

    header_left, header_right = st.columns([0.12, 0.88])
    with header_left:
        logo_path = Path(__file__).resolve().parent / "charlotte-hornets-logo-transparent.png"
        if logo_path.exists():
            st.image(str(logo_path), width=120)
    with header_right:
        st.title("Home")
        st.subheader("Charlotte Hornets Demand Intelligence Dashboard")

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(
            """
            This dashboard operationalizes the completed project notebooks into a live, presentation-ready app.

            It combines two analytical pillars:
            1. XGBoost attendance modeling for game-day demand and revenue opportunity.
            2. NLP-based review intelligence for customer sentiment, strengths, and pain points.

            The app is designed for leadership storytelling and live demo use, with notebook-derived outputs reused whenever available.
            """
        )
    with right:
        st.info(
            "How to use this dashboard:\n"
            "- Open XGBoost Analysis for demand and revenue insights.\n"
            "- Open Reviews Text Analysis for customer voice insights.\n"
            "- Use sidebar filters and executive summaries for presentation-ready talking points."
        )

    st.markdown("### Headline KPIs")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Games Analyzed</div><div class='kpi-value'>{format_int(hornets_metrics.get('total_games', float('nan')))}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Avg Home Attendance</div><div class='kpi-value'>{format_int(hornets_metrics.get('avg_actual_attendance', float('nan')))}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Reviews Analyzed</div><div class='kpi-value'>{format_int(float(len(reviews_df)) if not reviews_df.empty else float('nan'))}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        avg_rating = float(reviews_df["rating"].mean()) if (not reviews_df.empty and "rating" in reviews_df.columns) else float("nan")
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Average Review Rating</div><div class='kpi-value'>{format_float(avg_rating, 2)}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Business Context")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            """
            **Attendance modeling matters**
            - Identifies low-demand games before they happen.
            - Supports targeted promotions and dynamic pricing.
            - Translates demand volatility into quantified revenue opportunities.
            """
        )
    with m2:
        st.markdown(
            """
            **Review intelligence matters**
            - Reveals operational pain points that ratings alone cannot explain.
            - Highlights what customers value most so strengths can be amplified.
            - Converts unstructured fan voice into actionable recommendations.
            """
        )

    st.markdown("### Methodology Note")
    st.caption(
        "This app reuses completed notebook analyses and exported artifacts where available "
        "(for example: high-risk game tables, revenue scenarios, and review insight outputs), "
        "and only recomputes lightweight summaries for interactivity."
    )


home_page = st.Page(render_home, title="Home", icon="🏠", default=True)
xgb_page = st.Page("pages/2_XGBoost_Analysis.py", title="XGBoost Analysis", icon="📈")
reviews_page = st.Page("pages/3_Reviews_Text_Analysis.py", title="Reviews Text Analysis", icon="💬")

logo_sidebar_path = Path(__file__).resolve().parent / "charlotte-hornets-logo-transparent.png"
# Inject custom CSS to resize the logo and its container
# CSS to set height to 100px and center the logo
st.markdown(
    """
    <style>
        [data-testid="stSidebarHeader"] {
            height: 170px;
            display: flex;
            justify-content: center; /* Center horizontally */
            align-items: center;     /* Center vertically */
            padding-top: 1rem;
        }

        [data-testid="stSidebarHeader"] img {
            height: 100px !important;
            width: auto;
            margin: 0 auto;         /* Fallback centering */
        }
    </style>
    """,
    unsafe_allow_html=True
)

if logo_sidebar_path.exists():
    st.logo(str(logo_sidebar_path), size="large")

navigation = st.navigation(
    {
        "Dashboard": [home_page],
        "Analysis Pages": [xgb_page, reviews_page],
    }
)

navigation.run()
