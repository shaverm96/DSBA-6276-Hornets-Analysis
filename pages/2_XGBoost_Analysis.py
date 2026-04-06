import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils import (
    build_hornets_frame,
    discover_repo_assets,
    extract_tables_from_notebook,
    format_float,
    format_int,
    generate_gemini_insight,
    get_api_key_from_sources,
)

st.set_page_config(page_title="XGBoost Analysis", page_icon="📈", layout="wide")

assets = discover_repo_assets()
high_risk, revenue, weather_merged, metrics = build_hornets_frame(assets)
nb_tables = extract_tables_from_notebook(assets.get("notebook_hornets"))

st.title("XGBoost Attendance Analysis")
st.caption("Notebook-grounded model outcomes, demand-risk identification, and revenue opportunity views.")

with st.sidebar:
    st.header("Page Controls")
    st.caption("Filters update visual summaries on this page.")

    if not high_risk.empty and "game_date" in high_risk.columns:
        years = sorted([int(y) for y in high_risk["game_date"].dropna().dt.year.unique()])
    else:
        years = []

    year_filter = st.multiselect("Season Year", options=years, default=years)

    opponent_options = sorted(high_risk["opponent"].dropna().astype(str).unique().tolist()) if "opponent" in high_risk.columns else []
    chosen_opponents = st.multiselect("Opponent", options=opponent_options, default=[])

    st.divider()
    st.subheader("Gemini 2.5 Flash")
    key_input = st.text_input("Gemini API Key", type="password", placeholder="Paste key for live insights")
    api_key = get_api_key_from_sources(key_input)

# Filtered frame
view = high_risk.copy()
if not view.empty:
    if year_filter and "game_date" in view.columns:
        view = view[view["game_date"].dt.year.isin(year_filter)].copy()
    if chosen_opponents and "opponent" in view.columns:
        view = view[view["opponent"].isin(chosen_opponents)].copy()

st.markdown("### Model and Demand KPIs")

metric_source = nb_tables.get("metrics", pd.DataFrame())
if not metric_source.empty and {"metric", "value"}.issubset(set([str(c).lower() for c in metric_source.columns])):
    ms = metric_source.copy()
    ms.columns = [str(c).lower() for c in ms.columns]
    metric_lookup = dict(zip(ms["metric"].astype(str), pd.to_numeric(ms["value"], errors="coerce")))
    r2_value = metric_lookup.get("R2", metric_lookup.get("r2", metrics.get("r2", np.nan)))
    rmse_value = metric_lookup.get("RMSE", metric_lookup.get("rmse", metrics.get("rmse", np.nan)))
    mae_value = metric_lookup.get("MAE", metric_lookup.get("mae", metrics.get("mae", np.nan)))
else:
    r2_value = metrics.get("r2", np.nan)
    rmse_value = metrics.get("rmse", np.nan)
    mae_value = metrics.get("mae", np.nan)

weekday_weekend_diff = np.nan
if not view.empty and {"day_of_week", "actual_attendance"}.issubset(set(view.columns)):
    weekend_days = {"Friday", "Saturday", "Sunday"}
    tmp = view.copy()
    tmp["is_weekend"] = tmp["day_of_week"].isin(weekend_days)
    wkd = tmp.loc[tmp["is_weekend"], "actual_attendance"].mean()
    wkday = tmp.loc[~tmp["is_weekend"], "actual_attendance"].mean()
    weekday_weekend_diff = wkd - wkday

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Actual Attendance", format_int(view["actual_attendance"].mean()) if not view.empty else "N/A")
k2.metric("Model R²", format_float(r2_value, 3))
k3.metric("RMSE", format_float(rmse_value, 1))
k4.metric("MAE", format_float(mae_value, 1))
k5.metric("Weekend - Weekday", format_int(weekday_weekend_diff))

st.markdown("### Attendance and Model Behavior")

c1, c2 = st.columns(2)
with c1:
    if not view.empty and {"game_date", "actual_attendance"}.issubset(set(view.columns)):
        trend = view.groupby("game_date", as_index=False)["actual_attendance"].mean().sort_values("game_date")
        fig = px.line(trend, x="game_date", y="actual_attendance", markers=True, title="Actual Attendance Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Attendance trend unavailable for current filter.")

with c2:
    if not view.empty and {"actual_attendance", "predicted_attendance"}.issubset(set(view.columns)):
        fig = px.scatter(
            view,
            x="actual_attendance",
            y="predicted_attendance",
            color="demand_tier" if "demand_tier" in view.columns else None,
            title="Actual vs Predicted Attendance",
            opacity=0.75,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Actual vs predicted plot unavailable for current filter.")

c3, c4 = st.columns(2)
with c3:
    if not view.empty and {"day_of_week", "actual_attendance"}.issubset(set(view.columns)):
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_dow = view.groupby("day_of_week", as_index=False)["actual_attendance"].mean()
        by_dow["day_of_week"] = pd.Categorical(by_dow["day_of_week"], dow_order)
        by_dow = by_dow.sort_values("day_of_week")
        fig = px.bar(by_dow, x="day_of_week", y="actual_attendance", title="Average Attendance by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Day-of-week chart unavailable.")

with c4:
    if not view.empty and {"opponent", "actual_attendance"}.issubset(set(view.columns)):
        by_opp = view.groupby("opponent", as_index=False)["actual_attendance"].mean().sort_values("actual_attendance", ascending=False)
        fig = px.bar(by_opp.head(12), x="opponent", y="actual_attendance", title="Top Opponents by Average Attendance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Opponent chart unavailable.")

st.markdown("### Feature Importance and Risk Signals")

fi_df = nb_tables.get("feature_importance", pd.DataFrame())
if not fi_df.empty and {"feature", "importance"}.issubset(set([str(c).lower() for c in fi_df.columns])):
    f = fi_df.copy()
    f.columns = [str(c).lower() for c in f.columns]
    f = f.sort_values("importance", ascending=False).head(15)
    fig = px.bar(f.sort_values("importance"), x="importance", y="feature", orientation="h", title="Top Feature Importances (Notebook Output)")
    st.plotly_chart(fig, use_container_width=True)
else:
    if not view.empty and "key_explanatory_factors" in view.columns:
        split_factors = (
            view["key_explanatory_factors"].fillna("").astype(str).str.split(",")
            .explode().str.strip()
        )
        factor_counts = split_factors[split_factors.ne("")].value_counts().reset_index()
        factor_counts.columns = ["factor", "count"]
        fig = px.bar(factor_counts.head(10), x="factor", y="count", title="Most Common Risk Factors (Artifact Output)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature/risk importance view unavailable.")

st.markdown("### Weather and Revenue Opportunity")

w1, w2 = st.columns(2)
with w1:
    if not weather_merged.empty and {"precipitation", "actual_attendance"}.issubset(set(weather_merged.columns)):
        wm = weather_merged.copy()
        wm["precip_bin"] = pd.cut(wm["precipitation"].fillna(0), bins=[-0.001, 0, 0.1, 0.5, 100], labels=["0", "0-0.1", "0.1-0.5", ">0.5"])
        wp = wm.groupby("precip_bin", as_index=False)["actual_attendance"].mean()
        fig = px.bar(wp, x="precip_bin", y="actual_attendance", title="Attendance by Precipitation Bucket")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Weather impact chart unavailable.")

with w2:
    if not revenue.empty:
        fig = px.bar(
            revenue,
            x="scenario",
            y="seasonal_recovered_revenue",
            text="seasonal_recovered_revenue",
            title="Projected Seasonal Recovered Revenue",
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(revenue, use_container_width=True)
    else:
        st.info("Revenue scenario artifact not found.")

st.markdown("### High-Risk Games Table")
if not view.empty:
    cols = [c for c in ["game_date", "opponent", "day_of_week", "predicted_attendance", "actual_attendance", "demand_tier", "low_demand_risk_flag", "key_explanatory_factors"] if c in view.columns]
    st.dataframe(view.sort_values("predicted_attendance").head(50)[cols], use_container_width=True, height=420)
else:
    st.warning("No high-risk table data available after filters.")

st.markdown("### LLM Business Insights (Gemini 2.5 Flash)")

if st.button("Generate Executive Insight Summary", type="primary"):
    context_payload = {
        "filters": {
            "years": year_filter,
            "opponents": chosen_opponents[:10],
        },
        "kpis": {
            "avg_actual_attendance": metrics.get("avg_actual_attendance", np.nan),
            "r2": r2_value,
            "rmse": rmse_value,
            "mae": mae_value,
            "low_demand_games": metrics.get("low_demand_games", np.nan),
            "weekday_weekend_diff": weekday_weekend_diff,
        },
        "revenue_scenarios": revenue.to_dict(orient="records")[:3] if not revenue.empty else [],
        "top_risk_rows": view.sort_values("predicted_attendance").head(8).to_dict(orient="records") if not view.empty else [],
    }

    prompt = (
        "You are an executive sports analytics advisor.\n"
        "Use the provided dashboard data to produce a concise, presentation-ready summary.\n"
        "Return: (1) top 3 insights, (2) top 3 actions, (3) one risk to monitor.\n"
        "Ground every statement in the provided numbers. Do not invent values.\n\n"
        f"Data context:\n{context_payload}"
    )

    with st.spinner("Calling Gemini 2.5 Flash..."):
        text = generate_gemini_insight(api_key, prompt)
    st.write(text)

with st.expander("Methodology Notes"):
    st.write(
        "This page prioritizes notebook-derived artifacts (high-risk game table, revenue scenarios, and executed notebook metric tables). "
        "Fallback computations are used only when an artifact is unavailable for the selected view."
    )
