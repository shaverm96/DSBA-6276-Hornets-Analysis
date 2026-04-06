import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

# API key is sourced from st.secrets or environment (no sidebar input).
api_key = get_api_key_from_sources("")

# Filtered frame
view = high_risk.copy()
if not view.empty:
    if year_filter and "game_date" in view.columns:
        view = view[view["game_date"].dt.year.isin(year_filter)].copy()

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
        trend = view.dropna(subset=["game_date", "actual_attendance"]).copy()
        trend = trend.sort_values("game_date")

        # Monthly aggregation with observed-month categorical axis avoids large offseason timeline gaps.
        trend["year_month"] = trend["game_date"].dt.to_period("M").dt.to_timestamp()
        monthly = trend.groupby("year_month", as_index=False)["actual_attendance"].mean()

        # Build a robust 3-month trend that is less sensitive to extreme outlier months.
        q1 = monthly["actual_attendance"].quantile(0.25)
        q3 = monthly["actual_attendance"].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - (1.5 * iqr)
        trend_source = monthly["actual_attendance"].where(monthly["actual_attendance"] >= lower_fence)
        monthly["rolling_3m"] = trend_source.rolling(3, min_periods=1).mean().bfill()

        monthly["ym_label"] = monthly["year_month"].dt.strftime("%Y-%m")
        monthly["monthly_label"] = monthly["actual_attendance"].map(lambda v: f"{v:,.0f}")
        monthly["is_anomaly"] = monthly["actual_attendance"] < lower_fence
        overall_monthly_avg = monthly["actual_attendance"].mean()
        monthly["year"] = monthly["year_month"].dt.year
        monthly["month"] = monthly["year_month"].dt.month

        # Prefer NBA season anchors (October) for cleaner x-axis reading.
        season_anchor = monthly[monthly["month"] == 10][["ym_label", "year"]].copy()
        if not season_anchor.empty:
            tick_values = season_anchor["ym_label"].tolist()
            tick_text = [f"{int(y)}-{str((int(y) + 1) % 100).zfill(2)}" for y in season_anchor["year"]]
            xaxis_title = "Season"
        else:
            yearly_anchor = monthly.groupby("year", as_index=False).first()[["ym_label", "year"]]
            tick_values = yearly_anchor["ym_label"].tolist()
            tick_text = yearly_anchor["year"].astype(str).tolist()
            xaxis_title = "Year"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly["ym_label"],
                y=monthly["actual_attendance"],
                mode="lines+markers",
                name="Monthly Avg",
                line=dict(color="#60a5fa", width=2),
                marker=dict(size=5, color="#60a5fa"),
                opacity=0.65,
                hovertemplate="Month=%{x}<br>Avg Attendance=%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["ym_label"],
                y=monthly["rolling_3m"],
                mode="lines",
                name="3-Month Trend",
                line=dict(color="#22d3ee", width=3),
                hovertemplate="Month=%{x}<br>3-Month Trend=%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["ym_label"],
                y=[overall_monthly_avg] * len(monthly),
                mode="lines",
                name="Overall Monthly Avg",
                line=dict(color="#e5e7eb", width=1.5, dash="dash"),
                hovertemplate="Overall Avg=%{y:,.0f}<extra></extra>",
            )
        )

        anomaly_months = monthly[monthly["is_anomaly"]]
        if not anomaly_months.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_months["ym_label"],
                    y=anomaly_months["actual_attendance"],
                    mode="markers",
                    name="Anomaly Month",
                    marker=dict(color="#ef4444", size=9, symbol="diamond"),
                    hovertemplate="Month=%{x}<br>Anomaly Attendance=%{y:,.0f}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Actual Attendance Over Time<br><sup>Cyan line = 3-month trend | Dashed line = overall average | Red points = anomaly months</sup>",
            xaxis_title=xaxis_title,
            yaxis_title="Actual Attendance",
            legend_title="Series",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(type="category", tickangle=0, tickmode="array", tickvals=tick_values, ticktext=tick_text),
        )
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
            labels={
                "actual_attendance": "Actual Attendance",
                "predicted_attendance": "Predicted Attendance",
                "demand_tier": "Demand Tier",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Actual vs predicted plot unavailable for current filter.")

c3, c4 = st.columns(2)
with c3:
    if not view.empty and {"day_of_week", "actual_attendance"}.issubset(set(view.columns)):
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        overall_avg = view["actual_attendance"].mean()
        by_dow = view.groupby("day_of_week", as_index=False)["actual_attendance"].mean()
        by_dow["pct_from_avg"] = ((by_dow["actual_attendance"] - overall_avg) / overall_avg) * 100
        by_dow["direction"] = np.where(by_dow["pct_from_avg"] >= 0, "Above Avg", "Below Avg")
        by_dow["label"] = by_dow["pct_from_avg"].map(lambda v: f"{v:+.1f}%")
        by_dow["day_of_week"] = pd.Categorical(by_dow["day_of_week"], dow_order)
        by_dow = by_dow.sort_values("day_of_week")
        fig = px.bar(
            by_dow,
            x="day_of_week",
            y="pct_from_avg",
            color="direction",
            color_discrete_map={"Above Avg": "#22c55e", "Below Avg": "#ef4444"},
            title="Attendance by Day of Week vs Overall Average (0% Baseline)",
            text="label",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="% from Overall Average", xaxis_title="Day of Week", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Day-of-week chart unavailable.")

with c4:
    if not view.empty and {"opponent", "actual_attendance"}.issubset(set(view.columns)):
        overall_avg = view["actual_attendance"].mean()
        by_opp = view.groupby("opponent", as_index=False)["actual_attendance"].mean()
        by_opp["pct_from_avg"] = ((by_opp["actual_attendance"] - overall_avg) / overall_avg) * 100
        by_opp["direction"] = np.where(by_opp["pct_from_avg"] >= 0, "Above Avg", "Below Avg")
        by_opp["label"] = by_opp["pct_from_avg"].map(lambda v: f"{v:+.1f}%")

        top5 = by_opp.nlargest(5, "pct_from_avg")
        bottom5 = by_opp.nsmallest(5, "pct_from_avg")
        top_bottom = pd.concat([top5, bottom5], ignore_index=True).drop_duplicates(subset=["opponent"])
        top_bottom = top_bottom.sort_values("pct_from_avg", ascending=False)

        fig = px.bar(
            top_bottom,
            x="opponent",
            y="pct_from_avg",
            color="direction",
            color_discrete_map={"Above Avg": "#22c55e", "Below Avg": "#ef4444"},
            title="Top 5 and Bottom 5 Opponents vs Overall Average (0% Baseline)",
            text="label",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="% from Overall Average", xaxis_title="Opponent", showlegend=False)
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
        fig.update_layout(xaxis_title="Risk Factor", yaxis_title="Frequency")
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
        fig.update_layout(xaxis_title="Precipitation Bucket", yaxis_title="Average Actual Attendance")
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
            labels={
                "scenario": "Scenario",
                "seasonal_recovered_revenue": "Seasonal Recovered Revenue ($)",
            },
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(xaxis_title="Scenario", yaxis_title="Seasonal Recovered Revenue ($)")
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
        "You are a senior sports analytics strategy advisor preparing talking points for executives.\n"
        "Use ONLY the provided dashboard data. Do not invent values, teams, or outcomes.\n"
        "Write in clear business language suitable for a live presentation.\n\n"
        "Return EXACTLY these sections:\n"
        "1) Executive Summary (3 bullets):\n"
        "   - The most important demand, risk, and revenue story from this view.\n"
        "2) Quantified Insights (3-5 bullets):\n"
        "   - Each bullet must reference at least one number from the data context.\n"
        "3) Priority Actions (3 bullets):\n"
        "   - What leadership should do next, tied directly to the quantified findings.\n"
        "4) Risks and Caveats (2 bullets):\n"
        "   - Mention data/model limitations visible in this context.\n"
        "5) Slide Script (max 90 words):\n"
        "   - A concise narrative the presenter can read aloud.\n\n"
        "Style rules:\n"
        "- Be specific, numeric, and decision-oriented.\n"
        "- If a value is missing, say 'not available in current view' instead of guessing.\n"
        "- Keep total response concise and presentation-ready.\n\n"
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
