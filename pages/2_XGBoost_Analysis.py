import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re

from project_utils import (
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

    year_scope = st.selectbox(
        "Season Year",
        options=["All Seasons", "Custom Selection"],
        index=0,
        help="Use All Seasons for a full view, or switch to Custom Selection to pick specific years.",
    )

    if year_scope == "Custom Selection":
        year_filter = st.multiselect(
            "Choose Years",
            options=years,
            default=years,
        )
    else:
        year_filter = years

# API key is sourced from st.secrets or environment (no sidebar input).
api_key = get_api_key_from_sources("")

# Filtered frame
view = high_risk.copy()
if not view.empty:
    if year_filter and "game_date" in view.columns:
        view = view[view["game_date"].dt.year.isin(year_filter)].copy()


def build_marquee_weather_attendance(df: pd.DataFrame) -> pd.DataFrame:
    """Create grouped averages for marquee/non-marquee by weather quality."""
    if df.empty or "actual_attendance" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["actual_attendance"] = pd.to_numeric(work["actual_attendance"], errors="coerce")
    work = work.dropna(subset=["actual_attendance"]).copy()
    if work.empty:
        return pd.DataFrame()

    marquee_col_candidates = [
        "is_marquee_game",
        "marquee_game",
        "marquee_flag",
        "premium_opponent_flag",
        "high_profile_matchup",
    ]
    marquee_col = next((c for c in marquee_col_candidates if c in work.columns), None)

    if marquee_col is not None:
        marquee_raw = work[marquee_col]
        if pd.api.types.is_numeric_dtype(marquee_raw) or pd.api.types.is_bool_dtype(marquee_raw):
            marquee_mask = pd.to_numeric(marquee_raw, errors="coerce").fillna(0).astype(float).gt(0)
        else:
            marquee_mask = marquee_raw.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "marquee"])
    elif "opponent_all_stars_count" in work.columns:
        marquee_mask = pd.to_numeric(work["opponent_all_stars_count"], errors="coerce").fillna(0).ge(1)
    elif "key_explanatory_factors" in work.columns:
        # Existing notebook artifact uses "Weak Opponent" as the non-marquee signal.
        weak_mask = work["key_explanatory_factors"].fillna("").astype(str).str.contains("Weak Opponent", case=False, regex=False)
        marquee_mask = ~weak_mask
    else:
        marquee_mask = pd.Series(False, index=work.index)

    work["game_type"] = np.where(marquee_mask, "Marquee Game", "Non-Marquee Game")

    weather_col_candidates = [
        "weather_category",
        "weather_bucket",
        "weather_flag",
        "is_bad_weather",
        "bad_weather_flag",
    ]
    weather_col = next((c for c in weather_col_candidates if c in work.columns), None)

    if weather_col is not None:
        weather_raw = work[weather_col]
        if pd.api.types.is_numeric_dtype(weather_raw) or pd.api.types.is_bool_dtype(weather_raw):
            bad_weather_mask = pd.to_numeric(weather_raw, errors="coerce").fillna(0).astype(float).gt(0)
        else:
            bad_weather_mask = weather_raw.astype(str).str.strip().str.lower().isin(
                ["1", "true", "yes", "y", "bad", "bad weather", "unfavorable"]
            )
    elif "key_explanatory_factors" in work.columns:
        # Prefer existing model weather categorization when present in risk factors.
        bad_weather_mask = work["key_explanatory_factors"].fillna("").astype(str).str.contains("Bad Weather", case=False, regex=False)
    else:
        bad_weather_mask = pd.Series(False, index=work.index)

    unresolved_weather = ~bad_weather_mask
    if "precipitation" in work.columns:
        unresolved_weather = unresolved_weather & work["precipitation"].notna()

    if unresolved_weather.any():
        precip_mask = pd.Series(False, index=work.index)
        wind_mask = pd.Series(False, index=work.index)
        descr_mask = pd.Series(False, index=work.index)

        if "precipitation" in work.columns:
            precip_mask = pd.to_numeric(work["precipitation"], errors="coerce").fillna(0).ge(0.1)
        if "windy" in work.columns:
            wind_mask = pd.to_numeric(work["windy"], errors="coerce").fillna(0).gt(0)
        elif "wind_speed" in work.columns:
            wind_mask = pd.to_numeric(work["wind_speed"], errors="coerce").fillna(0).ge(15)
        if "weather_description" in work.columns:
            descr_mask = work["weather_description"].fillna("").astype(str).str.contains(
                r"rain|drizzle|storm|thunder|snow|sleet|ice|fog", case=False, regex=True
            )

        weather_from_meteo = precip_mask | wind_mask | descr_mask
        bad_weather_mask = bad_weather_mask | (unresolved_weather & weather_from_meteo)

    work["weather_type"] = np.where(bad_weather_mask, "Bad Weather", "Good Weather")

    grouped = (
        work.groupby(["game_type", "weather_type"], as_index=False)["actual_attendance"]
        .mean()
        .rename(columns={"actual_attendance": "avg_attendance"})
    )

    expected = pd.MultiIndex.from_product(
        [["Marquee Game", "Non-Marquee Game"], ["Good Weather", "Bad Weather"]],
        names=["game_type", "weather_type"],
    )
    grouped = grouped.set_index(["game_type", "weather_type"]).reindex(expected).reset_index()
    return grouped

st.markdown("### Business and Demand KPIs")

low_demand_games = np.nan
low_demand_share = np.nan
if not view.empty and "low_demand_risk_flag" in view.columns:
    low_flag = pd.to_numeric(view["low_demand_risk_flag"], errors="coerce").fillna(0)
    low_demand_games = float(low_flag.sum())
    low_demand_share = 100 * (low_demand_games / max(len(view), 1))

base_recovery_revenue = np.nan
if not revenue.empty and {"scenario", "seasonal_recovered_revenue"}.issubset(set(revenue.columns)):
    rev = revenue.copy()
    rev["scenario"] = rev["scenario"].astype(str).str.lower()
    rev["seasonal_recovered_revenue"] = pd.to_numeric(rev["seasonal_recovered_revenue"], errors="coerce")
    base_row = rev.loc[rev["scenario"].eq("base"), "seasonal_recovered_revenue"]
    if not base_row.empty:
        base_recovery_revenue = float(base_row.iloc[0])

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
k2.metric("Low-Demand Games", format_int(low_demand_games))
k3.metric("Low-Demand Share", f"{low_demand_share:.1f}%" if pd.notna(low_demand_share) else "N/A")
k4.metric("Weekend Premium", format_int(weekday_weekend_diff))
k5.metric("Base Recovery Revenue", f"${base_recovery_revenue:,.0f}" if pd.notna(base_recovery_revenue) else "N/A")

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

st.markdown("### Marquee vs Weather Attendance")
st.caption("Average attendance split by game profile (marquee vs non-marquee) and game-day weather quality.")

marquee_weather_source = weather_merged.copy()
if not marquee_weather_source.empty and year_filter and "game_date" in marquee_weather_source.columns:
    marquee_weather_source = marquee_weather_source[marquee_weather_source["game_date"].dt.year.isin(year_filter)].copy()

marquee_weather_avg = build_marquee_weather_attendance(marquee_weather_source)
with st.container(border=True):
    if not marquee_weather_avg.empty:
        fig = go.Figure()
        color_map = {"Good Weather": "#4A76C2", "Bad Weather": "#EC7C31"}

        for weather_type in ["Good Weather", "Bad Weather"]:
            series = marquee_weather_avg[marquee_weather_avg["weather_type"] == weather_type]
            fig.add_trace(
                go.Bar(
                    x=series["game_type"],
                    y=series["avg_attendance"],
                    name=weather_type,
                    marker_color=color_map[weather_type],
                    hovertemplate=(
                        "Game Type=%{x}<br>Weather=%{fullData.name}<br>Average Attendance=%{y:,.0f}<extra></extra>"
                    ),
                )
            )

        min_y = float(marquee_weather_avg["avg_attendance"].min(skipna=True)) if marquee_weather_avg["avg_attendance"].notna().any() else 0
        max_y = float(marquee_weather_avg["avg_attendance"].max(skipna=True)) if marquee_weather_avg["avg_attendance"].notna().any() else 1
        y_floor = max(min_y - 1200, 0)
        y_ceiling = max_y + 1200

        fig.update_layout(
            barmode="group",
            height=460,
            margin=dict(l=18, r=18, t=26, b=18),
            paper_bgcolor="rgba(17, 8, 68, 0.98)",
            plot_bgcolor="rgba(17, 8, 68, 0.98)",
            xaxis=dict(
                categoryorder="array",
                categoryarray=["Marquee Game", "Non-Marquee Game"],
                tickfont=dict(size=15, color="#e5e7eb"),
                showline=True,
                linecolor="#a3a3a3",
                gridcolor="rgba(255,255,255,0)",
            ),
            yaxis=dict(
                title=dict(text="Average Attendance", font=dict(size=13, color="#e5e7eb")),
                tickfont=dict(size=12, color="#e5e7eb"),
                range=[y_floor, y_ceiling],
                gridcolor="rgba(148, 163, 184, 0.28)",
                zeroline=False,
            ),
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                font=dict(size=13, color="#e5e7eb"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Marquee vs weather attendance view unavailable for current filter.")

st.markdown("### Feature Importance and Risk Signals")
st.caption("Frequency and concentration of demand-risk drivers identified in low-demand game diagnostics.")

fi_df = nb_tables.get("feature_importance", pd.DataFrame())
if not fi_df.empty and {"feature", "importance"}.issubset(set([str(c).lower() for c in fi_df.columns])):
    f = fi_df.copy()
    f.columns = [str(c).lower() for c in f.columns]
    f = f[~f["feature"].astype(str).str.lower().str.contains("season", na=False)].copy()
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
        factor_counts["share_pct"] = (factor_counts["count"] / factor_counts["count"].sum()) * 100
        factor_counts["rank"] = np.arange(1, len(factor_counts) + 1)
        factor_counts["bar_label"] = factor_counts["share_pct"].map(lambda v: f"{v:.1f}%")

        top_factors = factor_counts.head(10).copy()
        fig = px.bar(
            top_factors,
            x="factor",
            y="count",
            title="Most Common Risk Factors (Artifact Output)",
            text="bar_label",
            color="share_pct",
            color_continuous_scale="Blues",
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate=(
                "Risk Factor=%{x}<br>Frequency=%{y}<br>Share of All Signals=%{customdata[0]:.1f}%<br>"
                "Rank=%{customdata[1]}<extra></extra>"
            ),
            customdata=top_factors[["share_pct", "rank"]].to_numpy(),
        )
        fig.update_layout(
            xaxis_title="Risk Factor",
            yaxis_title="Frequency",
            coloraxis_colorbar_title="Share %",
        )
        st.plotly_chart(fig, use_container_width=True)

        total_signals = float(factor_counts["count"].sum())
        top1_share = float(factor_counts.iloc[0]["share_pct"]) if not factor_counts.empty else np.nan
        top3_share = float(factor_counts.head(3)["share_pct"].sum()) if not factor_counts.empty else np.nan

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Risk Signals", format_int(total_signals))
        s2.metric("Top Risk Factor Share", f"{top1_share:.1f}%" if pd.notna(top1_share) else "N/A")
        s3.metric("Top 3 Factor Concentration", f"{top3_share:.1f}%" if pd.notna(top3_share) else "N/A")

        details = factor_counts.copy()
        details = details[[c for c in ["rank", "factor", "count", "share_pct"] if c in details.columns]].copy()
        if "share_pct" in details.columns:
            details["share_pct"] = details["share_pct"].map(lambda v: f"{float(v):.1f}%" if pd.notna(v) else "N/A")
        details = details.rename(
            columns={
                "rank": "Rank",
                "factor": "Risk Factor",
                "count": "Frequency",
                "share_pct": "Share of Signals",
            }
        )
        st.dataframe(
            details.head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Risk Factor": st.column_config.TextColumn(width="medium"),
                "Frequency": st.column_config.NumberColumn(width="small"),
                "Share of Signals": st.column_config.TextColumn(width="small"),
            },
        )
    else:
        st.info("Feature/risk importance view unavailable.")

st.markdown("### SHAP Explainability (Notebook Output)")
st.caption("SHAP quantifies each feature's average contribution magnitude to model predictions on the holdout sample.")

shap_df = nb_tables.get("shap_callouts", pd.DataFrame())
if not shap_df.empty:
    shap_work = shap_df.copy()
    shap_work.columns = [str(c).lower() for c in shap_work.columns]

    if not {"feature", "mean_abs_shap"}.issubset(set(shap_work.columns)):
        shap_work = pd.DataFrame()

if not shap_df.empty and not shap_work.empty:

    if "rank" not in shap_work.columns:
        shap_work = shap_work.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_work["rank"] = shap_work.index + 1

    shap_work["mean_abs_shap"] = pd.to_numeric(shap_work["mean_abs_shap"], errors="coerce")
    shap_work = shap_work[~shap_work["feature"].astype(str).str.lower().str.contains("season", na=False)].copy()
    shap_work = shap_work.dropna(subset=["feature", "mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False)
    shap_work = shap_work.reset_index(drop=True)
    shap_work["rank"] = shap_work.index + 1

    if shap_work.empty:
        st.info("Season-related SHAP features were removed and no other SHAP rows remain.")
    else:
        shap_top = shap_work.head(12).copy()

        fig = px.bar(
            shap_top.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top SHAP Drivers (Mean Absolute SHAP, Excluding Season)",
            color="mean_abs_shap",
            color_continuous_scale="Blues",
            labels={"mean_abs_shap": "Mean |SHAP| Impact", "feature": "Feature"},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        sd1, sd2 = st.columns(2)
        total_shap = float(shap_top["mean_abs_shap"].sum()) if not shap_top.empty else np.nan
        top3_shap = float(shap_top.head(3)["mean_abs_shap"].sum()) if not shap_top.empty else np.nan
        top3_concentration = (100 * top3_shap / total_shap) if pd.notna(total_shap) and total_shap > 0 else np.nan

        sd1.metric("Top SHAP Driver", str(shap_top.iloc[0]["feature"]) if not shap_top.empty else "N/A")
        sd2.metric(
            "Top 3 Driver Concentration",
            f"{top3_concentration:.1f}%" if pd.notna(top3_concentration) else "N/A",
        )

        shap_display = shap_top[[c for c in ["rank", "feature", "mean_abs_shap"] if c in shap_top.columns]].copy()
        shap_display = shap_display.rename(
            columns={"rank": "Rank", "feature": "Feature", "mean_abs_shap": "Mean |SHAP| Impact"}
        )
        if "Mean |SHAP| Impact" in shap_display.columns:
            shap_display["Mean |SHAP| Impact"] = shap_display["Mean |SHAP| Impact"].map(
                lambda v: f"{v:,.4f}" if pd.notna(v) else "N/A"
            )
        st.dataframe(shap_display, use_container_width=True, hide_index=True)
else:
    st.info(
        "SHAP outputs were not found in notebook artifacts yet. Run the SHAP cells in hornets_attendance_xgboost.ipynb and refresh this page."
    )

st.markdown("### Weather and Revenue Opportunity")
st.caption("Weather sensitivity and recoverable revenue are shown with scenario context for budget and operations planning.")

w1, w2 = st.columns(2)
with w1:
    if not weather_merged.empty and {"precipitation", "actual_attendance"}.issubset(set(weather_merged.columns)):
        wm = weather_merged.copy()
        wm["precip_bin"] = pd.cut(wm["precipitation"].fillna(0), bins=[-0.001, 0, 0.1, 0.5, 100], labels=["0", "0-0.1", "0.1-0.5", ">0.5"])
        wp = wm.groupby("precip_bin", as_index=False)["actual_attendance"].mean()
        wp["precip_bin"] = wp["precip_bin"].astype(str)
        wp["precip_bin_label"] = wp["precip_bin"].replace(
            {
                "0": "0.0 in",
                "0-0.1": "0.0-0.1 in",
                "0.1-0.5": "0.1-0.5 in",
                ">0.5": ">0.5 in",
            }
        )
        overall_weather_avg = float(wm["actual_attendance"].mean())
        wp["pct_vs_overall"] = ((wp["actual_attendance"] - overall_weather_avg) / overall_weather_avg) * 100
        wp["label"] = wp["pct_vs_overall"].map(lambda v: f"{v:+.1f}%")

        fig = px.bar(
            wp,
            x="precip_bin_label",
            y="actual_attendance",
            title="Attendance by Precipitation Bucket (0.1 in Threshold Highlighted)",
            color="pct_vs_overall",
            color_continuous_scale="RdYlGn",
            text="label",
            category_orders={"precip_bin_label": ["0.0 in", "0.0-0.1 in", "0.1-0.5 in", ">0.5 in"]},
        )
        fig.add_hline(y=overall_weather_avg, line_dash="dash", line_color="white", annotation_text="Overall Avg", annotation_position="top left")
        fig.update_traces(
            textposition="outside",
            hovertemplate=(
                "Precipitation Bucket=%{x}<br>Average Attendance=%{y:,.0f}<br>"
                "vs Overall=%{customdata[0]:+.1f}%<extra></extra>"
            ),
            customdata=wp[["pct_vs_overall"]].to_numpy(),
        )
        fig.add_annotation(
            x=1,
            y=1.08,
            xref="paper",
            yref="paper",
            text="0.1 in is the cutoff between light rain and moderate+ precipitation buckets.",
            showarrow=False,
            font=dict(size=12, color="#cbd5e1"),
            align="right",
        )
        fig.update_layout(
            xaxis_title="Precipitation Bucket (inches)",
            yaxis_title="Average Actual Attendance",
            coloraxis_colorbar_title="% vs Overall",
        )
        st.plotly_chart(fig, use_container_width=True)

        wettest = wp.loc[wp["actual_attendance"].idxmin()] if not wp.empty else None
        driest = wp.loc[wp["actual_attendance"].idxmax()] if not wp.empty else None
        ww1, ww2, ww3 = st.columns(3)
        ww1.metric("Overall Weather Avg", format_int(overall_weather_avg))
        ww2.metric(
            "Highest Attendance Bucket",
            f"{driest['precip_bin_label']} ({driest['pct_vs_overall']:+.1f}%)" if driest is not None else "N/A",
        )
        ww3.metric(
            "Lowest Attendance Bucket",
            f"{wettest['precip_bin_label']} ({wettest['pct_vs_overall']:+.1f}%)" if wettest is not None else "N/A",
        )
    else:
        st.info("Weather impact chart unavailable.")

with w2:
    if not revenue.empty:
        rev = revenue.copy()
        rev["scenario"] = rev["scenario"].astype(str)
        base_value = pd.to_numeric(
            rev.loc[rev["scenario"].str.lower().eq("base"), "seasonal_recovered_revenue"],
            errors="coerce",
        )
        if base_value.empty:
            base_rev = float(pd.to_numeric(rev["seasonal_recovered_revenue"], errors="coerce").median())
        else:
            base_rev = float(base_value.iloc[0])

        rev["seasonal_recovered_revenue"] = pd.to_numeric(rev["seasonal_recovered_revenue"], errors="coerce")
        rev["pct_vs_base"] = ((rev["seasonal_recovered_revenue"] - base_rev) / base_rev) * 100
        rev["label"] = rev["pct_vs_base"].map(lambda v: f"{v:+.1f}% vs Base")

        fig = px.bar(
            rev,
            x="scenario",
            y="seasonal_recovered_revenue",
            text="seasonal_recovered_revenue",
            title="Projected Seasonal Recovered Revenue",
            labels={
                "scenario": "Scenario",
                "seasonal_recovered_revenue": "Seasonal Recovered Revenue ($)",
            },
            color="pct_vs_base",
            color_continuous_scale="Blues",
        )
        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            hovertemplate=(
                "Scenario=%{x}<br>Recovered Revenue=$%{y:,.0f}<br>"
                "Change vs Base=%{customdata[0]:+.1f}%<extra></extra>"
            ),
            customdata=rev[["pct_vs_base"]].to_numpy(),
        )
        fig.update_layout(
            xaxis_title="Scenario",
            yaxis_title="Seasonal Recovered Revenue ($)",
            coloraxis_colorbar_title="% vs Base",
        )
        st.plotly_chart(fig, use_container_width=True)

        best_row = rev.sort_values("seasonal_recovered_revenue", ascending=False).iloc[0]
        worst_row = rev.sort_values("seasonal_recovered_revenue", ascending=True).iloc[0]
        spread_value = float(best_row["seasonal_recovered_revenue"] - worst_row["seasonal_recovered_revenue"])

        rv1, rv2, rv3 = st.columns(3)
        rv1.metric("Base Scenario Revenue", f"${base_rev:,.0f}" if pd.notna(base_rev) else "N/A")
        rv2.metric("Best Scenario", f"{str(best_row['scenario']).title()} (${best_row['seasonal_recovered_revenue']:,.0f})")
        rv3.metric("Scenario Spread", f"${spread_value:,.0f}")

        table_cols = [
            c
            for c in [
                "scenario",
                "seasonal_recovered_revenue",
                "pct_vs_base",
                "flagged_low_demand_games",
                "avg_recovered_attendees_per_game",
                "avg_recovered_revenue_per_game",
            ]
            if c in rev.columns
        ]
        table_view = rev[table_cols].copy()
        if not table_view.empty:
            # Business-friendly headers and formatted values for presentation readability.
            display_table = table_view.copy()

            if "scenario" in display_table.columns:
                display_table["scenario"] = display_table["scenario"].astype(str).str.title()
            if "seasonal_recovered_revenue" in display_table.columns:
                display_table["seasonal_recovered_revenue"] = pd.to_numeric(display_table["seasonal_recovered_revenue"], errors="coerce").map(
                    lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
                )
            if "pct_vs_base" in display_table.columns:
                display_table["pct_vs_base"] = pd.to_numeric(display_table["pct_vs_base"], errors="coerce").map(
                    lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A"
                )
            if "flagged_low_demand_games" in display_table.columns:
                display_table["flagged_low_demand_games"] = pd.to_numeric(display_table["flagged_low_demand_games"], errors="coerce").map(
                    lambda v: f"{int(v):,}" if pd.notna(v) else "N/A"
                )
            if "avg_recovered_attendees_per_game" in display_table.columns:
                display_table["avg_recovered_attendees_per_game"] = pd.to_numeric(display_table["avg_recovered_attendees_per_game"], errors="coerce").map(
                    lambda v: f"{v:,.0f}" if pd.notna(v) else "N/A"
                )
            if "avg_recovered_revenue_per_game" in display_table.columns:
                display_table["avg_recovered_revenue_per_game"] = pd.to_numeric(display_table["avg_recovered_revenue_per_game"], errors="coerce").map(
                    lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
                )

            display_table = display_table.rename(
                columns={
                    "scenario": "Scenario",
                    "seasonal_recovered_revenue": "Seasonal Recovered Revenue",
                    "pct_vs_base": "Change vs Base",
                    "flagged_low_demand_games": "Flagged Low-Demand Games",
                    "avg_recovered_attendees_per_game": "Avg Recovered Attendees per Game",
                    "avg_recovered_revenue_per_game": "Avg Recovered Revenue per Game",
                }
            )
            st.dataframe(display_table, use_container_width=True, hide_index=True)
    else:
        st.info("Revenue scenario artifact not found.")

st.markdown("### High-Risk Games Table")
if not view.empty:
    cols = [c for c in ["game_date", "opponent", "day_of_week", "predicted_attendance", "actual_attendance", "demand_tier", "low_demand_risk_flag", "key_explanatory_factors"] if c in view.columns]
    table_view = view.sort_values("predicted_attendance").head(50)[cols].copy()

    if "game_date" in table_view.columns:
        table_view["game_date"] = pd.to_datetime(table_view["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "opponent" in table_view.columns:
        table_view["opponent"] = table_view["opponent"].astype(str)
    if "day_of_week" in table_view.columns:
        table_view["day_of_week"] = table_view["day_of_week"].astype(str)
    if "predicted_attendance" in table_view.columns:
        table_view["predicted_attendance"] = pd.to_numeric(table_view["predicted_attendance"], errors="coerce").map(
            lambda v: f"{v:,.0f}" if pd.notna(v) else "N/A"
        )
    if "actual_attendance" in table_view.columns:
        table_view["actual_attendance"] = pd.to_numeric(table_view["actual_attendance"], errors="coerce").map(
            lambda v: f"{v:,.0f}" if pd.notna(v) else "N/A"
        )
    if "demand_tier" in table_view.columns:
        table_view["demand_tier"] = table_view["demand_tier"].astype(str).str.title()
    if "low_demand_risk_flag" in table_view.columns:
        table_view["low_demand_risk_flag"] = pd.to_numeric(table_view["low_demand_risk_flag"], errors="coerce").map(
            lambda v: "Yes" if pd.notna(v) and int(v) == 1 else "No"
        )
    if "key_explanatory_factors" in table_view.columns:
        table_view["key_explanatory_factors"] = (
            table_view["key_explanatory_factors"]
            .fillna("")
            .astype(str)
            .str.replace(",", " | ", regex=False)
        )

    table_view = table_view.rename(
        columns={
            "game_date": "Game Date",
            "opponent": "Opponent",
            "day_of_week": "Day of Week",
            "predicted_attendance": "Predicted Attendance",
            "actual_attendance": "Actual Attendance",
            "demand_tier": "Demand Tier",
            "low_demand_risk_flag": "Low-Demand Risk",
            "key_explanatory_factors": "Key Explanatory Factors",
        }
    )
    st.dataframe(table_view, use_container_width=True, height=420, hide_index=True)

    with st.expander("Key Explanatory Factor Definitions"):
        factor_definitions = pd.DataFrame(
            [
                {
                    "Factor": "Weekday",
                    "Definition": "Game is scheduled on a weekday, which often has lower turnout than weekend games.",
                },
                {
                    "Factor": "Weak Opponent",
                    "Definition": "Opponent is projected to draw lower fan interest relative to stronger marquee matchups.",
                },
                {
                    "Factor": "Bad Weather",
                    "Definition": "Weather conditions are likely unfavorable for attendance (for example precipitation or severe conditions).",
                },
                {
                    "Factor": "Dense Schedule",
                    "Definition": "Game falls in a tightly packed sequence of events, which can reduce demand and discretionary attendance.",
                },
                {
                    "Factor": "No major risk signal",
                    "Definition": "No primary attendance risk driver was identified in the model diagnostics for this game.",
                },
            ]
        )

        present_factors = set()
        if "Key Explanatory Factors" in table_view.columns:
            for raw in table_view["Key Explanatory Factors"].dropna().astype(str):
                parts = [p.strip() for p in raw.split("|") if p.strip()]
                present_factors.update(parts)

        if present_factors:
            factor_definitions = factor_definitions[factor_definitions["Factor"].isin(present_factors)].copy()

        st.dataframe(factor_definitions, use_container_width=True, hide_index=True)
else:
    st.warning("No high-risk table data available after filters.")

st.markdown("### LLM Business Insights")


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

if st.button("Generate Executive Insight Summary", type="primary"):
    context_payload = {
        "filters": {
            "years": "All Seasons" if year_scope == "All Seasons" else year_filter,
        },
        "kpis": {
            "avg_actual_attendance": metrics.get("avg_actual_attendance", np.nan),
            "low_demand_games": metrics.get("low_demand_games", np.nan),
            "weekday_weekend_diff": weekday_weekend_diff,
        },
        "revenue_scenarios": revenue.to_dict(orient="records")[:3] if not revenue.empty else [],
        "top_risk_rows": view.sort_values("predicted_attendance").head(8).to_dict(orient="records") if not view.empty else [],
    }

    prompt = (
        "You are a senior sports analytics strategy advisor preparing talking points for executives.\n"
        "Use ONLY the provided dashboard data. Do not invent values, teams, or outcomes.\n"
        "Write in clear business language suitable for a live presentation.\n"
        "Pull details directly from the page context (filters, KPIs, revenue scenarios, and top risk rows).\n\n"
        "Do NOT focus on model-performance commentary; focus on business impact, operational risk, and revenue implications.\n\n"
        "Return EXACTLY these sections:\n"
        "1) Executive Summary (3 bullets):\n"
        "   - The most important demand, risk, and revenue story from this view.\n"
        "2) Demand and Risk Signals (4 bullets):\n"
        "   - Use specific values from KPIs and top risk rows (attendance levels, risk concentration, or context effects).\n"
        "3) Revenue Scenario Interpretation (3 bullets):\n"
        "   - Compare conservative/base/optimistic outcomes with explicit dollar differences or percentage gaps.\n"
        "4) Quantified Findings (4-6 bullets):\n"
        "   - Each bullet must include at least one concrete number from the data context.\n"
        "5) Priority Actions and Triggers (3 bullets):\n"
        "   - Actionable decisions tied to measurable triggers (for example thresholds or risk flags).\n"
        "6) Slide Script (max 110 words):\n"
        "   - A concise narrative the presenter can read aloud.\n\n"
        "Style rules:\n"
        "- Be specific, numeric, and decision-oriented.\n"
        "- Prefer concrete comparisons over generic statements.\n"
        "- If a value is missing, say 'not available in current view' instead of guessing.\n"
        "- Keep total response concise and presentation-ready.\n\n"
        f"Data context:\n{context_payload}"
    )

    with st.spinner("Calling Gemini..."):
        text = generate_gemini_insight(api_key, prompt)
    render_executive_sections(text)
