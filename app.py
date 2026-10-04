# -*- coding: utf-8 -*-
"""
HESOP — Heritage-aware Explainable Solar Optimization Pipeline
Streamlit dashboard for the Green Heritage Solar XAI study.

Run with:
    streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------
# Config & paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "green_energy_dataset.csv"
RESULTS_DIR = BASE_DIR / "Results"
METHOD_IMG = BASE_DIR / "HESOP Heritage-Aware Solar Retrofit Methodology.png"

TARGET = "Optimal_Solar_Utilization_%"
LEAKAGE_COLS = ["Carbon_Reduction_%", "Payback_Period_Years"]
RANDOM_STATE = 42

GREEN = "#1b8a5a"
ACCENT = "#e8a13a"

st.set_page_config(
    page_title="HESOP | Heritage-Aware Solar Optimization",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem;}
        div[data-testid="stMetric"] {
            background: #f4faf7;
            border: 1px solid #d9ead9;
            border-radius: 10px;
            padding: 12px 16px;
        }
        h1, h2, h3 {color: #14532d;}
        .hesop-note {
            background:#fff8ec;border:1px solid #f0d9a8;border-radius:10px;
            padding:12px 16px;color:#6b4e16;font-size:0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_result_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


@st.cache_resource
def train_deployment_model():
    """Retrain the winning deployment-safe model (Ridge) exactly as the
    notebook does, so the counterfactual explorer can predict live."""
    df = load_dataset()
    features = [c for c in df.columns if c != TARGET and c not in LEAKAGE_COLS]
    X, y = df[features], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    cats = X.select_dtypes(exclude=np.number).columns.tolist()
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cats)],
        remainder="passthrough",
    )
    model = Pipeline(
        [
            ("preprocess", prep),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(Xtr, ytr)
    return model, features, Xte, yte


def img(name: str, caption: str = ""):
    p = RESULTS_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def clean_shap_feature(f: str) -> str:
    if f.startswith("cat__"):
        f = f[5:]
        parts = f.split("_", 1)
        if len(parts) == 2:
            col, val = parts
            return f"{col} = {val}"
        return f
    return f.replace("remainder__", "")


df = load_dataset()
SAFE_FEATURES = [c for c in df.columns if c != TARGET and c not in LEAKAGE_COLS]
CAT_COLS = df.select_dtypes(exclude=np.number).columns.tolist()
NUM_COLS = [c for c in df.select_dtypes(exclude="object").columns if c != TARGET]

q1, q2 = df[TARGET].quantile([1 / 3, 2 / 3]).values


def solar_class(v: float) -> str:
    return "Low" if v <= q1 else ("Medium" if v <= q2 else "High")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview():
    st.title("HESOP — Heritage-aware Explainable Solar Optimization")
    st.markdown(
        "Decision-support dashboard for the study *Explainable Machine Learning for "
        "Optimizing Solar Energy Utilization in Historic Village Buildings while "
        "Preserving Cultural Heritage*."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buildings in dataset", f"{len(df):,}")
    c2.metric("Target", "Solar Utilization %")
    c3.metric("Best model (safe set)", "Ridge — R² 0.679")
    c4.metric("Pareto-optimal records", "27")

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("The framework")
        st.markdown(
            """
            HESOP treats solar retrofitting of historic buildings as a **constrained
            decision-support problem**, not just a prediction task:

            1. **Data-quality & leakage audit** — separates pre-intervention
               predictors from downstream outcomes (`Carbon_Reduction_%`,
               `Payback_Period_Years`).
            2. **Dual modeling** — full-feature benchmark vs. deployment-safe model.
            3. **Ensemble ML comparison** — linear, tree and boosting models.
            4. **Explainable AI** — SHAP + permutation importance.
            5. **Heritage-preserving counterfactuals** — only `Installation_Area_m2`
               is varied; heritage attributes stay fixed.
            6. **Multi-objective screening** — Pareto front over utilization,
               carbon reduction and payback period.
            """
        )
        st.markdown(
            '<div class="hesop-note"><b>Note:</b> Results are computational '
            'screening outputs, not engineering prescriptions. Heritage, structural '
            'and regulatory constraints must be assessed before any real retrofit.</div>',
            unsafe_allow_html=True,
        )
    with right:
        if METHOD_IMG.exists():
            st.image(str(METHOD_IMG), caption="HESOP methodology", use_container_width=True)

    st.divider()
    st.subheader("Dataset audit")
    audit = pd.DataFrame(
        {
            "Feature": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing": df.isna().sum().values,
            "unique": [df[c].nunique() for c in df.columns],
        }
    )
    a1, a2 = st.columns([1.4, 1])
    a1.dataframe(audit, use_container_width=True, hide_index=True)
    stats = df[TARGET].describe().to_frame("Value")
    a2.markdown("**Target statistics**")
    a2.dataframe(stats, use_container_width=True)
    a2.caption(f"Missing values: {int(df.isna().sum().sum())} · Duplicates: {int(df.duplicated().sum())}")


def page_data_explorer():
    st.title("Data Explorer")

    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        sel_type = f1.multiselect("Building type", sorted(df["Building_Type"].unique()), default=sorted(df["Building_Type"].unique()))
        sel_orient = f2.multiselect("Orientation", sorted(df["Orientation"].unique()), default=sorted(df["Orientation"].unique()))
        sel_mat = f3.multiselect("Material", sorted(df["Material"].unique()), default=sorted(df["Material"].unique()))
        sel_ins = f4.multiselect("Insulation", sorted(df["Insulation_Level"].unique()), default=sorted(df["Insulation_Level"].unique()))

    mask = (
        df["Building_Type"].isin(sel_type)
        & df["Orientation"].isin(sel_orient)
        & df["Material"].isin(sel_mat)
        & df["Insulation_Level"].isin(sel_ins)
    )
    dff = df[mask]
    st.caption(f"{len(dff):,} of {len(df):,} buildings selected")
    st.dataframe(dff, use_container_width=True, hide_index=True, height=300)

    st.divider()
    t1, t2 = st.columns(2)
    with t1:
        fig = px.histogram(
            dff, x=TARGET, nbins=30, color_discrete_sequence=[GREEN],
            title="Target distribution — Optimal Solar Utilization (%)",
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        corr = (
            df.select_dtypes(include=np.number)
            .corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .sort_values()
        )
        fig = px.bar(
            x=corr.values, y=corr.index, orientation="h",
            color=corr.values, color_continuous_scale="RdYlGn",
            title="Correlation with solar utilization",
            labels={"x": "Pearson r", "y": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Installation area vs. utilization")
    fig = px.scatter(
        dff, x="Installation_Area_m2", y=TARGET, color="Building_Type",
        hover_data=["Orientation", "Material", "Insulation_Level", "Solar_Potential_kWh_m2"],
        title="Installation Area (m²) vs. Optimal Solar Utilization (%)",
        opacity=0.65,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Group statistics")
    gcol = st.selectbox("Group by", CAT_COLS)
    g = (
        dff.groupby(gcol)[TARGET]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    gc1, gc2 = st.columns([1.2, 1])
    fig = px.bar(
        g.reset_index(), x=gcol, y="mean", error_y="std",
        color_discrete_sequence=[GREEN],
        title=f"Mean solar utilization by {gcol}", labels={"mean": "Mean utilization (%)"},
    )
    gc1.plotly_chart(fig, use_container_width=True)
    gc2.dataframe(g, use_container_width=True)

    with st.expander("Saved analysis figures from the Results folder"):
        img("02_target_distribution.png", "02 — Target distribution")
        img("04_installation_area_relationship.png", "04 — Installation area relationship")


def page_model_performance():
    st.title("Model Performance")

    all_res = load_result_csv("07_all_model_results.csv")
    comp = load_result_csv("06_model_comparison_results.csv")

    safe_rank = (
        all_res[all_res.Feature_Set == "Deployment_Safe"]
        .sort_values(["R2", "RMSE", "MAE"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    best = safe_rank.iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best deployment-safe model", best["Model"])
    m2.metric("R²", f"{best['R2']:.4f}")
    m3.metric("RMSE", f"{best['RMSE']:.3f}")
    m4.metric("MAE", f"{best['MAE']:.3f}")

    st.info(
        "Two feature sets are compared. **Deployment_Safe** excludes `Carbon_Reduction_%` "
        "and `Payback_Period_Years` (post-intervention outcomes) to avoid target leakage. "
        "The **Full** benchmark keeps them — its higher R² illustrates why leakage "
        "auditing matters."
    )

    fig = px.bar(
        all_res.sort_values("R2", ascending=True),
        x="R2", y="Model", color="Feature_Set", orientation="h",
        barmode="group", title="R² by model and feature set",
        color_discrete_map={"Deployment_Safe": GREEN, "Full": ACCENT},
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full results table")
    st.dataframe(
        all_res.sort_values(["Feature_Set", "R2"], ascending=[True, False]),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    preds = load_result_csv("08_best_model_predictions.csv")
    p1, p2 = st.columns(2)
    with p1:
        fig = px.scatter(
            preds, x="Actual", y="Predicted", opacity=0.6,
            title=f"Actual vs. Predicted — {best['Model']} (test set)",
            labels={"Actual": "Actual utilization (%)", "Predicted": "Predicted utilization (%)"},
        )
        lims = [
            min(preds["Actual"].min(), preds["Predicted"].min()),
            max(preds["Actual"].max(), preds["Predicted"].max()),
        ]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Ideal", line=dict(dash="dash", color="gray")))
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        fig = px.histogram(
            preds, x="Residual", nbins=25, color_discrete_sequence=[GREEN],
            title="Residual distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Saved figures from the Results folder"):
        img("09_actual_vs_predicted.png", "09 — Actual vs. predicted")
        img("10_residual_distribution.png", "10 — Residual distribution")


def page_explainability():
    st.title("Explainable AI")

    perm = load_result_csv("11_permutation_feature_importance.csv")
    shap_imp = load_result_csv("13_shap_importance.csv")
    shap_imp["Clean"] = shap_imp["Feature"].map(clean_shap_feature)

    st.info(
        "Feature importance is **associational**: it shows what the model relies on, "
        "not causal effects."
    )

    x1, x2 = st.columns(2)
    with x1:
        st.subheader("Permutation importance")
        top = perm.head(12).sort_values("Importance_Mean")
        fig = px.bar(
            top, x="Importance_Mean", y="Feature", orientation="h",
            error_x="Importance_STD", color_discrete_sequence=[GREEN],
            title="Top predictive drivers (RMSE increase when permuted)",
            labels={"Importance_Mean": "Importance"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(perm, use_container_width=True, hide_index=True)

    with x2:
        st.subheader("SHAP importance")
        tops = shap_imp.head(15).sort_values("MeanAbsSHAP")
        fig = px.bar(
            tops, x="MeanAbsSHAP", y="Clean", orientation="h",
            color_discrete_sequence=[ACCENT],
            title="Mean |SHAP value| (one-hot encoded features)",
            labels={"MeanAbsSHAP": "Mean |SHAP|", "Clean": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(shap_imp[["Clean", "MeanAbsSHAP"]].rename(columns={"Clean": "Feature"}),
                     use_container_width=True, hide_index=True)

    with st.expander("Saved figures from the Results folder"):
        img("12_feature_importance.png", "12 — Permutation feature importance")
        img("14_shap_bar.png", "14 — SHAP summary")


def page_counterfactual():
    st.title("Heritage-Preserving Counterfactual Explorer")
    st.markdown(
        "Pick a building, hold every heritage attribute fixed, and vary only the "
        "**installation area** — the controllable retrofit variable in HESOP — to see "
        "how predicted solar utilization responds."
    )

    model, features, Xte, yte = train_deployment_model()

    mode = st.radio(
        "Building source", ["Existing building from dataset", "Custom building"],
        horizontal=True,
    )

    if mode == "Existing building from dataset":
        idx = st.selectbox(
            "Building record",
            options=df.index.tolist(),
            format_func=lambda i: (
                f"#{i} · {df.loc[i,'Building_Type']} · {df.loc[i,'Orientation']} · "
                f"{df.loc[i,'Material']} · {df.loc[i,'Installation_Area_m2']} m² · "
                f"actual {df.loc[i,TARGET]:.1f}%"
            ),
        )
        base = df.loc[idx, features].copy()
        actual_area = float(df.loc[idx, "Installation_Area_m2"])
        actual_util = float(df.loc[idx, TARGET])
    else:
        c1, c2, c3, c4 = st.columns(4)
        btype = c1.selectbox("Building type", sorted(df["Building_Type"].unique()))
        orient = c2.selectbox("Orientation", sorted(df["Orientation"].unique()))
        mat = c3.selectbox("Material", sorted(df["Material"].unique()))
        ins = c4.selectbox("Insulation", sorted(df["Insulation_Level"].unique()))
        c5, c6, c7, c8 = st.columns(4)
        year = c5.number_input("Year built", int(df["Year_Built"].min()), int(df["Year_Built"].max()), 1900)
        floor = c6.number_input("Floor area (m²)", float(df["Floor_Area_m2"].min()), float(df["Floor_Area_m2"].max()), 100.0)
        solar = c7.number_input("Solar potential (kWh/m²)", float(df["Solar_Potential_kWh_m2"].min()), float(df["Solar_Potential_kWh_m2"].max()), 5.0)
        wind = c8.number_input("Wind potential (m/s)", float(df["Wind_Potential_m_s"].min()), float(df["Wind_Potential_m_s"].max()), 3.0)
        c9, c10 = st.columns(2)
        geo = c9.number_input("Geothermal potential", float(df["Geothermal_Potential"].min()), float(df["Geothermal_Potential"].max()), 1.0)
        demand = c10.number_input("Energy demand (kWh)", float(df["Energy_Demand_kWh"].min()), float(df["Energy_Demand_kWh"].max()), 8000.0)
        base = pd.Series(
            {
                "Building_Type": btype, "Year_Built": year, "Floor_Area_m2": floor,
                "Orientation": orient, "Material": mat, "Insulation_Level": ins,
                "Solar_Potential_kWh_m2": solar, "Wind_Potential_m_s": wind,
                "Geothermal_Potential": geo, "Energy_Demand_kWh": demand,
                "Installation_Area_m2": float(df["Installation_Area_m2"].median()),
            }
        )[features]
        actual_area, actual_util = None, None

    area_min, area_max = int(df["Installation_Area_m2"].min()), int(df["Installation_Area_m2"].max())
    area_grid = np.linspace(area_min, area_max, 60)
    scan = pd.DataFrame([base.copy() for _ in area_grid])
    scan["Installation_Area_m2"] = area_grid
    preds = model.predict(scan[features])

    target_level = st.slider(
        "Target utilization (%)", float(df[TARGET].min()), float(df[TARGET].max()),
        float(df[TARGET].quantile(0.75)), step=0.1,
    )
    eligible = area_grid[preds >= target_level]
    reached = len(eligible) > 0
    rec_area = float(eligible.min()) if reached else float(area_grid[np.argmax(preds)])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=area_grid, y=preds, mode="lines", name="Predicted utilization",
        line=dict(color=GREEN, width=3),
    ))
    fig.add_hline(y=target_level, line_dash="dash", line_color=ACCENT,
                  annotation_text=f"Target {target_level:.1f}%")
    fig.add_vline(x=rec_area, line_dash="dot", line_color="#c0392b",
                  annotation_text=f"{rec_area:.0f} m²")
    if actual_area is not None:
        fig.add_trace(go.Scatter(
            x=[actual_area], y=[model.predict(pd.DataFrame([base])[features])[0]],
            mode="markers", marker=dict(size=12, color="#c0392b", symbol="diamond"),
            name="Current installation",
        ))
    fig.update_layout(
        title="Predicted solar utilization vs. installation area (all other attributes fixed)",
        xaxis_title="Installation area (m²)", yaxis_title="Predicted utilization (%)",
    )
    st.plotly_chart(fig, use_container_width=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Smallest area reaching target", f"{rec_area:.0f} m²" if reached else "Not reachable")
    r2.metric("Predicted utilization at that area", f"{float(preds[np.argmin(np.abs(area_grid - rec_area))]):.2f}%")
    r3.metric("Target reached", "Yes" if reached else "No — best shown")
    if actual_util is not None:
        st.caption(f"For reference: recorded utilization for this building is {actual_util:.2f}% at {actual_area:.0f} m².")

    st.divider()
    st.subheader("Batch results from the original study (100 test profiles)")
    recs = load_result_csv("16_counterfactual_recommendations.csv")
    cf = load_result_csv("15_counterfactual_scenarios.csv")
    b1, b2 = st.columns(2)
    with b1:
        fig = px.histogram(
            recs, x="Recommended_Area_m2", color="Target_Reached",
            color_discrete_map={"True": GREEN, "False": "#c0392b", True: GREEN, False: "#c0392b"},
            title="Recommended minimum installation areas (target = 75th percentile)",
            nbins=30,
        )
        st.plotly_chart(fig, use_container_width=True)
    with b2:
        sel_prof = st.selectbox("Counterfactual profile", sorted(cf["Profile"].unique()))
        gp = cf[cf["Profile"] == sel_prof]
        fig = px.line(
            gp, x="Installation_Area_m2", y="Predicted_Utilization_%",
            title=f"Profile {sel_prof}: utilization vs. area", markers=True,
            color_discrete_sequence=[GREEN],
        )
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Recommendations table"):
        st.dataframe(recs, use_container_width=True, hide_index=True)


def page_pareto():
    st.title("Multi-Objective Pareto Screening")
    st.markdown(
        "Candidates are screened on three objectives: **maximize** solar utilization, "
        "**maximize** carbon reduction, **minimize** payback period. The Pareto set "
        "contains records not dominated on all three."
    )

    pareto = load_result_csv("17_pareto_optimal_retrofit_profiles.csv")

    m1, m2, m3 = st.columns(3)
    m1.metric("Pareto-optimal records", len(pareto))
    m2.metric("Best utilization in Pareto set", f"{pareto[TARGET].max():.2f}%")
    m3.metric("Shortest payback in Pareto set", f"{pareto['Payback_Period_Years'].min():.2f} yrs")

    fig = px.scatter(
        df, x="Payback_Period_Years", y=TARGET,
        color="Carbon_Reduction_%", color_continuous_scale="YlGn",
        opacity=0.45, title="Solar utilization vs. payback period (Pareto set highlighted)",
        labels={"Payback_Period_Years": "Payback period (years)", TARGET: "Solar utilization (%)"},
        hover_data=["Building_Type", "Installation_Area_m2"],
    )
    fig.add_trace(go.Scatter(
        x=pareto["Payback_Period_Years"], y=pareto[TARGET],
        mode="markers", name="Pareto-optimal",
        marker=dict(size=11, color="#c0392b", symbol="star",
                    line=dict(width=1, color="black")),
        text=pareto["Building_Type"],
    ))
    st.plotly_chart(fig, use_container_width=True)

    fig3d = px.scatter_3d(
        pareto, x="Payback_Period_Years", y="Carbon_Reduction_%", z=TARGET,
        color="Building_Type", size="Installation_Area_m2",
        title="Pareto set in objective space",
        labels={TARGET: "Utilization %"},
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("Pareto-optimal retrofit profiles")
    st.dataframe(
        pareto.sort_values(TARGET, ascending=False),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Saved figure from the Results folder"):
        img("18_pareto_tradeoff.png", "18 — Utilization–payback trade-off")


def page_report():
    st.title("Results & Discussion")
    md_path = RESULTS_DIR / "19_results_and_discussion.md"
    if md_path.exists():
        st.markdown(md_path.read_text(encoding="utf-8"))

    st.divider()
    st.subheader("Operational classes")
    st.markdown(
        f"Continuous utilization can be bucketed into data-derived quantile classes "
        f"for dashboards: **Low ≤ {q1:.2f}%**, **Medium ≤ {q2:.2f}%**, **High > {q2:.2f}%**."
    )
    cls = df[[TARGET]].copy()
    cls["Class"] = cls[TARGET].map(solar_class)
    c1, c2 = st.columns([1, 1.4])
    c1.dataframe(cls["Class"].value_counts().rename("Count").to_frame(), use_container_width=True)
    fig = px.histogram(
        cls, x=TARGET, color="Class",
        color_discrete_map={"Low": "#c0392b", "Medium": ACCENT, "High": GREEN},
        title="Utilization classes (quantile-derived)", nbins=30,
    )
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Novelty statement (from the study)")
    st.markdown(
        "> HESOP formulates solar retrofit planning for historic village buildings as a "
        "*heritage-preserving, explainable and multi-objective decision-support problem*: "
        "leakage-aware feature separation, ensemble prediction, model-agnostic/SHAP "
        "explanations, minimal-intervention counterfactuals, and Pareto screening."
    )


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
pages = {
    "Overview": page_overview,
    "Data Explorer": page_data_explorer,
    "Model Performance": page_model_performance,
    "Explainable AI": page_explainability,
    "Counterfactual Explorer": page_counterfactual,
    "Pareto Screening": page_pareto,
    "Report": page_report,
}

st.sidebar.title("HESOP Dashboard")
st.sidebar.caption("Heritage-aware Explainable Solar Optimization Pipeline")
choice = st.sidebar.radio("Navigate", list(pages.keys()))
st.sidebar.divider()
st.sidebar.caption(
    f"Dataset: `{DATA_PATH.name}` — {len(df):,} records\n\n"
    f"Results loaded from `Results/` ({len(list(RESULTS_DIR.glob('*')))} files)"
)
pages[choice]()
