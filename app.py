import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from linearmodels.panel import PanelOLS

st.set_page_config(layout="wide")
st.title("OECD Health System Aggregates & Age Structure")

# ===============================
# MAPS
# ===============================
COUNTRY_MAP = {
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Canada": "CAN",
    "Chile": "CHL",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "Czech Republic": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Japan": "JPN",
    "Korea": "KOR",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Mexico": "MEX",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "PRT",
    "Slovak Republic": "SVK",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Türkiye": "TUR",
    "United Kingdom": "GBR",
    "United States": "USA"
}

OUTCOME_MAP = {
    "Health Expenditure, Total, Per Capita, USD (PPP)": {
        "file": "health_exp_ppp_total.csv",
        "use_log": True
    }

}

# ===============================
# USER INPUT
# ===============================

country_label = st.selectbox(
    "Select Country",
    sorted(COUNTRY_MAP.keys())
)
exclude_usa = st.checkbox("Exclude USA from the analysis?", value=False)

country_code = COUNTRY_MAP[country_label]


outcome_label = st.selectbox(
    "Select Health Input of Interest",
    OUTCOME_MAP.keys()
)

selected_outcome = OUTCOME_MAP[outcome_label]

table_name = selected_outcome["file"]
use_log_y = selected_outcome["use_log"]

min_countries = st.selectbox(
    "Minimum countries required",
    [5, 10, 15, 20],
    index=1
)
# ===============================
# RUN PIPELINE
# ===============================

df_age = pd.read_csv("data/age_data.csv")
df_outcome = pd.read_csv(f"data/{table_name}")

df_reg = df_age.merge(df_outcome, on=["Country", "Year"], how="inner")


if exclude_usa:
    df_reg = df_reg[df_reg["Country"] != "USA"].copy()
    


year_counts = df_reg.groupby("Year")["Country"].nunique()
eligible_years = year_counts[year_counts >= min_countries]

if eligible_years.empty:
    st.error("No years satisfy minimum country threshold.")
    st.stop()

first_year = eligible_years.index.min()

# Drop last year from data, as it tends to have a lot of missings
last_year = df_reg["Year"].max()

df_reg = df_reg[
    (df_reg["Year"] >= first_year) &
    (df_reg["Year"] < last_year)
].copy()


if use_log_y:
    if (df_reg["Outcome"] <= 0).any():
        st.error("Non-positive expenditure values found.")
        st.stop()
    df_reg["Outcome"] = np.log(df_reg["Outcome"])

age_cols = [c for c in df_reg.columns if c not in ["Country","Year","Outcome"]]
oldest_age = "Y_GE85"
X_cols = [c for c in age_cols if c != oldest_age]

df_reg = df_reg.set_index(["Country","Year"]).sort_index()

y = df_reg["Outcome"]
X = df_reg[X_cols]

mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
res = mod.fit()

beta = res.params
r = y - X @ beta

mu_raw = r.groupby(level="Country").mean()
mu = mu_raw - mu_raw.mean()

r_minus_mu = r - mu.reindex(df_reg.index.get_level_values("Country")).values
gamma_raw = r_minus_mu.groupby(level="Year").mean()
gamma = gamma_raw - gamma_raw.mean()

alpha = (r - mu.reindex(df_reg.index.get_level_values("Country")).values
           - gamma.reindex(df_reg.index.get_level_values("Year")).values).mean()

# ===============================
# PREDICTION (NO COUNTRY FE)
# ===============================

df_country = df_reg.xs(country_code, level="Country")
xb = df_country[X_cols] @ beta

y_hat = alpha + xb + gamma.reindex(df_country.index).values

if use_log_y:
    actual = np.exp(df_country["Outcome"])
    y_hat = np.exp(y_hat)
    oecd_mean = np.exp(df_reg["Outcome"].groupby("Year").mean())
else:
    actual = df_country["Outcome"]
    oecd_mean = df_reg["Outcome"].groupby("Year").mean()

# ===============================
# PLOT
# ===============================

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=actual.index,
    y=actual.values,
    mode="lines",
    name=f"{country_label} Actual Values"
))

fig.add_trace(go.Scatter(
    x=oecd_mean.index,
    y=oecd_mean.values,
    mode="lines",
    name="OECD Actual Mean (Geometric)" if use_log_y else "OECD Actual Mean (Arithmetic)"

))

fig.add_trace(go.Scatter(
    x=y_hat.index,
    y=y_hat.values,
    mode="lines",
    name="OECD Age-Structure Normalized Mean"
))

fig.add_annotation(
    text="Prediction analysis estimated in natural logarithms"
         if use_log_y
         else "Prediction analysis estimated in levels",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.15,
    showarrow=False,
    font=dict(size=12),
    align="left"
)

fig.update_layout(
    xaxis_title="Year",
    yaxis_title=f"{selected_outcome}",
    margin=dict(b=100)
)

st.plotly_chart(fig, use_container_width=True)
