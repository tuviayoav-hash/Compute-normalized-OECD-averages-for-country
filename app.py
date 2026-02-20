import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from linearmodels.panel import PanelOLS

st.set_page_config(layout="wide")
st.title("OECD Health Expenditure & Age Structure")

# ===============================
# USER INPUT
# ===============================

country_name = st.selectbox(
    "Select Country",
    ["AUS","AUT","BEL","CAN","CHL","COL","CRI","CZE","DNK","EST",
     "FIN","FRA","DEU","GRC","HUN","ISL","IRL","ISR","ITA","JPN",
     "KOR","LVA","LTU","LUX","MEX","NLD","NZL","NOR","POL","PRT",
     "SVK","SVN","ESP","SWE","CHE","TUR","GBR","USA"]
)

outcome = st.selectbox(
    "Select outcome of interest",
    ["health_exp_ppp_total"]
)

first_year = st.slider("Start Year", 1980, 2015, 1995)
use_log_y = st.checkbox("Use log of expenditure", value=True)

# ===============================
# RUN PIPELINE
# ===============================

df_age = pd.read_csv("data/age_data.csv")
df_outcome = pd.read_csv("data/health_exp_ppp_total.csv")

df_reg = df_age.merge(df_outcome, on=["Country", "Year"], how="inner")

last_year = df_reg["Year"].max()
df_reg = df_reg[(df_reg["Year"] >= first_year) & (df_reg["Year"] < last_year)].copy()

y_col = "Health_Exp"

if use_log_y:
    if (df_reg[y_col] <= 0).any():
        st.error("Non-positive expenditure values found.")
        st.stop()
    df_reg[y_col] = np.log(df_reg[y_col])

age_cols = [c for c in df_reg.columns if c not in ["Country","Year",y_col]]
oldest_age = "Y_GE85"
X_cols = [c for c in age_cols if c != oldest_age]

df_reg = df_reg.set_index(["Country","Year"]).sort_index()

y = df_reg[y_col]
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

df_country = df_reg.xs(country_name, level="Country")
xb = df_country[X_cols] @ beta

y_hat = alpha + xb + gamma.reindex(df_country.index).values

if use_log_y:
    actual = np.exp(df_country[y_col])
    y_hat = np.exp(y_hat)
    oecd_mean = np.exp(df_reg[y_col].groupby("Year").mean())
else:
    actual = df_country[y_col]
    oecd_mean = df_reg[y_col].groupby("Year").mean()

# ===============================
# PLOT
# ===============================

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=actual.index,
    y=actual.values,
    mode="lines",
    name=f"{country_name} Actual"
))

fig.add_trace(go.Scatter(
    x=oecd_mean.index,
    y=oecd_mean.values,
    mode="lines",
    name="OECD Mean"
))

fig.add_trace(go.Scatter(
    x=y_hat.index,
    y=y_hat.values,
    mode="lines",
    name="OECD Normalized"
))

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Expenditure (PPP per capita)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
