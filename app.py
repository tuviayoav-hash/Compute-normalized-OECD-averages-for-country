import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from linearmodels.panel import PanelOLS

st.set_page_config(layout="wide")
st.title("Normalizing OECD Health System Inputs to Country's Age Structure")
st.markdown(
    """
    Usually, to check how they are faring compared to others, countries' compare their health system's inputs aggregates to the normal OECD average.
    
    However, as different countries have different age compositions, their actual need for health system inputs changes:
    - On the one hand, countries like Germany and Japan - with an older population mix - might have more need for said inputs than others;
    - On the other hand, countries like Israel and Mexico - with a much younger population - might have less need for said inputs.
    
    To tackle this issue, this small app allows comparing a country's input aggregate with the appropriate OECD average, normalized to said country's age structure.
    
    Have a go!
    """
)

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
    },

    "Health Expenditure, Public, Per Capita, USD (PPP)": {
        "file": "health_exp_ppp_public.csv",
        "use_log": True
    },

    "Health Expenditure, Private, Per Capita, USD (PPP)": {
        "file": "health_exp_ppp_private.csv",
         "use_log": True
    },

    "Physicians, Active, per 1,000 Inhabitants": {
        "file": "physicians_active.csv",
        "use_log": False
    },

    "Nurses, Active, per 1,000 Inhabitants": {
        "file": "nurses_active.csv",
        "use_log": False
    },

    "Hospital Beds, Curative, per 1,000 Inhabitants": {
        "file": "hospital_beds.csv",
        "use_log": True
    }
}

# ===============================
# USER INPUT
# ===============================

country_list = sorted(COUNTRY_MAP.keys())

default_country = "Israel"

default_index = country_list.index(default_country) if default_country in country_list else 0

country_label = st.selectbox(
    "Select Country",
    country_list,
    index=default_index
)



country_code = COUNTRY_MAP[country_label]


outcome_label = st.selectbox(
    "Select Health Input of Interest",
    OUTCOME_MAP.keys()
)

selected_outcome = OUTCOME_MAP[outcome_label]

table_name = selected_outcome["file"]
use_log_y = selected_outcome["use_log"]

min_countries = st.selectbox(
    "Minimum countries required in the analysis",
    [5, 10, 15, 20],
    index=1
)
st.caption(
    "Not all country data starts at the same year: some start early, some later on. "
    "To have more robust results, it is better to include in the analysis full panel data. "
    "However, this comes on the expense of 'losing' years from the graph. "
)

exclude_usa = st.checkbox("Exclude USA from the analysis?", value=True)
st.caption(
    "The United States has structurally different health system characteristics "
    "(financing mix, price levels, insurance structure) than the rest of the OECD countries, "
    "which may influence the prediction results."
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
    text="* Prediction analysis estimated in natural logarithms"
         if use_log_y
         else "* Prediction analysis estimated in levels",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.25,
    showarrow=False,
    font=dict(size=11, color="gray"),
    align="left"
)

fig.update_layout(
    xaxis_title="Year",
    yaxis_title=f"{outcome_label}",
    margin=dict(b=140),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# METHODOLOGICAL INFO
# ===============================


st.markdown("### Some methodological information for the nerds")

st.markdown(
"""
The underlying prediction analysis is conducted in two stages.  
The first stage estimates the following regression:
"""
)

st.latex(r"""
Y_{it} = \alpha + \beta X_{it} + \delta_i + \gamma_t + \varepsilon_{it}
""")

st.latex(r"""
\begin{aligned}
\text{Where:} \\
\\
Y_{it} &: \text{Outcome variable for country } i \text{ in year } t \\
\alpha &: \text{Model-wide intercept} \\
X_{it} &: \text{Vector of 5-year age-share proportions} \\
\beta &: \text{Vector of estimated coefficients} \\
\gamma_t &: \text{Year fixed effects} \\
\delta_i &: \text{Country fixed effects} \\
\varepsilon_{it} &: \text{Error term}
\end{aligned}
""")

st.markdown(
"""
Estimation is performed using the within-transformation for fixed effects, which imposes:
"""
)

st.latex(r"""
\sum_i \delta_i = 0
""")

st.markdown(
"""
Also note that for some of the outcomes - for example, the health expenditures, and the hospital beds - data suggests it is better to use log-transformed values in the analysis".
"""
)

st.markdown(
"""
In the second stage, the normalized OECD average is predicted using the country's age structure while **excluding** the country-specific fixed effect.
"""
)

st.latex(r"""
\hat{Y}_{it} = \alpha + \hat{\beta} X_{it} + \hat{\gamma}_t 
""")

st.markdown(
"""
The gap between the actual value and the predicted value is:
"""
)

st.latex(r"""
Y_{it} - \hat{Y}_{it} = \delta_i + \varepsilon_{it}
""")




