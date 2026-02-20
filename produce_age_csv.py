# ===============================
# LOAD AGE DATA
# ===============================

import requests
import pandas as pd
from io import StringIO

countries = (
        "AUT+BEL+CAN+CHL+COL+CRI+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ISR+ITA+"
        "JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA+W+AUS"
    )

ages = (
    "Y_LE4+Y5T9+Y10T14+Y15T19+Y20T24+Y25T29+Y30T34+Y35T39+"
    "Y40T44+Y45T49+Y50T54+Y55T59+Y60T64+Y65T69+Y70T74+"
    "Y75T79+Y80T84+Y_GE85"
)

key = f"{countries}.POP.PS._T.{ages}."

url = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ELS.SAE,DSD_POPULATION@DF_POP_HIST,1.0/"
    f"{key}"
)

response = requests.get(
    url,
    headers={
        "Accept": "text/csv",
        "User-Agent": "Mozilla/5.0 (Streamlit App)"
    },
    timeout=30
)


if response.status_code != 200:
    st.error(f"Status code: {response.status_code}")
    st.write(response.text[:500])
    st.stop()

      
df = pd.read_csv(StringIO(response.text))

# DEBUG SAFETY: ensure expected columns exist
expected_cols = {"REF_AREA", "TIME_PERIOD", "AGE", "OBS_VALUE"}
if not expected_cols.issubset(set(df.columns)):
        st.error("Unexpected column structure from OECD population API.")
        st.write("Returned columns:", df.columns.tolist())
        st.stop()

df = df[["REF_AREA", "TIME_PERIOD", "AGE", "OBS_VALUE"]]
df.columns = ["Country", "Year", "Age", "Population"]

# ===================================
# TRANSFORM TO PROPORTION AND RESHAPE
# ===================================

df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
df = df.dropna()

df["Total_Pop"] = df.groupby(["Country", "Year"])["Population"].transform("sum")
df["Age_Share"] = df["Population"] / df["Total_Pop"]

df_age = (
    df.pivot_table(
        index=["Country", "Year"],
        columns="Age",
        values="Age_Share"
    ).reset_index()
)

df_age.columns.name = None
df_age.to_csv('df_age.csv', index=False)
