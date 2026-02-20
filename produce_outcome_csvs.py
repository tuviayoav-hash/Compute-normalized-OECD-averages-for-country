import requests
import pandas as pd
from io import StringIO

# ===============================
# OUTCOMES KEYS
# ===============================

OUTCOMES = {
  "health_exp_ppp_total": {
    "source": "OECD.ELS.HD,DSD_SHA@DF_SHA,1.0/",
    "specs": "A.EXP_HEALTH.USD_PPP_PS._T.._T._T._T...Q"
  },
  
  "health_exp_ppp_public": {
    "source": "OECD.ELS.HD,DSD_SHA@DF_SHA,1.0/",
    "specs": "A.EXP_HEALTH.USD_PPP_PS.HF1.._T._T._T...Q"
  },

  "health_exp_ppp_private": {
    "source": "OECD.ELS.HD,DSD_SHA@DF_SHA,1.0/",
    "specs": "A.EXP_HEALTH.USD_PPP_PS.HF2HF3.._T._T._T...Q"
  },

  "physicians_active": {
    "source": "OECD.ELS.HD,DSD_HEALTH_EMP_REAC@DF_REAC,1.0/",
    "specs": "HSE.10P3HB...PHYS..P."
  },

  "nurses_active": {
    "source": "OECD.ELS.HD,DSD_HEALTH_EMP_REAC@DF_REAC,1.0/",
    "specs": "HSE.10P3HB...MINU..P."
  },

  "hospital_beds": {
    "source": "OECD.ELS.HD,DSD_HEALTH_REAC_HOSP@DF_BEDS_FUNC,1.0/",
    "specs": ".10P3HB...HC1._T.."
  }
}


countries = (
        "AUS+AUT+BEL+CAN+CHL+COL+CRI+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ISR+ITA+"
        "JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA"
    )


# ===============================
# PULL AND SAVE OUTCOME DATA
# ===============================

BASE_URL = "https://sdmx.oecd.org/public/rest/data/"

for outcome, config in OUTCOMES.items():

    source = config["source"]
    specs = config["specs"]

    key = f"{countries}.{specs}"
    url = BASE_URL + source + key

    print("Pulling:", outcome)

    response = requests.get(
        url,
        headers={
            "Accept": "text/csv",
            "User-Agent": "Mozilla/5.0"
        },
        timeout=30
    )

    if response.status_code != 200:
        print(f"{outcome} failed:", response.status_code)
        continue

    df = pd.read_csv(StringIO(response.text))
    df = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]]
    df.columns = ["Country", "Year", "Outcome"]

    df.to_csv(f"{outcome}.csv", index=False)
