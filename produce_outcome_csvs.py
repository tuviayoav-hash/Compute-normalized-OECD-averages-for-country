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
  }

}


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

    df.to_csv(f"{outcome}.csv", index=False)
