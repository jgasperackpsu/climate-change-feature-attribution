"""Optional step 0 for parent paper 3: download the panel from the World Bank API.
Munir, Rashid, Shams & Ahmed (2026), "Infrastructure, institutions, and the resilience dividend: A hybrid
Econometric-Machine learning analysis of CO2 emissions in Central and Eastern Europe",
Environmental Management: Economics and Policy 2, 100006.

Usage:
    python fetch_wdi.py [--out ../../PP_data/pp3_data/pp3_panel_raw.csv] [--start 2000] [--end 2024]

Writes one row per country-year with the raw (unfilled) series; prepare_pp3.py does the paper's
forward/back fill and the log transforms. You can skip this script and download the same series by hand from
https://databank.worldbank.org (WDI and WGI) into the same columns.

    column       WB code               source  paper name
    co2_pc       EN.GHG.CO2.PC.CE.AR5  WDI     CO2E  CO2 per capita excl. LULUCF (t CO2e/capita)
    up           SP.URB.TOTL.IN.ZS     WDI     UP    urban population (% of total)
    ic           GOV_WGI_GE.EST        WGI     IC    government effectiveness (estimate, -2.5..2.5)
    eupe         GOV_WGI_RQ.EST        WGI     EUPE  regulatory quality (estimate, -2.5..2.5)
    gdp_pc       NY.GDP.PCAP.KD        WDI     EC    GDP per capita, constant US$
    elec_kwh_pc  EG.USE.ELEC.KH.PC     WDI     IEI   electric power consumption (kWh per capita)
    eo           NE.TRD.GNFS.ZS        WDI     EO    trade (% of GDP)
    ce           GOV_WGI_VA.EST        WGI     CE    voice and accountability (estimate, -2.5..2.5)

The WGI codes gained a GOV_WGI_ prefix in the 2024 revision; the .EST series are still on the -2.5..2.5 scale
the paper uses. Older codes (GE.EST etc., and the retired CO2 code EN.ATM.CO2E.PC) are tried as fallbacks.
The script prints each series' last available year so gaps that the fill rule would cover are visible.
"""
import argparse
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

try:  # python.org macOS builds ship without root certificates; certifi's bundle avoids CERTIFICATE_VERIFY_FAILED
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COUNTRIES = {"ALB": "Albania", "BIH": "Bosnia and Herzegovina", "BGR": "Bulgaria", "HRV": "Croatia",
             "CZE": "Czech Republic", "EST": "Estonia", "HUN": "Hungary", "LVA": "Latvia", "LTU": "Lithuania",
             "MKD": "North Macedonia", "POL": "Poland", "ROU": "Romania", "SVK": "Slovakia", "SVN": "Slovenia",
             "MDA": "Moldova", "UKR": "Ukraine", "ARM": "Armenia", "GEO": "Georgia"}
SERIES = {  # column: (codes to try in order, WB API source id)
    "co2_pc": (["EN.GHG.CO2.PC.CE.AR5", "EN.ATM.CO2E.PC"], 2),
    "up": (["SP.URB.TOTL.IN.ZS"], 2),
    "ic": (["GOV_WGI_GE.EST", "GE.EST"], 3),
    "eupe": (["GOV_WGI_RQ.EST", "RQ.EST"], 3),
    "gdp_pc": (["NY.GDP.PCAP.KD"], 2),
    "elec_kwh_pc": (["EG.USE.ELEC.KH.PC"], 2),
    "eo": (["NE.TRD.GNFS.ZS"], 2),
    "ce": (["GOV_WGI_VA.EST", "VA.EST"], 3),
}
API = "https://api.worldbank.org/v2/country/{c}/indicator/{i}?source={s}&date={a}:{b}&format=json&per_page=20000"


def get(code, source, start, end):
    url = API.format(c=";".join(COUNTRIES), i=code, s=source, a=start, b=end)
    req = urllib.request.Request(url, headers={"User-Agent": "ds340w-replication"})
    try:
        with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as r:
            payload = json.loads(r.read())
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise SystemExit("SSL certificate check failed. Run `pip install certifi`, or on a python.org macOS "
                             "install run '/Applications/Python 3.x/Install Certificates.command'.") from e
        raise
    if len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["iso3", "year", "value"])
    return pd.DataFrame([dict(iso3=o["countryiso3code"], year=int(o["date"]), value=o["value"])
                         for o in payload[1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "PP_data" / "pp3_data" / "pp3_panel_raw.csv")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2024)
    a = ap.parse_args()

    grid = pd.MultiIndex.from_product([list(COUNTRIES), range(a.start, a.end + 1)], names=["iso3", "year"])
    panel = pd.DataFrame(index=grid)
    for col, (codes, source) in SERIES.items():
        for code in codes:
            df = get(code, source, a.start, a.end).dropna(subset=["value"])
            time.sleep(0.5)
            if len(df):
                panel[col] = df.set_index(["iso3", "year"]).value
                cov = panel[col].dropna().reset_index().groupby("iso3").year.max()
                print(f"{col:<12} {code:<22} {len(df):>4} values; last year by country: "
                      f"min {cov.min()}, max {cov.max()}")
                break
            print(f"{col:<12} {code:<22} no data, trying next code")
    panel = panel.reset_index()
    panel.insert(0, "country", panel.iso3.map(COUNTRIES))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(panel)} rows; missing values per column:\n{panel.isna().sum().to_string()}")


if __name__ == "__main__":
    main()
