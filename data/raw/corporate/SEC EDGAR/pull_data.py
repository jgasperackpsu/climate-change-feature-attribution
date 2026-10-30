import json
import os
import pandas as pd
import requests

# Get the current working directory (same folder as this python file)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Saving files to local directory: {current_dir}")

# SEC requires a custom User-Agent header with your name and contact email:
sec_headers = {"User-Agent": "Sample Name admin@sample.com"}

# Example: ExxonMobil CIK (0000034088). CIK must be 10 digits zero-padded.
cik = "0000034088"
sec_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"

print("Fetching SEC EDGAR Company Facts data...")
sec_response = requests.get(sec_url, headers=sec_headers)

if sec_response.status_code == 200:
  sec_data = sec_response.json()
  sec_path = os.path.join(current_dir, f"sec_company_facts_{cik}.json")

  with open(sec_path, "w") as f:
    json.dump(sec_data, f, indent=4)
  print(f"-> Saved SEC Company Facts JSON to: {sec_path}")

  # Optional: Extracting CapEx tags (e.g., PaymentsToAcquirePropertyPlantAndEquipment)
  try:
    us_gaap = sec_data["facts"]["us-gaap"]
    capex_key = "PaymentsToAcquirePropertyPlantAndEquipment"
    if capex_key in us_gaap:
      units = us_gaap[capex_key]["units"]["USD"]
      df_capex = pd.DataFrame(units)
      capex_csv_path = os.path.join(current_dir, f"sec_capex_{cik}.csv")
      df_capex.to_csv(capex_csv_path, index=False)
      print(f"-> Extracted and saved CapEx history to: {capex_csv_path}")
  except Exception as e:
    print(f"-> Note: Could not automatically parse CapEx tag: {e}")
else:
  print(
      f"-> SEC API Error ({sec_response.status_code}): Check your User-Agent"
      " header string."
  )