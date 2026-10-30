import pandas as pd
import os
import re
from datetime import datetime

curr_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(curr_dir, "sales_data.xlsx")

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

xls = pd.ExcelFile(path)
all_data = []

for sheet_name in xls.sheet_names:
    year = int(sheet_name)

    # Read with no header so we control parsing; row 0 is the header row
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # Row 0 contains column labels; extract them
    raw_headers = df.iloc[0].tolist()

    # Identify the id columns (first two: NAICS Code, Kind of Business)
    # and the month columns (everything after)
    id_col_count = 2
    month_headers_raw = raw_headers[id_col_count:]

    # Parse each month header into a (year, month) tuple
    month_labels = []
    for h in month_headers_raw:
        if isinstance(h, datetime):
            month_labels.append((h.year, h.month))
        elif isinstance(h, str):
            # Patterns like "Jan. 2025", "Jul. 2026(p)", "Aug. 2026(a)"
            m = re.match(r"([A-Za-z]{3})\.?\s*(\d{4})", h)
            if m:
                mon_str, yr_str = m.group(1), m.group(2)
                month_labels.append((int(yr_str), MONTH_MAP[mon_str]))
            else:
                month_labels.append(None)
        else:
            month_labels.append(None)

    # Data rows start at row 1 (skip header row 0); we need to detect sections
    data = df.iloc[1:].copy()
    data.columns = range(data.shape[1])
    data = data.rename(columns={0: "NAICS_Code", 1: "Kind_of_Business"})

    # Tag each row with its adjustment section by scanning column 1
    # for section labels like "NOT ADJUSTED" or "ADJUSTED(2)"
    current_section = "Not Adjusted"  # default for rows before any label
    section_labels = []
    section_marker_rows = []

    for idx, row in data.iterrows():
        biz = row["Kind_of_Business"]
        if isinstance(biz, str):
            biz_upper = biz.strip().upper()
            if biz_upper.startswith("NOT ADJUSTED"):
                current_section = "Not Adjusted"
                section_marker_rows.append(idx)
            elif biz_upper.startswith("ADJUSTED"):
                current_section = "Adjusted"
                section_marker_rows.append(idx)
        section_labels.append(current_section)

    data["Adjustment"] = section_labels

    # Drop section marker rows and blank/NaN Kind_of_Business rows
    data = data.drop(index=section_marker_rows, errors="ignore")
    data = data.dropna(subset=["Kind_of_Business"])
    # Also drop rows where Kind_of_Business is blank or NaN-like
    data = data[data["Kind_of_Business"].astype(str).str.strip() != ""]
    data = data[data["Kind_of_Business"].astype(str).str.strip().str.lower() != "nan"]

    # Iterate over each month column and build long-form rows
    for col_idx, label in enumerate(month_labels, start=id_col_count):
        if label is None:
            continue
        yr, mo = label
        month_slice = data[["NAICS_Code", "Kind_of_Business", "Adjustment", col_idx]].copy()
        month_slice = month_slice.rename(columns={col_idx: "Sales_Amount"})
        month_slice["Year"] = yr
        month_slice["Month"] = mo
        all_data.append(month_slice)

# Combine into a single master dataframe
combined_df = pd.concat(all_data, ignore_index=True)

# Clean up
combined_df.dropna(subset=["Kind_of_Business", "Sales_Amount"], inplace=True)

# Strip whitespace from string columns
combined_df["NAICS_Code"] = combined_df["NAICS_Code"].astype(str).str.strip()
combined_df["Kind_of_Business"] = combined_df["Kind_of_Business"].astype(str).str.strip()

# Remove rows where Sales_Amount is not numeric (e.g. "(*)" suppressed values)
combined_df["Sales_Amount"] = pd.to_numeric(combined_df["Sales_Amount"], errors="coerce")
combined_df.dropna(subset=["Sales_Amount"], inplace=True)
combined_df["Sales_Amount"] = combined_df["Sales_Amount"].astype(int)

# ---------------------------------------------------------------------------
# Pivot: Make each Kind_of_Business into a column with Sales_Amount as values
# ---------------------------------------------------------------------------
pivoted_df = combined_df.pivot_table(
    index=["Year", "Month", "Adjustment"],
    columns="Kind_of_Business",
    values="Sales_Amount",
    aggfunc="first",  # one value per combination
)

# Flatten the multi-level column index
pivoted_df.columns.name = None
pivoted_df.reset_index(inplace=True)

# Sort for readability
pivoted_df.sort_values(by=["Adjustment", "Year", "Month"], inplace=True)
pivoted_df.reset_index(drop=True, inplace=True)

# Save to CSV in the same directory
output_path = os.path.join(curr_dir, "combined_retail_sales_2007_2026.csv")
pivoted_df.to_csv(output_path, index=False)

print(f"Saved {len(pivoted_df)} rows x {len(pivoted_df.columns)} columns to {output_path}")
print(f"\nColumns: {list(pivoted_df.columns)}")
print(f"\nYear range: {pivoted_df['Year'].min()} – {pivoted_df['Year'].max()}")
print(f"Adjustment values: {pivoted_df['Adjustment'].unique().tolist()}")
print(f"\nFirst 5 rows (truncated):")
print(pivoted_df.head().to_string(max_cols=8))