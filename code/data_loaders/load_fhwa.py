"""Step 2: turn FHWA Highway Statistics vehicle-miles tables into data/interim/fhwa.csv.

Reads one file per year from a folder, named vm2_YYYY.<ext>:
  - .xls / .xlsx : table VM-2 (2007 onward; layout varies by year)
  - .pdf         : table PS-1 "Selected measures for identifying peer states" (e.g. 1997),
                   which carries the same rural and urban annual VMT columns

Usage:
    python src/load/load_fhwa.py <folder> <path/to/state_crosswalk.csv> <out_dir>

Output is tidy: state_fips, year, variable, value. Units: millions of vehicle-miles.
Needs: pandas, xlrd (old .xls), openpyxl (.xlsx), pdfplumber (.pdf).
"""
import re
import sys
from pathlib import Path

import pandas as pd


def clean_state(s):
    """'Indiana  2/' -> 'Indiana', 'Oklahoma ' -> 'Oklahoma'."""
    s = re.sub(r"\(.*?\)|\d+/", "", str(s))
    return re.sub(r"\s+", " ", s).strip()


def read_vm2(path, states):
    xl = pd.ExcelFile(path)
    # newer files carry a hidden 'CRYSTAL_PERSIST' sheet; the table is on sheet 'A'
    sheet = "A" if "A" in xl.sheet_names else xl.sheet_names[0]
    raw = pd.read_excel(xl, sheet_name=sheet, header=None)

    # some years (e.g. 2000) were published as table PS-1, not VM-2
    head = " ".join(str(v) for v in raw.head(15).values.ravel() if pd.notna(v)).upper()
    if "PS-1" in head:
        return read_ps1_sheet(raw, path, states)

    # header row = the one whose first cell is 'STATE'
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.upper() == "STATE"][0]
    labels = raw.iloc[hdr].astype(str).str.strip().str.upper()
    total_cols = [c for c, v in labels.items() if v == "TOTAL"]
    if len(total_cols) != 3:
        raise ValueError(f"{path.name}: expected 3 TOTAL columns, found {len(total_cols)}; "
                         "this year's layout is new, so send the file to update the loader")
    rural, urban, total = total_cols

    body = raw.iloc[hdr + 1:].copy()
    body["state"] = body.iloc[:, 0].map(clean_state)
    body = body[body["state"].isin(states)]
    out = pd.DataFrame({
        "state_name": body["state"],
        "vmt_total_millions": pd.to_numeric(body[total], errors="coerce"),
        "vmt_rural_millions": pd.to_numeric(body[rural], errors="coerce"),
        "vmt_urban_millions": pd.to_numeric(body[urban], errors="coerce"),
    })
    # sanity check: rural + urban should equal the total
    gap = (out.vmt_rural_millions + out.vmt_urban_millions - out.vmt_total_millions).abs().max()
    if gap > 1:
        print(f"  warning {path.name}: rural+urban differs from total by up to {gap:.1f}")
    return out


def ps1_rows_to_frame(rows, path):
    """rows = [(state_name, [token, ...]), ...] from sheet 1 of table PS-1.
    The travel columns sit at the end of each row: rural VMT, % trucks, urban VMT, % trucks,
    VMT per capita, lane-miles rural, lane-miles urban. Reading from the end keeps this working
    if the general-measures columns shift between years."""
    def num(tok):
        try:
            return float(str(tok).replace(",", ""))
        except ValueError:          # '-' or blank
            return float("nan")

    out = []
    for name, toks in rows:
        if len(toks) < 7:
            raise ValueError(f"{path.name}: could not parse the row for {name}")
        rural, urban = num(toks[-7]), num(toks[-5])
        out.append({"state_name": name,
                    "vmt_total_millions": rural + urban,
                    "vmt_rural_millions": rural,
                    "vmt_urban_millions": urban,
                    "_pop_thousands": num(toks[3]) + num(toks[4]),
                    "_vmt_per_capita": num(toks[-3])})
    out = pd.DataFrame(out)

    # sanity check: total VMT / population should match the table's own per-capita column
    implied = out.vmt_total_millions * 1e6 / (out._pop_thousands * 1e3)
    off = (implied / out._vmt_per_capita - 1).abs()
    if (off > 0.01).any():
        print(f"  warning {path.name}: per-capita check off by >1% for "
              f"{sorted(out.loc[off > 0.01, 'state_name'])}; check the column positions")
    return out.drop(columns=["_pop_thousands", "_vmt_per_capita"])


def read_ps1(path, states):
    """Table PS-1 as a PDF (e.g. 1997, 1998)."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text()  # sheet 1 of 2 holds VMT; sheet 2 is state-agency roads
    if "VEHICLE-MILES" not in text.upper():
        raise ValueError(f"{path.name}: page 1 has no vehicle-miles columns; is this table PS-1?")

    rows = []
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z. ]+?(?:\s+\d+/)?)\s+(-?[\d,.]+|-)\s", line)
        if not m:
            continue
        name = clean_state(m.group(1))
        if name in states:
            rows.append((name, line[len(m.group(1)):].split()))
    return ps1_rows_to_frame(rows, path)


def read_ps1_sheet(raw, path, states):
    """Table PS-1 as a spreadsheet (e.g. 2000-2006). Columns are located from the header row, not
    counted from the end, because some years (e.g. 2003) carry worksheet notes to the right of
    the table and a second block of state rows below it."""
    first = raw.iloc[:, 0].map(lambda v: str(v).strip().upper() if pd.notna(v) else "")
    # header row whose cells include 'ANNUAL' twice: rural VMT, then urban VMT
    hdr = None
    for i in range(min(len(raw), 30)):
        cells = [str(v).strip().upper() if pd.notna(v) else "" for v in raw.iloc[i]]
        ann = [c for c, v in enumerate(cells) if v == "ANNUAL"]
        if len(ann) >= 2:
            hdr, (a_rural, a_urban) = i, ann[:2]
            break
    if hdr is None:
        raise ValueError(f"{path.name}: no 'ANNUAL' VMT headings found; send this file to update the loader")
    last = a_urban + 4                      # urban lane-miles, the table's last column

    rows, seen = [], set()
    for i in range(hdr + 1, len(raw)):
        if first.iat[i].startswith(("U.S. TOTAL", "GRAND TOTAL")):
            break                           # end of the table; anything below is worksheet notes
        name = clean_state(raw.iat[i, 0])
        if name in states and name not in seen:
            seen.add(name)
            rows.append((name, raw.iloc[i, 1:last + 1].tolist()))
    return ps1_rows_to_frame(rows, path)


def read_any(path, states):
    if path.suffix.lower() == ".pdf":
        return read_ps1(path, states)
    return read_vm2(path, states)


def main(folder, crosswalk_path, out_dir):
    xw = pd.read_csv(crosswalk_path, dtype=str)
    name_to_fips = dict(zip(xw["state_name"], xw["fips_code"]))

    frames = []
    paths = [p for p in sorted(Path(folder).glob("vm2_*")) if p.suffix.lower() in (".xls", ".xlsx", ".pdf")]
    for path in paths:
        m = re.search(r"(19|20)\d{2}", path.name)
        if not m:
            raise ValueError(f"{path.name} has no year in its name; rename it to vm2_YYYY{path.suffix}")
        year = int(m.group(0))
        df = read_any(path, set(name_to_fips))
        df["year"] = year
        frames.append(df)
        print(f"{path.name}: {len(df)} states")
        missing = set(name_to_fips) - set(df.state_name)
        if missing:
            print(f"  missing: {sorted(missing)}")

    w = pd.concat(frames, ignore_index=True)
    dup = w.duplicated(["state_name", "year"])
    if dup.any():
        raise ValueError(f"two files for the same year: {sorted(w.loc[dup, 'year'].unique())}")
    w["urban_vmt_share"] = w["vmt_urban_millions"] / w["vmt_total_millions"]
    tidy = w.melt(id_vars=["state_name", "year"], var_name="variable", value_name="value")
    tidy.insert(0, "state_fips", tidy.pop("state_name").map(name_to_fips))
    tidy = tidy.sort_values(["state_fips", "year", "variable"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "fhwa.csv", index=False)
    print(f"\nwrote {len(tidy):,} rows, years {w.year.min()}-{w.year.max()}, to {out / 'fhwa.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
