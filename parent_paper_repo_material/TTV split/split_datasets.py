#!/usr/bin/env python3
"""
split_datasets.py

Performs a 70 / 15 / 15 (Train / Validation / Test) row-level split on every dataset
found in the `PP_data` directory, saving the resulting datasets into:
    parent_paper_repo_material/TTV split/train/
    parent_paper_repo_material/TTV split/validation/
    parent_paper_repo_material/TTV split/test/

Preserves the hierarchical subfolder structure within each split folder as well as
multi-sheet workbooks.
"""

import os
import sys
import copy
import argparse
from pathlib import Path
import openpyxl


def find_header_and_data_rows(sheet):
    """
    Identifies header rows (preamble/metadata/column titles) and data rows.
    Finds the row within the first 15 rows having the most populated columns,
    treating rows 1..best_header_row as header rows and subsequent rows as data rows.
    """
    max_r = sheet.max_row or 0
    max_c = sheet.max_column or 0
    if max_r == 0 or max_c == 0:
        return list(range(1, max_r + 1)), []

    best_r = 1
    max_non_empty = 0
    for r in range(1, min(15, max_r + 1)):
        vals = [sheet.cell(row=r, column=c).value for c in range(1, max_c + 1)]
        non_empty = sum(1 for v in vals if v is not None and str(v).strip() != "")
        if non_empty > max_non_empty:
            max_non_empty = non_empty
            best_r = r

    header_rows = list(range(1, best_r + 1))
    data_rows = list(range(best_r + 1, max_r + 1))
    return header_rows, data_rows


def compute_split_indices(data_rows, train_ratio=0.70, val_ratio=0.15):
    """
    Splits the data row indices sequentially into 70% train, 15% validation, and 15% test.
    """
    n = len(data_rows)
    if n == 0:
        return [], [], []

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Guarantee at least 1 in each partition if n >= 3
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))

    train_rows = data_rows[:n_train]
    val_rows = data_rows[n_train : n_train + n_val]
    test_rows = data_rows[n_train + n_val :]

    return train_rows, val_rows, test_rows


def copy_cells(src_sheet, dst_sheet, row_indices):
    """
    Copies cell values and basic formatting from src_sheet at row_indices to dst_sheet.
    """
    for new_r, orig_r in enumerate(row_indices, start=1):
        for c in range(1, src_sheet.max_column + 1):
            src_cell = src_sheet.cell(row=orig_r, column=c)
            dst_cell = dst_sheet.cell(row=new_r, column=c, value=src_cell.value)
            if src_cell.has_style:
                dst_cell.number_format = src_cell.number_format


def split_single_file(src_path: Path, train_path: Path, val_path: Path, test_path: Path):
    """
    Loads an Excel file and creates the corresponding Train, Validation, and Test workbooks.
    """
    wb_src = openpyxl.load_workbook(src_path, data_only=True)

    wb_train = openpyxl.Workbook()
    wb_train.remove(wb_train.active)  # remove default sheet

    wb_val = openpyxl.Workbook()
    wb_val.remove(wb_val.active)

    wb_test = openpyxl.Workbook()
    wb_test.remove(wb_test.active)

    for sheetname in wb_src.sheetnames:
        sheet = wb_src[sheetname]
        ws_train = wb_train.create_sheet(title=sheetname)
        ws_val = wb_val.create_sheet(title=sheetname)
        ws_test = wb_test.create_sheet(title=sheetname)

        header_rows, data_rows = find_header_and_data_rows(sheet)

        if not data_rows:
            # Preamble/reference only sheet (e.g. definitions or methodology)
            copy_cells(sheet, ws_train, header_rows)
            copy_cells(sheet, ws_val, header_rows)
            copy_cells(sheet, ws_test, header_rows)
        else:
            train_r, val_r, test_r = compute_split_indices(data_rows, 0.70, 0.15)
            copy_cells(sheet, ws_train, header_rows + train_r)
            copy_cells(sheet, ws_val, header_rows + val_r)
            copy_cells(sheet, ws_test, header_rows + test_r)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    wb_train.save(train_path)
    wb_val.save(val_path)
    wb_test.save(test_path)
    wb_src.close()


def process_all_datasets(pp_data_dir: Path, output_ttv_dir: Path):
    """
    Walks through `pp_data_dir`, finds all dataset files (.xlsx), and executes
    the 70/15/15 train/validation/test split into their respective directories.
    """
    train_dir = output_ttv_dir / "train"
    val_dir = output_ttv_dir / "validation"
    test_dir = output_ttv_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    excel_files = sorted(
        [
            p
            for p in pp_data_dir.rglob("*.xlsx")
            if not p.name.startswith("~$") and not p.name.startswith(".")
        ]
    )

    print(f"Found {len(excel_files)} datasets in '{pp_data_dir}'.")
    print(f"Destination: '{output_ttv_dir}'")
    print("Splitting datasets (70% train / 15% validation / 15% test)...")

    for i, file_path in enumerate(excel_files, start=1):
        rel_path = file_path.relative_to(pp_data_dir)
        target_train = train_dir / rel_path
        target_val = val_dir / rel_path
        target_test = test_dir / rel_path

        print(f"[{i}/{len(excel_files)}] Processing: {rel_path}")
        split_single_file(file_path, target_train, target_val, target_test)

    print(f"\nCompleted split for all {len(excel_files)} files successfully!")
    print(f"Train split saved to:      {train_dir}")
    print(f"Validation split saved to: {val_dir}")
    print(f"Test split saved to:       {test_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split all datasets in PP_data into 70/15/15 train/validation/test folders."
    )
    current_script_dir = Path(__file__).resolve().parent
    default_base_dir = current_script_dir.parent

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_base_dir / "PP_data",
        help="Path to the PP_data folder (default: parent_paper_repo_material/PP_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=current_script_dir,
        help="Path to the output TTV split directory (default: parent_paper_repo_material/TTV split)",
    )

    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"Error: Source directory '{args.source_dir}' does not exist.")
        sys.exit(1)

    process_all_datasets(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
