"""
Module containing custom functions exclusively used by the program.
"""

import pyarrow.parquet as pq
from pathlib import Path
import pyarrow as pa
import pandas as pd
import logging

# SET UP LOGGING
logger = logging.getLogger(__name__)

def parse_object(cls,obj:dict):
    """
    Base method used to parse dicts recieved from Upstox into relevent dataclass
    """
    from dataclasses import fields
    
    required_fields = {f.name for f in fields(cls)}
    filtered_dict = {k: v for k, v in obj.items() if k in required_fields}
    return cls(**filtered_dict)



def save_parquet(df: pd.DataFrame, path: Path, **kwargs):
    """
    Saves a DataFrame to a Parquet file with metadata. The metadata is passed as keyword arguments and stored in the Parquet file's schema metadata.
    """
    try:
        metadata_bytes = {
            key.encode("utf-8"): str(value).encode("utf-8")
            for key, value in kwargs.items()
        }
        table = pa.Table.from_pandas(df)
        existing_metadata = table.schema.metadata or {}
        final_metadata = {**existing_metadata, **metadata_bytes}
        table = table.replace_schema_metadata(final_metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
    except Exception as e:
        logger.exception(f"Error saving parquet file at {path}: {e}")


def load_parquet(path: Path):
    """
    Loads a DataFrame and metadata from a Parquet file. The metadata is returned as a dictionary with string keys and values.
    If there's an error during loading, it logs the exception and returns None for both data and metadata.
    """
    try:
        table = pq.read_table(path)
        df = table.to_pandas()
        metadata = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in (table.schema.metadata or {}).items()
        }
        return dict(data=df, metadata=metadata)
    except Exception as e:
        logger.exception(f"Error loading parquet file from {path}: {e}")
        return dict(data=None, metadata=None)


def to_ist(target: pd.Series | list | int | float, unit="ms"):
    """
    Converts UNIX timestamps (ms) to strict naive IST objects.
    Safely handles scalars, lists, and Pandas Series.
    """
    target = pd.to_numeric(target, errors="coerce")
    try:
        parsed = pd.to_datetime(target, unit="ms", utc=True)
    except ValueError:
        parsed = pd.to_datetime(target, utc=True)

    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    return parsed.tz_convert("Asia/Kolkata").tz_localize(None)


def generate_tear_sheet(report_df: pd.DataFrame, starting_capital: float = 300000.0):
    import quantstats as qs

    """
    Converts a trade log DataFrame into a QuantStats HTML report.
    """
    # 1. Ensure Timestamp is a proper datetime object and set it as the index
    # (Skip this if your Timestamp is already the index)
    df = report_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)

    # 2. Group the PnL by Day (Resample to 'D')
    # If you made 5 trades on Monday, this sums them into one daily PnL number.
    daily_pnl = df["PnL"].resample("D").sum().fillna(0)

    # 3. Build the Equity Curve
    # Add the compounding daily PnL to your starting cash
    equity_curve = starting_capital + daily_pnl.cumsum()

    # 4. Calculate Percentage Returns
    # This is the exact format QuantStats requires: e.g., 0.015 for +1.5%
    daily_returns = equity_curve.pct_change().fillna(0)

    # 5. Generate the HTML Report
    # Note: 'quantstats' needs the timezone removed to calculate standard benchmarks
    daily_returns.index = daily_returns.index.tz_localize(None)

    print("Generating QuantStats Tear Sheet...")
    qs.reports.html(
        daily_returns, title="Options Strategy Backtest", output="backtest_report.html"
    )
    print("✅ Report saved as 'backtest_report.html'. Open it in your web browser!")


def push_report_to_sheets(report_df: pd.DataFrame, sheet_url: str):
    import gspread
    from gspread_dataframe import set_with_dataframe

    """
    Pushes the backtest Trade Report DataFrame directly to a live Google Sheet.
    """
    print("Authenticating with Google Cloud...")
    # Point this to your downloaded JSON key
    gc = gspread.service_account(filename="google_secret.json")

    # Open the specific Google Sheet using its URL
    spreadsheet = gc.open_by_url(sheet_url)
    worksheet = spreadsheet.sheet1

    # Optional: Format the Timestamp so Google Sheets reads it cleanly
    export_df = report_df.copy()
    export_df["Timestamp"] = export_df["Timestamp"].astype(str)

    print("Uploading data to Google Sheets...")
    # Clear out the old backtest data
    worksheet.clear()
    # Paste the new DataFrame starting at cell A1
    set_with_dataframe(worksheet, export_df)

    print("✅ Trade report successfully pushed to Google Sheets!")

