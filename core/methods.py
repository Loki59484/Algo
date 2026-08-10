"""
Module containing custom functions exclusively used by the program.
"""

import pyarrow.parquet as pq
from typing import Literal
from pathlib import Path
import pyarrow as pa
import pandas as pd
import logging
import hashlib
import json
import datetime
import zmq
import time
import threading

# SET UP LOGGING
logger = logging.getLogger(__name__)

import numpy as np
from requests import post
from os import environ
TELEGRAM_TOKEN = environ.get("TELEGRAM_TOKEN")
CHAT_ID = environ.get("TELEGRAM_CHAT_ID")

import pyotp
import time

def generate_setup_code(secret_key: str):
    """
    Generates the current TOTP code and shows the time remaining 
    before it refreshes.
    """
    # Initialize the TOTP object with your Upstox secret
    totp = pyotp.TOTP(secret_key)
    
    # Get the active 6-digit code
    current_code = totp.now()
    
    # TOTP codes refresh every 30 seconds. 
    # This calculates how many seconds are left in the current window.
    time_remaining = 30 - (int(time.time()) % 30)
    
    print("\n=== Upstox TOTP Setup ===")
    print(f"Current Code: {current_code}")
    print(f"Time Remaining: {time_remaining} seconds")
    print("=========================\n")
    
    return current_code

def send_telegram_update(message):
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = post(url, json=payload)
    return response.status_code == 200

def calculate_trade_charges(buy_price, sell_price, qty, instrument:Literal['EQ','F','O']="O", trade_type: Literal['I', 'D']="D", return_breakdown=False):
    """
    Calculates charges based on Instrument (equity/futures/options) and Trade Type (intraday/delivery).
    """
    trade_map = {'I':'intraday','D':'delivery'}
    instrument_map = {'EQ':'equity','F':'futures','O':'options'}
    if trade_type in trade_map:
        trade_type = trade_map[trade_type]
    if instrument in instrument_map:
        instrument = instrument_map[instrument]
        
    buy_value = buy_price * qty
    sell_value = sell_price * qty
    total_value = buy_value + sell_value

    # ---------------------------------------------------------
    # 1. BROKERAGE
    # ---------------------------------------------------------
    if instrument == 'options':
        # Options is flat ₹20 regardless of intraday or overnight
        brokerage = 40.0 # 20 buy + 20 sell
    elif instrument == 'futures':
        # Futures is ₹20 or 0.05%, regardless of intraday or overnight
        brokerage = min(20.0, buy_value * 0.0005) + min(20.0, sell_value * 0.0005)
    elif instrument == 'equity':
        if trade_type == 'delivery':
            brokerage = 40.0 # Flat ₹20 per executed order
        else: # intraday
            brokerage = min(20.0, buy_value * 0.001) + min(20.0, sell_value * 0.001)
    else:
        raise ValueError("Instrument must be 'equity', 'futures', or 'options'.")

    # ---------------------------------------------------------
    # 2. STT (Securities Transaction Tax)
    # ---------------------------------------------------------
    if instrument == 'options':
        stt = np.round(sell_value * 0.001)
    elif instrument == 'futures':
        stt = np.round(sell_value * 0.0002)
    elif instrument == 'equity':
        if trade_type == 'delivery':
            stt = np.round(buy_value * 0.001) + np.round(sell_value * 0.001)
        else: # intraday
            stt = np.round(sell_value * 0.00025)

    # ---------------------------------------------------------
    # 3. EXCHANGE TRANSACTION CHARGES (NSE)
    # ---------------------------------------------------------
    if instrument == 'options':
        txn_charge = total_value * 0.000495   # 0.0495%
    elif instrument == 'futures':
        txn_charge = total_value * 0.0000188  # 0.00188%
    elif instrument == 'equity':
        txn_charge = total_value * 0.0000345  # 0.00345%

    # ---------------------------------------------------------
    # 4. SEBI CHARGES (Universal)
    # ---------------------------------------------------------
    sebi_charge = total_value * 0.000001 # ₹10 per crore

    # ---------------------------------------------------------
    # 5. STAMP DUTY (Charged on Buy Side Only)
    # ---------------------------------------------------------
    if instrument == 'options':
        stamp_duty = np.round(buy_value * 0.00003)
    elif instrument == 'futures':
        stamp_duty = np.round(buy_value * 0.00002)
    elif instrument == 'equity':
        if trade_type == 'delivery':
            stamp_duty = np.round(buy_value * 0.00015)
        else: # intraday
            stamp_duty = np.round(buy_value * 0.00003)

    # ---------------------------------------------------------
    # 6. GST (18% on Brokerage + Txn Charges + SEBI)
    # ---------------------------------------------------------
    gst = (brokerage + txn_charge + sebi_charge) * 0.18

    # TOTAL CALCULATION
    total_charges = brokerage + stt + txn_charge + sebi_charge + stamp_duty + gst
    
    if return_breakdown:
        return {
            "Brokerage": round(brokerage, 2),
            "STT": round(stt, 2),
            "Transaction Charge": round(txn_charge, 2),
            "SEBI Charge": round(sebi_charge, 2),
            "Stamp Duty": round(stamp_duty, 2),
            "GST": round(gst, 2),
            "Total": round(total_charges, 2)
        }
        
    return round(total_charges, 2)

class ZMQErrorLogger(logging.Handler):
    """Intercepts ERROR level logs and broadcasts them over ZMQ."""
    def __init__(self, component_name: str, port: int):
        super().__init__()
        self.component_name = component_name
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        
        # FIX: Keep messages in memory for up to 2 seconds if the script crashes/exits
        self.socket.setsockopt(zmq.LINGER, 5000)
        # FIX: Buffer up to 1000 messages in case of a rapid burst of errors
        self.socket.setsockopt(zmq.SNDHWM, 1000)
        
        self.socket.bind(f"tcp://0.0.0.0:{port}")

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            try:
                error_msg = self.format(record)
                self.socket.send_string(f"ERROR:{self.component_name}:{error_msg}")
                logger.info(f"Error msg sent : {error_msg}")
            except Exception:
                self.handleError(record)

def start_heartbeat(component_name: str, port: int):
    """Starts a background thread that broadcasts a ping every second."""
    def ping_loop():
        context = zmq.Context().instance()
        socket = context.socket(zmq.PUB)
        
        # BIND TO 0.0.0.0 (All interfaces, including Tailscale)
        socket.bind(f"tcp://0.0.0.0:{port}")
        
        while True:
            socket.send_string(f"PING:{component_name}")
            time.sleep(1) 

    thread = threading.Thread(target=ping_loop, daemon=True)
    thread.start()
    return thread


def generate_cache_key(date_str, params):
    """
    Creates a unique filename based on the exact parameters used.
    If you change a parameter, the hash changes, and the script builds a new matrix.
    """
    # Create a string representation of your exact current rules
    param_string = json.dumps(params, sort_keys=True)

    # Hash it to keep the filename short and clean
    param_hash = hashlib.md5(param_string.encode()).hexdigest()[:8]

    return f"{date_str}_v_{param_hash}"


def filter_options(options: pd.DataFrame):
    """Filter options based on strike price and option type."""
    calls = [
        (
            option["instrument_key"],
            option["market_data"]["oi"],
            option["market_data"]["ltp"],
        )
        for option in options.call_options
        if option["market_data"]["oi"] > 0
    ]
    sorted_calls = sorted(calls, key=lambda x: x[1], reverse=True)
    puts = [
        (
            option["instrument_key"],
            option["market_data"]["oi"],
            option["market_data"]["ltp"],
        )
        for option in options.put_options
        if option["market_data"]["oi"] > 0
    ]
    sorted_puts = sorted(puts, key=lambda x: x[1], reverse=True)
    return sorted_calls[5], sorted_puts[5]


def parse_obj_to_dataclass(cls, obj: dict):
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

        return {"data": df, "metadata": metadata}
    except Exception as e:
        logger.exception(f"Error loading parquet file from {path}: {e}")
        return {"data": None, "metadata": None}


def to_ist(target: pd.Series | list | int | float, unit="ms"):
    """
    Converts UNIX timestamps (ms) to strict naive IST objects.
    Safely handles scalars, lists, and Pandas Series.
    """
    is_scalar = not isinstance(target, (pd.Series, list, tuple))

    if is_scalar:
        target = pd.Series([target])
    elif isinstance(target, (list, tuple)):
        target = pd.Series(target)

    try:
        numeric_target = pd.to_numeric(target, errors="raise")
        parsed = pd.to_datetime(numeric_target, unit=unit, utc=True)

    except (ValueError, TypeError):
        parsed = pd.to_datetime(target, utc=True)

    ist_parsed = parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    if is_scalar:
        return ist_parsed.iloc[0]

    return ist_parsed


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

    # 2. Group the pnl by Day (Resample to 'D')
    # If you made 5 trades on Monday, this sums them into one daily pnl number.
    daily_pnl = df["pnl"].resample("D").sum().fillna(0)

    # 3. Build the Equity Curve
    # Add the compounding daily pnl to your starting cash
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


def setup_cli():
    """
    Function to accept arguInitiates engine in a chronological tickwise mode for the given FILEments from cli for setting up type of engine [Live/Simulation], ui [TUI/GUI/HEADLESS] and
    the files or directories with files to be simulated

    Returns:
        Namespace
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch Trading or Simulation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    liveparser = subparsers.add_parser("live", help="Start live trading engine")
    simparser = subparsers.add_parser(
        "sim", help="Start Trading engine in simulation mode"
    )
    simparser.add_argument(
        "-idx", "--index", required=False, default="NSE_INDEX|Nifty 50"
    )
    liveparser.add_argument(
        "-idx", "--index", required=False, default="NSE_INDEX|Nifty 50"
    )
    mode_group = simparser.add_mutually_exclusive_group(required=False)
    
    simparser.add_argument(
    "--date", 
    type=datetime.date.fromisoformat, 
    help="Date in YYYY-MM-DD format"
)
    mode_group.add_argument(
        "-b",
        "--bulk",
        nargs="+",
        metavar="PATH",
        help="(Default Mode) Initiates engine in bulk simulation mode for given FILES or files inside the given DIRECTORY.",
    )
    simparser.add_argument(
        "-nc",
        "--no-cache",
        action="store_true",
        help="Forces the engine to ignore cached data.",
    )
    mode_group.add_argument(
        "-t",
        "--tickwise",
        type=str,
        metavar="FILE",
        help="Initiates engine in a chronological tickwise mode for the given FILE.",
    )
    simparser.add_argument(
        "-pe",
        "--put",
        type=str,
        metavar="FILE",
        help="Stores put data filepath.",
    )
    simparser.add_argument(
        "-ce",
        "--call",
        type=str,
        metavar="FILE",
        help="Stores call data filepath.",
    )
    simparser.add_argument(
        "-s",
        "--spot",
        type=str,
        metavar="FILE",
        help="Stores spot data filepath.",
    )

    ui_group = parser.add_mutually_exclusive_group(required=False)
    ui_group.add_argument(
        "--gui", action="store_true", help="Lauch engine with Graphical User Interface"
    )
    ui_group.add_argument(
        "--tui", action="store_true", help="Lauch engine with Terminal User Interface"
    )
    ui_group.add_argument(
        "--headless",
        action="store_true",
        help="Lauch engine in headless mode (DEFAULT)",
    )

    return parser.parse_args()
