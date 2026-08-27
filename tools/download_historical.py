from pathlib import Path
from tqdm import tqdm
import datetime as dt
import pandas as pd
import argparse
import logging
import sys
import os

# Adding root directory to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT_DIR / "core"
HIST_DATA_DIR = ROOT_DIR / "data" / "historical"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.upstox_methods import UpstoxClient, DATA_DIR, CONFIG_DIR
from core.methods import save_parquet
from core.datatypes import to_ist

# Initiating logger
logger = logging.getLogger(__name__)
ustox = UpstoxClient()

def historical(key, from_date, to_date, isexpired=False, expiry=None):
    if isexpired and expiry is None and "INDEX" not in key:
        raise ValueError(f"Must provide the expiry date for expired instrument {key}")

    if isexpired:
        # OPTIMIZED: Use 'expired_key' directly to bypass redundant resolution
        history = ustox.get_historical(
            dtype="historical",
            expired_key=key, 
            to_date=to_date,
            from_date=from_date
        )
    else:
        history = ustox.get_historical(
            dtype="intraday", 
            instrument_key=key, 
            from_date=from_date, 
            to_date=to_date
        )
        if history is None or history.empty:
            history = ustox.get_historical(
                dtype="historical",
                instrument_key=key,
                from_date=from_date,
                to_date=to_date,
            )

    return history if history is not None else pd.DataFrame()

def is_cached(target_dir: Path, date: dt.date, key: str, interval: int, unit: str, exchange: str) -> bool:
    """Checks if a specific instrument's parquet file already exists for a given date."""
    file_path = (
        target_dir
        / exchange
        / date.strftime("%Y")
        / date.strftime("%m")
        / date.strftime("%d")
        / f"{interval}_{unit}"
        / f"{key}.parquet"
    )
    return file_path.exists()

def save_datewise(data: pd.DataFrame, target_dir: Path, force: bool = False, **metadata):
    if data.empty:
        return
        
    data["date"] = to_ist(data["timestamp"]).dt.date
    grouped = data.groupby("date")
    
    for date, group in grouped:
        key = metadata.get("instrument_key", None)
        exchange = f"{metadata.get('exchange', key[:3])}"
        
        file_path = (
            target_dir
            / exchange
            / date.strftime("%Y")
            / date.strftime("%m")
            / date.strftime("%d")
            / f"{metadata.get('interval', '1')}_{metadata.get('unit', 'minutes')}"
            / f"{key}.parquet"
        )

        if file_path.exists() and not force:
            continue

        clean_group = group.drop(columns=["date"])
        save_parquet(clean_group, file_path, date=date, **metadata)


def download_data(spot, is_expired=False, interval=1, unit="minutes", force=False):
    exchange_prefix = spot[:3]
    EXPIRED_KEYS_FILE = DATA_DIR / f"{exchange_prefix}_expired_keys.parquet"

    # Fetch available expiries
    expiries = ustox.get_options_with_expiry(options=spot, is_expired=is_expired)
    if not expiries:
        logger.error(f"No expiries found for {spot}.")
        return

    # Cache expired keys to avoid redundant API lookups
    if EXPIRED_KEYS_FILE.exists() and os.path.getsize(EXPIRED_KEYS_FILE) != 0 and not force:
        instruments = pd.read_parquet(EXPIRED_KEYS_FILE)
    else:
        instruments_list = []
        for date_str in tqdm(expiries, desc="Caching Expired Instruments"):
            data = ustox.get_expired_instruments(expiry_date=date_str, underlying=spot)
            if data is not None and not data.empty:
                instruments_list.append(data)
        
        if not instruments_list:
            logger.error("No expired instruments found across any expiry.")
            return
            
        instruments = pd.concat(instruments_list, ignore_index=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        instruments.to_parquet(EXPIRED_KEYS_FILE, index=False, engine="pyarrow")

    # Step Configuration for Bracket Building
    step = 50 if "NSE" in exchange_prefix else 100
    kwargs = {"interval": interval, "unit": unit}

    for exp_str in tqdm(expiries, desc="Downloading Option Data", position=0, leave=True):
        # 1. Holiday Adjustment: Shift expiry to the last valid trading day
        exp_dt = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        while ustox.is_exchange_holiday(exp_dt, exchange=exchange_prefix):
            exp_dt -= dt.timedelta(days=1)
            
        from_dt = exp_dt - dt.timedelta(days=7)

        # 2. Fetch Underlying Data
        underlying = historical(spot, from_date=from_dt, to_date=exp_dt, isexpired=False)
        if underlying.empty:
            logger.warning(f"No underlying data for {exp_str}. Skipping.")
            continue
            
        # Save Underlying
        HIST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_datewise(underlying, HIST_DATA_DIR, force=force, instrument_key=spot, **kwargs)

        # 3. Dynamic Strike Selection (±5 Bracket from Weekly Median)
        median_price = underlying["close"].median()
        atm_strike = round(median_price / step) * step
        target_strikes = [atm_strike + (i * step) for i in range(-5, 6)]

        # Filter Instruments for current expiry and target strikes
        valid_opts = instruments[
            (instruments["expiry"] == exp_str) & 
            (instruments["strike_price"].isin(target_strikes))
        ]
        
        if valid_opts.empty:
            continue

        # 4. Download and Cache Options
        for _, opt in valid_opts.iterrows():
            opt_key = opt["instrument_key"]
            
            # Check pre-flight cache: Assume if expiry day exists, the week is downloaded
            if not force and is_cached(HIST_DATA_DIR, exp_dt, opt_key, interval, unit, exchange_prefix):
                continue
                
            opt_data = historical(
                key=opt_key,
                expiry=exp_str,
                isexpired=is_expired,
                from_date=from_dt,
                to_date=exp_dt,
            )
            
            if not opt_data.empty:
                save_datewise(opt_data, HIST_DATA_DIR, force=force, **opt.to_dict(), **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download historical data for expired options.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    HELP_TABLE = """Historical Data Retrieval Limits:
------------------------------------------------------------------------------
Unit    | Intervals   | Available From | Max Retrieval Limit
------------------------------------------------------------------------------
minutes | 1, 2... 300 | Jan 2022       | 1 month (for 1-15 min intervals)
        |             |                | 1 quarter (for >15 min intervals)
hours   | 1, 2... 5   | Jan 2022       | 1 quarter leading up to to_date
days    | 1           | Jan 2000       | 1 decade leading up to to_date
weeks   | 1           | Jan 2000       | No limit
months  | 1           | Jan 2000       | No limit
------------------------------------------------------------------------------
    """
    parser.add_argument("-s", "--spot", default="NSE_INDEX|Nifty 50", type=str, help="Download underlying data | Requires the instrument key for the underlying instrument")
    parser.add_argument("-i", "--interval", default=1, type=int, help="Specify the interval for historical data | Default is 1 ")
    parser.add_argument("-u", "--unit", default="minutes", type=str, help="Specify the unit for the interval | Default is minutes |\n" + HELP_TABLE)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-exp", "--expired", action="store_true", help="Download expired option data")
    parser.add_argument("-f", "--force", action="store_true", help="Force download and overwrite existing local data")

    args = parser.parse_args()

    # Setup verbose logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    is_expired = True if args.expired else False
    
    # Execute Database Build
    download_data(
        args.spot,
        is_expired=is_expired,
        interval=args.interval,
        unit=args.unit,
        force=args.force,
    )