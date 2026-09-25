import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# Adding root directory to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo.core.upstox_methods import UpstoxClient, DATA_DIR
from algo.core.datatypes import to_ist

logger = logging.getLogger(__name__)
ustox = UpstoxClient()
HIST_DATA_DIR = DATA_DIR / "historical"


def download_cache(
    client,
    instrument_key: str,
    from_date,
    to_date,
    out_path: Path,
    is_expired: bool = True,
    expiry=None
) -> pd.DataFrame:
    """
    Downloads historical data from Upstox and saves it to a local Parquet cache file.
    """
    logger.info(f"Downloading cache for {instrument_key} to {out_path}")
    
    # Fetch data using your existing client logic
    data = client.get_historical(
        dtype="historical",
        instrument_key=instrument_key,
        from_date=from_date,
        to_date=to_date,
        is_expired=is_expired,
        expiry_date=expiry
    )
    
    if data is not None and not data.empty:
        # Ensure the cache directory exists before saving
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out_path, index=False)
        logger.info(f"Successfully cached {len(data)} rows.")
    else:
        logger.warning(f"No data returned from Upstox for {instrument_key}.")
        data = pd.DataFrame()  # Return an empty DataFrame to prevent downstream crashes
        
    return data


def fetch_chunked_history(
    key: str, 
    from_date: dt.date, 
    to_date: dt.date, 
    is_expired: bool = False, 
    expiry: str | None = None,
    interval: int = 1,
    unit: str = "minute"
) -> pd.DataFrame:
    """
    Downloads historical data by slicing long date ranges into 1-month API-compliant chunks.
    """
    historical_dfs = []
    chunk_start = from_date

    while chunk_start <= to_date:
        # Step forward by 1 month to respect Upstox limit for 1-15 min intervals
        chunk_end = min(chunk_start + relativedelta(months=1) - dt.timedelta(days=1), to_date)
        
        logger.info(f"Downloading {key} chunk: {chunk_start} to {chunk_end}")
        
        # Route strictly through the unified client method
        data = ustox.get_historical(
            dtype="historical" if is_expired else "intraday",
            instrument_key=key if not is_expired else None,
            expired_key=key if is_expired else None,
            from_date=chunk_start,
            to_date=chunk_end,
            is_expired=is_expired,
            expiry_date=expiry,
            interval=interval,
            unit=unit
        )
        
        if data is not None and not data.empty:
            historical_dfs.append(data)
            
        chunk_start = chunk_end + dt.timedelta(days=1)

    if not historical_dfs:
        return pd.DataFrame()

    clean_data = pd.concat(historical_dfs, ignore_index=True)
    clean_data.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    return clean_data


def save_monthly(data: pd.DataFrame, target_dir: Path, key: str, exchange: str, interval: int, unit: str, force: bool = False):
    """
    Saves DataFrame into consolidated monthly Parquet files to optimize ML loading speeds.
    """
    if data.empty:
        return
        
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["year_month"] = data["timestamp"].dt.strftime("%Y_%m")
    
    grouped = data.groupby("year_month")
    
    # Sanitize the key for valid filenames (e.g., NSE_INDEX|Nifty 50 -> NSE_INDEX_Nifty 50)
    safe_key = key.replace("|", "_").replace(" ", "")
    
    for ym, group in grouped:
        file_path = target_dir / exchange / safe_key / f"{ym}_{interval}_{unit}.parquet"

        if file_path.exists() and not force:
            continue

        clean_group = group.drop(columns=["year_month"])
        file_path.parent.mkdir(parents=True, exist_ok=True)
        clean_group.to_parquet(file_path, index=False, engine="pyarrow")
        logger.debug(f"Saved {len(clean_group)} rows to {file_path.name}")


def download_data(spot: str, start_date: dt.date, end_date: dt.date, is_expired: bool = False, interval: int = 1, unit: str = "minute", force: bool = False):
    """
    Orchestrates the downloading of continuous spot data and bracketed options chains.
    """
    exchange_prefix = spot.split("_")[0] if "_" in spot else spot[:3]
    kwargs = {"interval": interval, "unit": unit}

    # 1. Download Continuous Underlying Spot Data
    logger.info(f"Fetching continuous underlying data for {spot}...")
    underlying = fetch_chunked_history(
        key=spot, 
        from_date=start_date, 
        to_date=end_date, 
        is_expired=False,
        **kwargs
    )
    
    if not underlying.empty:
        save_monthly(underlying, HIST_DATA_DIR, key=spot, exchange=exchange_prefix, force=force, **kwargs)
    else:
        logger.error(f"Failed to fetch underlying data for {spot}. Aborting options download.")
        return

    if not is_expired:
        logger.info("Spot data fully downloaded. Exiting as options (-exp) were not requested.")
        return

    # 2. Download Options Data Bracketed Around ATM
    EXPIRED_KEYS_FILE = DATA_DIR / f"{exchange_prefix}_expired_keys.parquet"
    
    expiries = ustox.get_options_with_expiry(options=spot, is_expired=True)
    if not expiries:
        logger.error(f"No expiries found for {spot}.")
        return

    # Filter expiries to only those that fall within the requested date range
    target_expiries = [exp for exp in expiries if start_date <= dt.datetime.strptime(exp, "%Y-%m-%d").date() <= end_date]
    if not target_expiries:
        logger.warning(f"No expiries fall between {start_date} and {end_date}.")
        return

    # Cache expired keys to avoid redundant API lookups
    if EXPIRED_KEYS_FILE.exists() and os.path.getsize(EXPIRED_KEYS_FILE) > 0 and not force:
        instruments = pd.read_parquet(EXPIRED_KEYS_FILE)
    else:
        instruments_list = []
        for date_str in tqdm(target_expiries, desc="Caching Expired Instruments"):
            data = ustox.get_expired_instruments(expiry_date=date_str, underlying=spot)
            if data is not None and not data.empty:
                instruments_list.append(data)
        
        if not instruments_list:
            logger.error("No expired instruments found.")
            return
            
        instruments = pd.concat(instruments_list, ignore_index=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        instruments.to_parquet(EXPIRED_KEYS_FILE, index=False, engine="pyarrow")

    step = 50 if "NSE" in exchange_prefix else 100

    for exp_str in tqdm(target_expiries, desc="Downloading Options Chains", position=0, leave=True):
        exp_dt = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        
        # Calculate ATM strike using the median price of the underlying during this contract's lifecycle
        contract_life = underlying[(underlying["timestamp"].dt.date <= exp_dt)]
        if contract_life.empty:
            continue
            
        median_price = contract_life["close"].median()
        atm_strike = round(median_price / step) * step
        target_strikes = [atm_strike + (i * step) for i in range(-5, 6)]

        valid_opts = instruments[
            (instruments["expiry"] == exp_str) & 
            (instruments["strike_price"].isin(target_strikes))
        ]
        
        if valid_opts.empty:
            continue

        # Option contract lifecycle typically spans a maximum of 3 months backward from expiry
        opt_start_date = max(start_date, exp_dt - dt.timedelta(days=90))

        for _, opt in valid_opts.iterrows():
            opt_key = opt["instrument_key"]
            opt_data = fetch_chunked_history(
                key=opt_key,
                from_date=opt_start_date,
                to_date=exp_dt,
                is_expired=True,
                expiry=exp_str,
                **kwargs
            )
            
            if not opt_data.empty:
                save_monthly(opt_data, HIST_DATA_DIR, key=opt_key, exchange=exchange_prefix, force=force, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download continuous historical data for Spot and Expired Options.",
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
------------------------------------------------------------------------------
    """
    
    # Calculate default 1-year window
    default_end = dt.date.today()
    default_start = default_end - relativedelta(years=1)

    parser.add_argument("-s", "--spot", default="NSE_INDEX|Nifty 50", type=str, help="The instrument key for the underlying asset")
    parser.add_argument("--start-date", type=lambda s: dt.datetime.strptime(s, '%Y-%m-%d').date(), default=default_start, help="Start date (YYYY-MM-DD). Defaults to 1 year ago.")
    parser.add_argument("--end-date", type=lambda s: dt.datetime.strptime(s, '%Y-%m-%d').date(), default=default_end, help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("-i", "--interval", default=1, type=int, help="Candle interval (default: 1)")
    parser.add_argument("-u", "--unit", default="minute", type=str, help="Interval unit (default: minute)\n" + HELP_TABLE)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-exp", "--expired", action="store_true", help="Download expired options chain around the ATM strike")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing local parquet files")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    download_data(
        spot=args.spot,
        start_date=args.start_date,
        end_date=args.end_date,
        is_expired=args.expired,
        interval=args.interval,
        unit=args.unit,
        force=args.force,
    )
