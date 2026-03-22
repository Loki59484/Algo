from pathlib import Path
from tqdm import tqdm
import datetime as dt
import pandas as pd
import logging
import json
import sys
import os
import argparse

# Adding root directory to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT_DIR / "core"
HIST_DATA_DIR = ROOT_DIR / "data" / "historical"


if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    sys.path.insert(0, str(CORE_DIR))
from core import upstox_func as ustox

# Intitiating logger
logger = logging.getLogger(__name__)
EXPIRED_KEYS_FILE = ustox.DATA_DIR / "expired_keys.parquet"


def historical(key, simdate, isexpired=False, expiry=None):
    if isexpired and expiry is None:
        raise ValueError("Must provide the expiry date for an expired instrument")
    if isexpired:
        history = ustox.get_historical(
            dtype="historical",
            expired_key=key,
            to_date=simdate,
            from_date=simdate,
            is_expired=True,
            expiry_date=expiry,
        )
    else:
        history = ustox.get_historical(
            dtype="intraday", instrument_key=key, from_date=simdate, to_date=simdate
        )
        history = (
            history
            if not history.empty
            else ustox.get_historical(
                dtype="historical",
                instrument_key=key,
                from_date=simdate,
                to_date=(simdate),
            )
        )
    return history


def download_data(spot, is_expired=False, interval=1, unit="minutes"):

    def save_datewise(data: pd.DataFrame, target_dir: Path, option_key: str):
        data["date"] = pd.to_datetime(data["timestamp"]).dt.date
        grouped = data.groupby("date")
        for date, group in grouped:
            date_str = date.strftime("%Y-%m-%d")
            file_path = (
                target_dir
                / f"{date.year}/{date.month}/{date.day}/{interval}_{unit}/{option_key[:-11]}.parquet"
            )
            group.drop(columns=["date"], inplace=True)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            group.to_parquet(file_path, index=False, engine="pyarrow")

    expiries = list(set(ustox.get_expiry(options=spot, is_expired=is_expired)))
    keys = {}
    if (
        os.path.exists(EXPIRED_KEYS_FILE)
        and not os.path.getsize(EXPIRED_KEYS_FILE) == 0
    ):
        keys = pd.read_parquet(EXPIRED_KEYS_FILE)
    else:
        for date in tqdm(expiries, "Getting expires"):
            # Get all expired instruments for the given expiry date and filter for CE with minimum lot of 65 or 75
            data = ustox.get_expired_instruments(expiry_date=date, underlying=spot)
            if data.empty:
                continue
            call_filter = data["trading_symbol"].str.contains("CE")
            put_filter = data["trading_symbol"].str.contains("PE")
            call_key = (data[call_filter])["instrument_key"]
            put_key = (data[put_filter])["instrument_key"]
            keys[date] = [call_key.values.tolist(), put_key.values.tolist()]

        # SAVE EXPIRED KEYS TO A FILE
        df = pd.DataFrame(keys)
        df.to_parquet(EXPIRED_KEYS_FILE, index=False, engine="pyarrow")

    holidays = pd.read_json(ustox.CONFIG_DIR / "holidays.json")
    for exp in tqdm(expiries, "Downloading Option data"):
        exp_dt = dt.datetime.strptime(exp, "%Y-%m-%d").date()

        if not (
            (holidays["date"] == exp)
            & (holidays["closed_exchanges"].str.contains("NSE", na=False))
        ).any():
            try:
                call_keys = keys[exp][0]
                call_key = call_keys[len(call_keys) // 2]
                put_keys = keys[exp][1]
                put_key = put_keys[len(put_keys) // 2]
                call_data = historical(
                    key=call_key,
                    expiry=exp,
                    isexpired=is_expired,
                    simdate=exp_dt,
                )
                put_data = historical(
                    key=put_key,
                    expiry=exp,
                    isexpired=is_expired,
                    simdate=exp_dt,
                )
                underlying = ustox.get_historical(
                    from_date=exp_dt - dt.timedelta(days=7), to_date=exp_dt
                )
            except KeyError:
                continue
            if call_data.empty or put_data.empty or underlying.empty:
                continue
            HIST_DATA_DIR.mkdir(parents=True, exist_ok=True)
            save_datewise(call_data, HIST_DATA_DIR, call_key)
            save_datewise(put_data, HIST_DATA_DIR, put_key)
            save_datewise(underlying, HIST_DATA_DIR, spot)


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
    parser.add_argument(
        "-s",
        "--spot",
        default="NSE_INDEX|Nifty 50",
        action="store_true",
        help="Download underlying data | Requires the instrument key for the underlying instrument",
    )
    parser.add_argument(
        "-i",
        "--interval",
        default=1,
        type=int,
        help="specify the interval for historical data | Default is 1 ",
    )
    parser.add_argument(
        "-u",
        "--unit",
        default="minutes",
        action="store_true",
        help="Specify the unit for the interval | Default is minutes |\n" + HELP_TABLE,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-exp", "--expired", action="store_true", help="Download expired option data"
    )

    args = parser.parse_args()

    # Setup verbose logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    is_expired = True if args.expired else False
    # Download data
    download_data(
        args.spot, is_expired=is_expired, interval=args.interval, unit=args.unit
    )
