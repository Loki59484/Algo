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

from core import upstox_methods as ustox
from core.anatomy import save_parquet
from core.datatypes import to_ist

# Intitiating logger
logger = logging.getLogger(__name__)

EXPIRED_KEYS_FILE = ustox.DATA_DIR / "expired_keys.parquet"


def historical(key, from_date, to_date, isexpired=False, expiry=None):
    if isexpired and expiry is None:
        raise ValueError("Must provide the expiry date for an expired instrument")
    if isexpired:
        history = ustox.get_historical(
            dtype="historical",
            expired_key=key,
            to_date=to_date,
            from_date=from_date,
            is_expired=True,
            expiry_date=expiry,
        )
    else:
        history = ustox.get_historical(
            dtype="intraday", instrument_key=key, from_date=from_date, to_date=to_date
        )
        history = (
            history
            if not history.empty
            else ustox.get_historical(
                dtype="historical",
                instrument_key=key,
                from_date=from_date,
                to_date=to_date,
            )
        )
    return history


def download_cache(
    option_key: str,
    expiry: str,
    is_expired: bool,
    date: dt.datetime,
    out_path: Path = None,
):
    history = historical(option_key, isexpired=is_expired, from_date=date,to_date=date, expiry=expiry)
    if history is None or history.empty:
        raise Exception("Failed to save cache!")
    if out_path:
        history.to_parquet(out_path, engine="pyarrow", compression="snappy")
        logger.info(f"Cache create for {option_key}")
    return history


def save_datewise(
    data: pd.DataFrame, target_dir: Path, makedirs: bool = True, **metadata
):
    data["date"] = to_ist(data["timestamp"]).dt.date
    grouped = data.groupby("date")
    for date, group in grouped:
        file_path = (
            target_dir
            / date.strftime("%Y")
            / date.strftime("%m")
            / date.strftime("%d")
            / f"{metadata.get('interval',"1")}_{metadata.get('unit',"minutes")}"
            / f"{metadata.get('instrument_key', 'unknown')}.parquet"
        )
        if file_path.exists():
            continue
        clean_group = group.drop(columns=["date"])
        save_parquet(clean_group, file_path, date=date, **metadata)


def download_data(spot, is_expired=False, interval=1, unit="minutes"):

    expiries = list(set(ustox.get_expiry(options=spot, is_expired=is_expired)))
    if (
        os.path.exists(EXPIRED_KEYS_FILE)
        and not os.path.getsize(EXPIRED_KEYS_FILE) == 0
    ):
        instruments = pd.read_parquet(EXPIRED_KEYS_FILE)
    else:
        instruments = []
        for date in tqdm(expiries, "Getting expires"):
            # Get all expired instruments for the given expiry date
            data = ustox.get_expired_instruments(expiry_date=date, underlying=spot)
            if data.empty:
                continue
            instruments.append(data)
        instruments = pd.concat(instruments, ignore_index=True)
        instruments.to_parquet(EXPIRED_KEYS_FILE, index=False, engine="pyarrow")

    holidays = pd.read_json(ustox.CONFIG_DIR / "holidays.json")
    for exp in tqdm(expiries, "Downloading Option data", position=0, leave=True):
        exp_dt = dt.datetime.strptime(exp, "%Y-%m-%d").date()

        if not (
            (holidays["date"] == exp)
            & (holidays["closed_exchanges"].str.contains("NSE", na=False))
        ).any():
            call_keys = instruments[
                (instruments["instrument_type"] == "CE")
                & (instruments["expiry"] == exp)
            ]
            call = call_keys.iloc[len(call_keys) // 2]
            put_keys = instruments[
                (instruments["instrument_type"] == "PE")
                & (instruments["expiry"] == exp)
            ]
            put = put_keys.iloc[len(put_keys) // 2]
            call_data = historical(
                key=call["instrument_key"],
                expiry=exp,
                isexpired=is_expired,
                from_date=exp_dt - dt.timedelta(days=7),
                to_date=exp_dt,
            )
            put_data = historical(
                key=put["instrument_key"],
                expiry=exp,
                isexpired=is_expired,
                from_date=exp_dt - dt.timedelta(days=7),
                to_date=exp_dt,
            )
            underlying = ustox.get_historical(
                from_date=exp_dt - dt.timedelta(days=7), to_date=exp_dt
            )
            if call_data.empty or put_data.empty or underlying.empty:
                continue
            HIST_DATA_DIR.mkdir(parents=True, exist_ok=True)
            kwargs = {
                "interval": interval,
                "unit": unit,
            }
            save_datewise(
                call_data,
                HIST_DATA_DIR,
                **call.to_dict(),
                **kwargs,
            )
            save_datewise(
                put_data,
                HIST_DATA_DIR,
                **put.to_dict(),
                **kwargs,
            )
            save_datewise(underlying, HIST_DATA_DIR, instrument_key=spot, **kwargs)


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
        type=str,
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
        type=str,
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
