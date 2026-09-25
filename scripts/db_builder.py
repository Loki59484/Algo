import sys
import datetime as dt
from pathlib import Path
import pandas as pd
from dateutil.relativedelta import relativedelta

# Ensure the core package is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo.tools.download_historical import fetch_chunked_history
from algo.core.upstox_methods import UpstoxClient

def build_continuous_backtest_db(spot_key: str, start_date: dt.date, end_date: dt.date):
    """
    Builds a custom, liquidity-optimized backtest database.
    Strategy: Always trades the near-month expiry. Brackets +/- 5 strikes.
    """
    client = UpstoxClient()
    db_path = ROOT_DIR / "data" / "backtest_db" / spot_key.replace("|", "_")
    db_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Get continuous spot data to calculate dynamic ATMs
    print(f"Fetching underlying spot data for {spot_key}...")
    spot_data = fetch_chunked_history(
        key=spot_key, from_date=start_date, to_date=end_date, interval=1, unit="minute"
    )
    spot_data['timestamp'] = pd.to_datetime(spot_data['timestamp'])
    
    # 2. Get all expiries for the asset
    expiries = client.get_options_with_expiry(options=spot_key, is_expired=True)
    target_expiries = sorted([dt.datetime.strptime(e, "%Y-%m-%d").date() for e in expiries])
    
    step_size = 50 if "NSE" in spot_key else 100
    
    # 3. Roll through time, downloading only the liquid window for each expiry
    current_date = start_date
    while current_date <= end_date:
        # Find the active near-month expiry (the first expiry AFTER our current date)
        active_expiry = next((exp for exp in target_expiries if exp >= current_date), None)
        if not active_expiry:
            break
            
        # Define the liquidity window (e.g., trade this contract until its expiry day)
        window_end = min(active_expiry, end_date)
        
        # Calculate ATM based on the spot price at the start of this specific window
        window_spot = spot_data[spot_data['timestamp'].dt.date == current_date]
        if window_spot.empty:
            current_date += dt.timedelta(days=1)
            continue
            
        atm_strike = round(window_spot['close'].iloc[0] / step_size) * step_size
        target_strikes = [atm_strike + (i * step_size) for i in range(-5, 6)]
        
        print(f"[{current_date} to {window_end}] Target Expiry: {active_expiry} | ATM: {atm_strike}")
        
        # TODO: Query InstrumentMaster for these specific strikes and call fetch_chunked_history
        # Save them into the partitioned db_path using PyArrow
        
        # Roll forward to the day after this contract expires
        current_date = window_end + dt.timedelta(days=1)

if __name__ == "__main__":
    start = dt.date(2023, 1, 1)
    end = dt.date(2023, 12, 31)
    build_continuous_backtest_db("MCX_FO|284561", start, end)
