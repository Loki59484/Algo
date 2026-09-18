from collections import defaultdict
from pathlib import Path
import pandas_ta as ta
import numpy as np
from tqdm import tqdm
import pandas as pd
import logging
import joblib
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
logger = logging.getLogger(__name__)

# IMPORTING CUSTOM MODULES
from core.datatypes import Instrument, Bucket
from core.upstox_methods import UpstoxClient, DATA_DIR
from core.methods import setup_cli, to_ist
from core import anatomy as ana
from backtesting import get_cached_probability_matrix
ML_DIR = ROOT_DIR / "ml"
MASTER_DF_PATH = ML_DIR / 'master_df.parquet'
TRAINING_DATA_PATH = ML_DIR / 'training_data.parquet'
TESTING_DATA_PATH = ML_DIR / 'testing_data.parquet'

def secure_prep(df):
    if df is None or df.empty:
        logger.info("secure prep failed")
        return pd.DataFrame()

    df = df.rename(columns={"vol": "volume"})

    if "timestamp" in df.columns:
        # THE FIX: Removed the double to_ist() conversion here since our 
        # datatypes.py factory now already perfectly localizes it to Asia/Kolkata!
        df = df.set_index("timestamp")

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def _worker(df: pd.DataFrame, study: ta.Study, **kwargs):
    if len(df) < 200:
        return pd.DataFrame(columns=df.columns)
    return ana.Strategy.apply_study(df, study=study, **kwargs)

def process_bucket(bucket: Bucket, trader: ana.Trader, master_list: list,  **kwargs):
    spot: Instrument = bucket.spot
    call_option: Instrument = bucket.legs.get("CE")
    put_option: Instrument = bucket.legs.get("PE")

    spot_df = spot.load_historical_df(ustox, 30, False)
    ce_df = call_option.load_historical_df(ustox, 1, False)
    pe_df = put_option.load_historical_df(ustox, 1, False)

    spot_df = secure_prep(spot_df)
    ce_df = secure_prep(ce_df)
    pe_df = secure_prep(pe_df)

    try:
        ce_df["ATR"] = ce_df.ta.atr()
        pe_df["ATR"] = pe_df.ta.atr()
    except Exception:
        ce_df["ATR"] = 0.0
        pe_df["ATR"] = 0.0

    spot_df = _worker(spot_df, trader.strategy.indicators, **kwargs)
    
    # Added safe fallbacks for varying pandas_ta naming conventions
    spot_df = spot_df.rename(
    columns={
        "SUPERT_14_2.0": "SUPERT",
        "SUPERTl_14_2.0": "SUPERTl",
        "SUPERTs_14_2.0": "SUPERTs",
        "SUPERTd_14_2.0": "SUPERTd",
        "ADX_14": "ADX",
        "ADXR_14_2": "ADXR",
        "DMP_14": "DMP",
        "DMN_14": "DMN",
        "ATRr_14": "ATR",
        "RSI_14": "RSI",
        "MACD_12_26_9": "MACD",            
        "MACDs_12_26_9": "MACD_signal",    
        "BBL_20_2.0_2.0": "BBL",               
        "BBU_20_2.0_2.0": "BBU", 
        "BBL_20_2.0": "BBL",      # Safety Catch           
        "BBU_20_2.0": "BBU",      # Safety Catch          
    }
    )
    
    spot_df["SUPERT_slope"] = spot_df["SUPERT"].diff()
    spot_df["BB_width"] = (spot_df["BBU"] - spot_df["BBL"]) / spot_df["close"]

    if spot_df.empty or ce_df.empty or pe_df.empty:
        return False

    # THE FIX: Timezone-Safe Date Comparison
    # We use `.date` to extract the raw calendar date, bypassing all 
    # naive/aware timezone matching bugs!
    target_date = pd.to_datetime(bucket.date).date()

    historical_training_data = spot_df[spot_df.index.date < target_date]
    spot_df_new = spot_df[spot_df.index.date == target_date]

    ce_df_new = ce_df[ce_df.index.date == target_date].reindex(
        spot_df_new.index, method="ffill"   
    )
    pe_df_new = pe_df[pe_df.index.date == target_date].reindex(
        spot_df_new.index, method="ffill"
    )
    
    if spot_df_new.empty or ce_df_new.empty or pe_df_new.empty:
        logger.info(
            f"empty df - spot: {spot_df_new.empty} | ce: {ce_df_new.empty} | pe: {pe_df_new.empty}"
        )
        return False

    prob_matrix = get_cached_probability_matrix(
        df=historical_training_data,
        instrument_key=spot.key,
        date_str=str(bucket.date),
        strategy_params={"lookback": 30, "supertrend": 2.0, "adxr": 16},
    )

    bucket.probability_matrix = prob_matrix

    ce_df_new["next_open"] = ce_df_new["open"].shift(-1)
    pe_df_new["next_open"] = pe_df_new["open"].shift(-1)

    combined_df = spot_df_new.copy()
    
    for df, side in zip((ce_df_new, pe_df_new), ("ce", "pe")):
        for obj in ["open", "high", "low", "close", "ATR", "next_open"]: 
            combined_df[f"{side}_{obj}"] = df.get(obj)

    columns_to_drop = ["SUPERTl", "SUPERTs", "VWAP", "buy_signal", "sell_signal"]
    existing_cols_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
    combined_df.drop(columns=existing_cols_to_drop, inplace=True)
    
    combined_df = combined_df.dropna()
    
    if not combined_df.empty:
        master_list.append(combined_df)

    return True

def integrity_check(df):
        
    print("="*50)
    print("📊 MASTER DATAFRAME INTEGRITY REPORT 📊")
    print("="*50)

    # 1. Size Check
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")

    # 2. Time Sorting Check
    is_sorted = df.index.is_monotonic_increasing
    print(f"Chronologically Sorted: {'✅ YES' if is_sorted else '❌ NO (CRITICAL ERROR)'}")

    # 3. Duplicate Timestamp Check
    duplicate_times = df.index.duplicated().sum()
    if duplicate_times == 0:
        print("Duplicate Timestamps: ✅ 0")
    else:
        print(f"Duplicate Timestamps: ❌ {duplicate_times} (Warning: Will confuse the model)")

    # 4. NaN / Missing Value Check
    total_nans = df.isna().sum().sum()
    if total_nans == 0:
        print("Missing Values (NaNs): ✅ 0")
    else:
        print("Missing Values (NaNs): ❌ FAILED. See breakdown:")
        print(df.isna().sum()[df.isna().sum() > 0])

    # 5. Infinity Check
    has_inf = np.isinf(df.select_dtypes(include=[np.number])).values.any()
    print(f"Infinity Values (inf): {'❌ FOUND' if has_inf else '✅ 0'}")

    # 6. Basic Value Logic Checks
    print("\n--- Logic Checks ---")
    ce_logic = (df['ce_high'] >= df['ce_low']).all()
    pe_logic = (df['pe_high'] >= df['pe_low']).all()
    print(f"CE High >= Low: {'✅ PASS' if ce_logic else '❌ FAIL'}")
    print(f"PE High >= Low: {'✅ PASS' if pe_logic else '❌ FAIL'}")

    print("\nReport Complete.")
    
    
def split_data(df : pd.DataFrame, training_ratio: float = 0.7):
    df = df.sort_index()
    total_rows = len(df)
    split_index = int(total_rows * training_ratio)
    target_split_date = df.index[split_index].date()    
    train_df = df[df.index.date < target_split_date].copy()
    test_df = df[df.index.date >= target_split_date].copy()

    print("\n" + "="*50)
    print("🪓 CHRONOLOGICAL TRAIN/TEST SPLIT REPORT 🪓")
    print("="*50)
    
    print(f"\n[TRAINING SET] - Optuna's Sandbox")
    print(f"Total Rows : {len(train_df):,}")
    print(f"Start Date : {train_df.index.min()}")
    print(f"End Date   : {train_df.index.max()}")
    
    print(f"\n[TESTING SET] - The Final Reality Check")
    print(f"Total Rows : {len(test_df):,}")
    print(f"Start Date : {test_df.index.min()}")
    print(f"End Date   : {test_df.index.max()}")
    
    train_df.to_parquet(TRAINING_DATA_PATH)
    test_df.to_parquet(TESTING_DATA_PATH)
                       
    print(f"✅ Success! Saved to:\n- {TRAINING_DATA_PATH}\n- {TESTING_DATA_PATH}")

if __name__ == "__main__":
    args = setup_cli()
    ustox = UpstoxClient()
    if not MASTER_DF_PATH.exists() or args.no_cache:
        master_list = []
        strat = ana.Strategy()
        strat.add_indicators(
    [
        {"kind": "supertrend", "length": 14, "multiplier": 2.0},
        {"kind": "adx", "length": 14},
        {"kind": "atr", "length": 14},
        {"kind": "ema", "length": 200},
        {"kind": "ema", "length": 50},
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},  
        {"kind": "bbands", "length": 20, "std": 2.0},           
    ]
)
        trader = ana.Trader(None, strat, None, None)
        INSTRUMENT_CACHE = DATA_DIR / "cache" / f"{args.bulk[0]}_instruments_cache.joblib"
        files = list((DATA_DIR / "historical" / args.bulk[0]).rglob("*.parquet"))
        files.sort()

        if INSTRUMENT_CACHE.exists() and not args.no_cache:
            print("Loading instruments from cache...", end="\r")
            insts = joblib.load(INSTRUMENT_CACHE)
            print(f"Loaded {len(insts)} instruments from cache.", end="\r")
        else:
            insts = Instrument.load_multiple(client=ustox, source=files, lookback=30)
            joblib.dump(insts, INSTRUMENT_CACHE)
        insts_dict = {(item.key, item.date): item for item in insts}
        trader.add_instrument(insts_dict)
        
        # CREATE BUCKETS FOR EACH DAY
        daily_buckets = defaultdict(dict)
        for instrument in tqdm(
            trader.instruments.values(), desc="Filtering instruments", leave=False
        ):
            if "CE" in instrument.type:
                leg_type = "CE"
            elif "PE" in instrument.type:
                leg_type = "PE"
            else:
                leg_type = "INDEX"
            daily_buckets[instrument.date][leg_type] = instrument
            
        for trade_date, legs in tqdm(
            daily_buckets.items(), desc="Loading Buckets", leave=False
        ):
            options = {k: v for k, v in legs.items() if k in ["CE", "PE"]}
            bucket = Bucket(trade_date, legs=options, spot=legs.get("INDEX", None))
            trader.buckets.append(bucket)
            
        for bucket in tqdm(trader.buckets, desc="Processing Buckets", leave=False):
            process_bucket(bucket=bucket, trader=trader,master_list=master_list )

        master_df = pd.concat(master_list)
        master_df = master_df.sort_index()
        master_df.to_parquet(MASTER_DF_PATH)
        
    else:
        print("LOADING FROM FILE",end='\r')
        master_df = pd.read_parquet(MASTER_DF_PATH)
    
    integrity_check(master_df)
    split_data(master_df)
    print("DATABASE BUILT SUCCESSFULLY")