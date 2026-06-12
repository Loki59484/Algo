from collections import defaultdict
from dataclasses import asdict
from functools import partial
from typing import Optional
from pathlib import Path
import pandas_ta as ta
from tqdm import tqdm
import datetime as dt
import pandas as pd
import numpy as np
import hashlib
import logging
import joblib
import shutil
import json
import sys


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
logger = logging.getLogger(__name__)

# IMPORTING CUSTOM MODULES
from core.datatypes import Instrument, Trade, Portfolio, Bucket, Funds
from core.upstox_methods import UpstoxClient, DATA_DIR
from core.methods import setup_cli, to_ist
from core import anatomy as ana

ustox = UpstoxClient()
supert = "SUPERT_14_2.0"

MATRIX_CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache" / "matrices"
MATRIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(
    "--------------------SIMULATOR--------------------".center(
        shutil.get_terminal_size().columns
    )
)

# DEFINING BUY-SELL PARAMETERS


# DEFINING BUY-SELL PARAMETERS
#def buy_signal(df, **kwargs):
#    try:
#        # --- SHARED MOMENTUM (Both CE and PE need the same ADXR waking up) ---
#        adxr_trend = (16 < df["ADXR_14_2"]) & (df["ADXR_14_2"] < 25)
#
#        dmi_gap = abs(df["DMP_14"] - df["DMN_14"])
#
#        ce_signal = (
#            (df["close"] > df[supert])
#            & (df[supert] > df[supert].shift(1))  # Supertrend steps UP
#            & adxr_trend
#            & (df["DMP_14"] > df["DMN_14"])  # Bulls in control
#            & (dmi_gap > 10)  # Massive bullish gap
#            & (df["RSI_14"] > 59)  # Spot RSI shows extreme breakout strength
#        )
#
#        pe_signal = (
#            (df["close"] < df[supert])
#            & (df[supert] < df[supert].shift(1))  # Supertrend steps DOWN
#            & adxr_trend
#            & (df["DMN_14"] > df["DMP_14"])  # Bears in control
#            & (dmi_gap > 11)  # Massive bearish gap
#            & (df["RSI_14"] < 37)  # Spot RSI shows extreme breakdown weakness
#        )
#
#        signal_col = pd.Series(None, index=df.index, dtype=object)
#        signal_col[ce_signal] = "CE"
#        signal_col[pe_signal] = "PE"
#
#        return signal_col
#
#    except Exception as e:
#        logger.exception(e)

def buy_signal(df, **kwargs):
    try:
        breakpoint()
        # ---------------------------------------------------------
        # STEP 1: Discretize the Data into States (Vectorized)
        # ---------------------------------------------------------
        # You still need to define what "State" the row is in based on your indicators.
        trend_up = df["close"] > df["SUPERT_14_2.0"]
        adxr_trend = (16 < df["ADXR_14_2"]) & (df["ADXR_14_2"] < 25)
        
        df['State'] = np.where(trend_up & adxr_trend, 'Strong Bull',
                      np.where(trend_up & ~adxr_trend, 'Weak Bull',
                      np.where(~trend_up & adxr_trend, 'Strong Bear', 'Weak Bear')))

        # ---------------------------------------------------------
        # STEP 2: The Matrix Math (Vectorized Inference)
        # ---------------------------------------------------------
        # Convert the textual states into a binary One-Hot Matrix (Rows x 4 States)
        # We reindex to ensure the columns exactly match the rows of your probability_matrix
        state_matrix = pd.get_dummies(df['State']).reindex(
            columns=probability_matrix.index, fill_value=0
        ).to_numpy()

        # MULTIPLY!
        # (N_Rows x 4 States) @ (4 States x 3 Outcomes) = (N_Rows x 3 Probabilities)
        # Let's assume your matrix columns are ['DOWN', 'FLAT', 'UP']
        prob_output = state_matrix @ probability_matrix.to_numpy()

        # Extract the probability columns into the dataframe for easy handling
        df['Prob_DOWN'] = prob_output[:, 0]
        df['Prob_FLAT'] = prob_output[:, 1]
        df['Prob_UP'] = prob_output[:, 2]

        # ---------------------------------------------------------
        # STEP 3: The Confidence Threshold (Bridge to Actions)
        # ---------------------------------------------------------
        # Define how certain the matrix must be to trigger a trade
        CONFIDENCE_THRESHOLD = 0.65  # 65% probability required

        ce_signal = df['Prob_UP'] > CONFIDENCE_THRESHOLD
        pe_signal = df['Prob_DOWN'] > CONFIDENCE_THRESHOLD

        # Return exactly what your existing execution engine expects
        signal_col = pd.Series(None, index=df.index, dtype=object)
        signal_col[ce_signal] = "CE"
        signal_col[pe_signal] = "PE"

        return signal_col

    except Exception as e:
        logger.exception(e)




def sell_signal(df, **kwargs):
    square_off = pd.Series(df.index == df.index[-1], index=df.index)
    ce_signal = (df["close"] > df[supert]) & (
        df[supert] == df[supert].shift(1)
    ) | square_off
    pe_signal = (df["close"] < df[supert]) & (
        df[supert] == df[supert].shift(1)
    ) | square_off

    signal_col = pd.Series(None, index=df.index, dtype=object)
    signal_col[ce_signal] = "CE"
    signal_col[pe_signal] = "PE"

    return signal_col


def buy_cons(**kwargs):
    target = kwargs.get("bucket", None)
    return True if target.open_position is None else False


def sell_cons(**kwargs):
    target = kwargs.get("bucket", None)
    return False if target.open_position is None else True


def _worker(df: pd.DataFrame, study: ta.Study, kwargs):
    if len(df) < 200:
        return pd.DataFrame(columns=df.columns)
    return ana.Strategy.apply_study(df, study=study, **kwargs)


def check_tradable_hours(row):
    t = row.timestamp.time()
    morning_window = dt.time(10, 0) < t < dt.time(12, 0)
    noon_window = dt.time(13, 0) < t < dt.time(14, 0)
    final_window = t > dt.time(15, 0)
    return any([morning_window, noon_window, final_window])


def execute_sell(spot_row, row, option, trader, bucket, side=None):
    stoploss = bucket.open_position.stoploss
    target = bucket.open_position.target

    stoploss_hit = row.low <= stoploss if stoploss is not None else False
    target_hit = row.high >= target if target is not None else False
    signal_hit = spot_row.sell_signal == side

    if not (stoploss_hit or target_hit or signal_hit):
        return

    if pd.isna(row.next_open):
        return  # End of dataset

    if stoploss_hit:
        base_price = max(0.05, min(stoploss, row.open))
        exec_time, remark, buffer_points = row.timestamp, "SL", 5.00
    elif target_hit:
        base_price = max(target, row.open)
        exec_time, remark, buffer_points = row.timestamp, "Target", 2.00
    else:
        base_price = max(0.05, row.next_open if pd.notna(row.next_open) else row.close)
        exec_time = row.next_time if pd.notna(row.next_time) else row.timestamp
        remark, buffer_points = "Signal", 2.00

    buffered_sell_price = max(0.05, round(base_price - buffer_points, 2))

    status = trader.broker.sell_order(
        key=option.key,
        qty=trader.portfolio.report[-1].buy_qty,
        price=buffered_sell_price,
    )

    if status == -1:
        logger.error(f"Sell order not placed for {option.key}. Gapped down past limit.")
        return

    report: Trade = trader.portfolio.report[-1]
    report.sell_conditions = spot_row._asdict()
    report.sell_qty = report.buy_qty
    report.sell_timestamp = exec_time
    report.sell_price = base_price
    report.remark = remark
    report.movement = report.sell_price - report.buy_price
    report.pnl = (report.sell_price * report.sell_qty) - (
        report.buy_price * report.buy_qty
    )
    report.total = trader.portfolio.funds.total

    bucket.open_position = None
    logger.info(f"Trade executed successfully for sell side ({remark}).")


def execute_buy(spot_row, row, option, trader, strategy, bucket, **kwargs):
    if check_tradable_hours(row):
        return -1

    buy_cons = (
        strategy.buy_constraints(bucket=bucket, **kwargs)
        if strategy.buy_constraints
        else True
    )
    if not buy_cons:
        return -1

    exec_price = row.next_open if pd.notna(row.next_open) else row.close
    exec_time = row.next_time if pd.notna(row.next_time) else row.timestamp
    if exec_price < 50.0:
        return -1

    qty = trader.calculate_units(close=exec_price, lot_size=option.lot_size)
    logger.info(f"lot_size = {option.lot_size}, qty = {qty}")

    if qty == 0:
        return -1

    status = trader.broker.buy_order(
        key=option.key, price=exec_price, qty=qty, order_type="LIMIT"
    )
    if status is None:
        return -1

    pos, ord_id = status[0], status[1]
    bucket.open_position = pos
    bucket.open_position.stoploss = max(0.05, exec_price - (1.5 * row.ATR))
    bucket.open_position.target = exec_price + (22 * row.ATR)

    trader.portfolio.report.append(
        Trade(
            trade_id=ord_id,
            instrument_key=option.key,
            buy_timestamp=exec_time,
            side=option.type,
            buy_price=exec_price,
            buy_qty=qty,
            buy_conditions=spot_row._asdict(),
        )
    )
    logger.info("Trade executed successfully for buy side.")
    return 1


import pandas as pd

def prepare_bucket_data(
    bucket: Bucket,
    strategy: ana.Strategy,
    call_option: Instrument,
    put_option: Instrument,
    spot: Instrument,
    **kwargs,
):
    """Lazy loads data, calculates TA, builds probability matrix, and formats it safely."""

    if not call_option or not put_option:
        logger.warning(f"NoneType option found for bucket {bucket.date}")
        return False

    # 1. Load 30-day history
    spot_df = spot.load_historical_df(ustox, lookback=30)
    ce_df = call_option.load_historical_df(ustox, lookback=3)
    pe_df = put_option.load_historical_df(ustox, lookback=3)

    # ---------------------------------------------------------
    # 2. BULLETPROOF DATA FORMATTER
    # ---------------------------------------------------------
    def secure_prep(df):
        if df is None or df.empty:
            return pd.DataFrame()
            
        df = df.rename(columns={"vol": "volume"})
        
        if "timestamp" in df.columns:
            df["timestamp"] = to_ist(df["timestamp"])
            # Explicit reassignment (no inplace=True)
            df = df.set_index("timestamp")
            
        # Hard-enforce DatetimeIndex to prevent .normalize() crashes
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    spot_df = secure_prep(spot_df)
    ce_df = secure_prep(ce_df)
    pe_df = secure_prep(pe_df)

    # 3. Calculate Technical Indicators on the full 30-day history
    spot_df = _worker(spot_df, strategy.indicators, kwargs)
    spot_df = spot_df.rename(
        columns={
            "SUPERT_14_2.0": "SUPERT", # Included this in case you use it in matrix logic
            "SUPERTl_14_2.0": "SUPERTl",
            "SUPERTs_14_2.0": "SUPERTs",
            "SUPERTd_14_2.0": "SUPERTd",
            "ADX_14": "ADX",
            "ADXR_14_2": "ADXR",
            "DMP_14": "DMP",
            "DMN_14": "DMN",
            "ATRr_14": "ATR",
            "EMA_200": "EMA",
            "RSI_14": "RSI",
        }
    )

    # Guard clause in case indicator generation fails or data is missing
    if spot_df.empty or ce_df.empty or pe_df.empty:
        return False

    target_date = pd.Timestamp(bucket.date)

    historical_training_data = spot_df[spot_df.index < target_date]
    
    prob_matrix = get_cached_probability_matrix(
        df=historical_training_data,
        instrument_key=spot.key,
        date_str=str(bucket.date),
        strategy_params={"lookback": 30, "supertrend": 2.0, "adxr": 16} 
    )
    
    # Bypass dataclass freeze lock if Bucket is frozen
    bucket.probability_matrix= prob_matrix

    spot_df = spot_df[spot_df.index.normalize() == target_date]
    ce_df = ce_df[ce_df.index.normalize() == target_date]
    pe_df = pe_df[pe_df.index.normalize() == target_date]

    if spot_df.empty or ce_df.empty or pe_df.empty:
        return False
        

    spot.historical_df = spot_df.reset_index()

    # 6. Finalize Option specific logic
    for opt, df in zip((call_option, put_option), (ce_df, pe_df)):
        df = df.reset_index()
        df["next_open"] = df["open"].shift(-1)
        df["next_time"] = df["timestamp"].shift(-1)
        df["ATR"] = df.ta.atr()
        opt.historical_df = df.reset_index(drop=True)
        
    return True


def process_single_bucket(bucket, trader, strategy, **kwargs):
    spot: Instrument = bucket.spot
    call_option: Instrument = bucket.legs.get("CE")
    put_option: Instrument = bucket.legs.get("PE")
    bucket_status = prepare_bucket_data(
        bucket=bucket,
        strategy=strategy,
        call_option=call_option,
        put_option=put_option,
        spot=spot,
        **kwargs,
    )
    if not bucket_status:
        return False
    try:
        for row_ce, row_pe, row_spot in zip(
            call_option.historical_df.itertuples(),
            put_option.historical_df.itertuples(),
            spot.historical_df.itertuples(),
        ):

            if bucket.open_position is None:
                if (
                    all(
                        (
                            row_spot.buy_signal == "CE",
                            execute_buy(
                                spot_row=row_spot,
                                row=row_ce,
                                option=call_option,
                                trader=trader,
                                strategy=strategy,
                                bucket=bucket,
                                **kwargs,
                            ),
                        )
                    )
                    == 1
                ):
                    continue
                if (
                    all(
                        (
                            row_spot.buy_signal == "PE",
                            execute_buy(
                                spot_row=row_spot,
                                row=row_pe,
                                option=put_option,
                                trader=trader,
                                strategy=strategy,
                                bucket=bucket,
                                **kwargs,
                            ),
                        )
                    )
                    == 1
                ):
                    continue
            else:
                if bucket.open_position.instrument_token == call_option.key:
                    execute_sell(
                        row_spot, row_ce, call_option, trader, bucket, side="CE"
                    )
                elif bucket.open_position.instrument_token == put_option.key:
                    execute_sell(
                        row_spot, row_pe, put_option, trader, bucket, side="PE"
                    )

    except Exception as e:
        logger.exception(f"{e}")
        breakpoint()
    finally:
        spot.historical_df = None
        call_option.historical_df = None
        put_option.historical_df = None


def procedure(trader: ana.Trader, strategy: ana.Strategy, **kwargs):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.colheader_justify", "center")

    trader.buckets.sort(key=lambda b: b.date)

    for bucket in tqdm(trader.buckets, desc="Backtesting buckets", leave=False):
        status = process_single_bucket(bucket, trader, strategy, **kwargs)
        if status == False:
            continue
        trader.portfolio.funds.settle()

    logger.info("Simulation Complete")


strat = ana.Strategy()
strat.add_indicators(
    [
        {"kind": "supertrend", "length": 14, "multiplier": 2.0},
        {"kind": "adx", "length": 14},
        {"kind": "atr", "length": 14},
        {"kind": "ema", "length": 200},
        {"kind": "rsi", "length": 14},
        {"kind": "vwap"},
    ]
)

def _assign_market_states(df: pd.DataFrame) -> pd.Series:
    """Discretizes continuous indicators into a finite set of market states."""
    # Using your previous SuperTrend and ADXR logic as the baseline
    trend_up = df['close'] > df['SUPERT']
    adxr_strong = df['ADXR'] > 16

    # Vectorized state assignment using np.where
    states = np.where(trend_up & adxr_strong, 'Strong Bull',
             np.where(trend_up & ~adxr_strong, 'Weak Bull',
             np.where(~trend_up & adxr_strong, 'Strong Bear', 'Weak Bear')))
    
    return pd.Series(states, index=df.index, name="State")


def _assign_future_outcomes(df: pd.DataFrame, buffer_pts: float = 2.0) -> pd.Series:
    """Classifies the NEXT candle's movement as UP, DOWN, or FLAT."""
    next_close = df['close'].shift(-1)
    
    # If the next close moved more than 'buffer_pts', classify as UP/DOWN. Otherwise FLAT.
    outcomes = np.where(next_close > df['close'] + buffer_pts, 'UP', 
               np.where(next_close < df['close'] - buffer_pts, 'DOWN', 'FLAT'))
    
    return pd.Series(outcomes, index=df.index, name="Outcome")


def generate_probability_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Builds a column-stochastic transition matrix from a historical DataFrame.
    Rows = Future Outcomes (DOWN, FLAT, UP)
    Columns = Current States
    """
    if df.empty or len(df) < 2:
        return None

    df = df.copy()
    
    # 1. Attach States and Outcomes
    df['State'] = _assign_market_states(df)
    df['Outcome'] = _assign_future_outcomes(df)

    # Drop the very last row, as we cannot know its "future" outcome
    df = df.iloc[:-1]

    # 2. Build the Raw Tally Matrix
    # We put Outcomes on the index (rows) and States on the columns
    tally = pd.crosstab(df['Outcome'], df['State'])

    # Ensure all outcomes exist in the matrix rows, even if the market never went "UP" in this 30-day window
    expected_outcomes = ['DOWN', 'FLAT', 'UP']
    tally = tally.reindex(expected_outcomes, fill_value=0)

    # 3. Normalize into Probabilities (Column-Stochastic)
    # Divide every cell by the total sum of its specific column
    prob_matrix = tally.div(tally.sum(axis=0), axis=1)

    # 4. Handle Edge Cases (States that never occurred)
    # If a state never happened, its column sum is 0, resulting in NaNs.
    # We fill these with uniform uncertainty (33% chance for all directions).
    uniform_prob = 1.0 / len(expected_outcomes)
    prob_matrix = prob_matrix.fillna(uniform_prob)

    return prob_matrix


def get_cached_probability_matrix(df: pd.DataFrame, instrument_key: str, date_str: str, strategy_params: dict):
    """Retrieves a cached matrix, or builds and caches a new one if missing/updated."""
    
    # Create a unique hash based on your strategy parameters
    param_string = json.dumps(strategy_params, sort_keys=True)
    param_hash = hashlib.md5(param_string.encode()).hexdigest()[:6]
    
    safe_key = instrument_key.replace("|", "_").replace(" ", "_")
    cache_path = MATRIX_CACHE_DIR / f"{safe_key}_{date_str}_{param_hash}.joblib"
    
    if cache_path.exists():
        return joblib.load(cache_path)
        
    # Build it if it doesn't exist
    matrix = generate_probability_matrix(df)
    
    if matrix is not None:
        joblib.dump(matrix, cache_path)
        
    return matrix


prtf = Portfolio(funds=Funds(starting_capital=100000))

trader = ana.BulkSimulator(
    strategy=strat,
    portfolio=prtf,
    broker=ana.SimBroker(portfolio=prtf),
)
trader.strategy.buy_conditon = buy_signal
trader.strategy.sell_condition = sell_signal
trader.strategy.buy_constraints = buy_cons
trader.strategy.sell_constraints = sell_cons
trader.set_processor(
    partial(
        procedure,
        trader=trader,
        strategy=trader.strategy,
        buy_condition=buy_signal,
        sell_condition=sell_signal,
    )
)
args = setup_cli()
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

trader.run()
