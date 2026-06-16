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
SUPERT = "SUPERT"
ADXR = "ADXR"

MATRIX_CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache" / "matrices"
MATRIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(
    "--------------------SIMULATOR--------------------".center(
        shutil.get_terminal_size().columns
    )
)

DEFAULT_BUY_PARAMS = {
    "adxr_min": 16,
    "adxr_max": 25,
    "dmi_gap_ce": 10,
    "dmi_gap_pe": 11,
    "rsi_ce_min": 59,
    "rsi_pe_max": 37
}
def buy_signal(df, params=None, **kwargs):
    """Executes trades strictly based on the 30-Day Probability Matrix."""
    if params is None:
        params = DEFAULT_BUY_PARAMS

    try:
        bucket: Bucket = kwargs.get("bucket", None)
        probability_matrix = getattr(bucket, "probability_matrix", None)

        # Guard against a missing matrix (e.g., first few days of simulation)
        if probability_matrix is None or probability_matrix.empty:
            return pd.Series(None, index=df.index, dtype=object)

        # 1. Ask the central engine: What is the exact State of every minute today?
        df['State'] = _assign_market_states(df, params=params)

        # 2. Convert the states into a binary dummy matrix
        state_matrix = pd.get_dummies(df['State']).reindex(
            columns=probability_matrix.columns, fill_value=0
        ).to_numpy()

        # 3. Matrix Multiplication: State (N, 6) @ Matrix.T (6, 3) = Probabilities (N, 3)
        prob_output = state_matrix @ probability_matrix.T.to_numpy()

        # Extract the specific probabilities into the DataFrame
        df['Prob_DOWN'] = prob_output[:, 0]
        df['Prob_FLAT'] = prob_output[:, 1]
        df['Prob_UP']   = prob_output[:, 2]

        # ---------------------------------------------------------
        # THE EXECUTION TRIGGERS
        # ---------------------------------------------------------
        CONFIDENCE_THRESHOLD = 0.12
        # Optional: Master Volatility Gatekeeper. Even if probability is high, 
        # we might only want to trade if ADXR shows the market is actually moving.
        adxr_safe = (params["adxr_min"] < df["ADXR"]) & (df["ADXR"] < params["adxr_max"])

        ce_signal = (df['Prob_UP'] > CONFIDENCE_THRESHOLD) & adxr_safe
        pe_signal = (df['Prob_DOWN'] > CONFIDENCE_THRESHOLD) & adxr_safe

        signal_col = pd.Series(None, index=df.index, dtype=object)
        signal_col[ce_signal] = "CE"
        signal_col[pe_signal] = "PE"

        return signal_col

    except Exception as e:
        logger.exception(f"Error generating matrix buy signals: {e}")
        return pd.Series(None, index=df.index, dtype=object)


def sell_signal(df, params=None, enable_trailing=False, **kwargs):
    """Generates square-off signals based on opposing market probabilities."""
    if params is None:
        params = DEFAULT_BUY_PARAMS

    try:
        bucket: Bucket = kwargs.get("bucket", None)
        probability_matrix = getattr(bucket, "probability_matrix", None)
        
        # 1. HARD RULE: End of dataset square-off (Guarantees no orphaned positions overnight)
        square_off = pd.Series(df.index == df.index[-1], index=df.index)

        # 2. HARD RULE: Trailing stop condition (If enabled, acts as an absolute disaster-stop)
        if enable_trailing:
            ce_trend_break = df["close"] < df[SUPERT]
            pe_trend_break = df["close"] > df[SUPERT]
        else:
            ce_trend_break = False
            pe_trend_break = False

        # 3. PROBABILISTIC EXITS
        if probability_matrix is not None and not probability_matrix.empty:
            # Re-evaluate the state using the central engine
            df['State'] = _assign_market_states(df, params=params)
            
            # Matrix Multiplication (Same as buy_signal)
            state_matrix = pd.get_dummies(df['State']).reindex(
                columns=probability_matrix.columns, fill_value=0
            ).to_numpy()

            prob_output = state_matrix @ probability_matrix.T.to_numpy()
            
            prob_down = prob_output[:, 0]
            prob_up   = prob_output[:, 2]

            EXIT_CONFIDENCE = 0.70 
            
            ce_prob_exit = prob_down > EXIT_CONFIDENCE  # If holding CE, bail if high chance of DOWN
            pe_prob_exit = prob_up > EXIT_CONFIDENCE    # If holding PE, bail if high chance of UP
            
        else:
            # Fallback if the matrix is missing during early warm-up days
            ce_prob_exit = (df["close"] > df[SUPERT]) & (df[SUPERT] == df[SUPERT].shift(1))
            pe_prob_exit = (df["close"] < df[SUPERT]) & (df[SUPERT] == df[SUPERT].shift(1))

        ce_signal = ce_prob_exit | ce_trend_break 
        pe_signal = pe_prob_exit | pe_trend_break 

        signal_col = pd.Series(None, index=df.index, dtype=object)
        signal_col[ce_signal] = "CE"
        signal_col[pe_signal] = "PE"        
        signal_col[square_off] = "SQUARE_OFF" 

        return signal_col

    except Exception as e:
        logger.exception(f"Error generating matrix sell signals: {e}")
        return pd.Series(None, index=df.index, dtype=object)


def buy_cons(**kwargs):
    target = kwargs.get("bucket", None)
    return True if target.open_position is None else False


def sell_cons(**kwargs):
    target = kwargs.get("bucket", None)
    return False if target.open_position is None else True


def _worker(df: pd.DataFrame, study: ta.Study, **kwargs):
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
    signal_hit = spot_row.sell_signal in [side, "SQUARE_OFF"]

    if not (stoploss_hit or target_hit or signal_hit):
        return


    if stoploss_hit:
        base_price = max(0.05, min(stoploss, row.open))
        exec_time, remark, buffer_points = row.timestamp, "SL", 5.00
    elif target_hit:
        base_price = max(target, row.open)
        exec_time, remark, buffer_points = row.timestamp, "Target", 2.00
    else:
        # This fallback elegantly handles the final candle of the day
        base_price = max(0.05, row.next_open if pd.notna(row.next_open) else row.close)
        exec_time = row.next_time if pd.notna(row.next_time) else row.timestamp
        remark, buffer_points = "Square off" if spot_row.sell_signal == "SQUARE_OFF" else "Signal", 2.00

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
    
    report.sell_price = buffered_sell_price 
    
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
    logger.info(f"lot_size = {option.lot_size}, qty = {qty}, balance= {trader.portfolio.funds.available_margin}")

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
    bucket.open_position.target = exec_price + (5 * row.ATR)

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

    spot_df = spot.load_historical_df(ustox, lookback=30)
    ce_df = call_option.load_historical_df(ustox, lookback=1)
    pe_df = put_option.load_historical_df(ustox, lookback=1)

    def secure_prep(df):
        if df is None or df.empty:
            logger.info("secure prep failed")
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
    spot_df = _worker(spot_df, strategy.indicators, **kwargs)
    spot_df = spot_df.rename(
        columns={
            "SUPERT_14_2.0":"SUPERT",
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
            "VWAP_D": "VWAP",
        }
    )

    # Guard clause in case indicator generation fails or data is missing
    if spot_df.empty or ce_df.empty or pe_df.empty:
        return False

    target_date = pd.Timestamp(bucket.date)

    historical_training_data = spot_df[spot_df.index < target_date]
    spot_df_new = spot_df[spot_df.index.normalize() == target_date]
    ce_df_new = ce_df[ce_df.index.normalize() == target_date].reindex(spot_df_new.index, method="ffill")
    pe_df_new = pe_df[pe_df.index.normalize() == target_date].reindex(spot_df_new.index, method="ffill")
    if spot_df_new.empty or ce_df_new.empty or pe_df_new.empty:
        logger.info(f"empty df - spot: {spot_df_new.empty} | ce: {ce_df_new.empty} | pe: {pe_df_new.empty}")
        return False

    prob_matrix = get_cached_probability_matrix(
        df=historical_training_data,
        instrument_key=spot.key,
        date_str=str(bucket.date),
        strategy_params={"lookback": 30, "supertrend": 2.0, "adxr": 16} 
    )
    
    # Bypass dataclass freeze lock if Bucket is frozen
    bucket.probability_matrix= prob_matrix

    for opt, df in zip((call_option, put_option), (ce_df, pe_df)):
        df = df.reset_index()
        df["next_open"] = df["open"].shift(-1)
        df["next_time"] = df["timestamp"].shift(-1)
        df["ATR"] = df.ta.atr()
        opt.historical_df = df.reset_index(drop=True)
    

    ana.Strategy.gen_signals(spot_df_new,buy_cond=trader.strategy.buy_conditon, sell_cond=trader.strategy.sell_condition,bucket=bucket,**kwargs)


    spot.historical_df = spot_df_new.reset_index()

    # 6. Finalize Option specific logic

    logger.info("Bucket Prepared")        
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
        logging.info("Bucket not prepared")
        return False
        
    try:
        for row_ce, row_pe, row_spot in zip(
            call_option.historical_df.itertuples(),
            put_option.historical_df.itertuples(),
            spot.historical_df.itertuples(),
        ):

            if bucket.open_position is None:
                if row_spot.buy_signal == "CE" and execute_buy(
                    spot_row=row_spot,
                    row=row_ce,
                    option=call_option,
                    trader=trader,
                    strategy=strategy,
                    bucket=bucket,
                    **kwargs,
                ) == 1:
                    continue
                    
                if row_spot.buy_signal == "PE" and execute_buy(
                    spot_row=row_spot,
                    row=row_pe,
                    option=put_option,
                    trader=trader,
                    strategy=strategy,
                    bucket=bucket,
                    **kwargs,
                ) == 1:
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
        if trader.portfolio.funds.available_margin < 1000:
            logger.info("Trader Bankrupt!")
            break

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

def _assign_market_states(df: pd.DataFrame, params: dict = None) -> pd.Series:
    if params is None:
        params = DEFAULT_BUY_PARAMS

    # Base Trend (Supertrend)
    trend_up = df["close"] > df["SUPERT"]
    
    # Redefining Momentum using your updated params dictionary
    dmi_gap = abs(df["DMP"] - df["DMN"])
    bullish_momentum = (df["RSI"] > params["rsi_ce_min"]) & (dmi_gap > params["dmi_gap_ce"])
    bearish_momentum = (df["RSI"] < params["rsi_pe_max"]) & (dmi_gap > params["dmi_gap_pe"])
    price_above_vwap = df["close"] > df["VWAP"]

    states = np.where(trend_up & bullish_momentum & price_above_vwap, 'Strong Bull',
             np.where(trend_up & ~bullish_momentum, 'Weak Bull',
             np.where(~trend_up & bearish_momentum, 'Strong Bear',
             np.where(~trend_up & ~bearish_momentum, 'Weak Bear', 
             'Sideways'))))
    
    return pd.Series(states, index=df.index, name="State")

def _assign_future_outcomes(df: pd.DataFrame, buffer_pts: float = 10.0) -> pd.Series:
    
    next_close = df['close'].shift(-1)
    
    outcomes = np.where(next_close > df['close'] + buffer_pts, 'UP', 
               np.where(next_close < df['close'] - buffer_pts, 'DOWN', 'FLAT'))
    
    return pd.Series(outcomes, index=df.index, name="Outcome")


def generate_probability_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Builds the probability matrix from the 30-day historical data."""
    if df.empty or len(df) < 2:
        return None

    df = df.copy()
    
    # 1. Ask the central engine what the states were over the last 30 days
    df['State'] = _assign_market_states(df)
    
    # 2. Assign the outcomes (UP, DOWN, FLAT)
    df['Outcome'] = _assign_future_outcomes(df)
    df = df.iloc[:-1]

    # 3. Build the Matrix
    tally = pd.crosstab(df['Outcome'], df['State'])
    expected_outcomes = ['DOWN', 'FLAT', 'UP']  
    tally = tally.reindex(expected_outcomes, fill_value=0)
    
    prob_matrix = tally.div(tally.sum(axis=0), axis=1)
    
    uniform_prob = 1.0 / len(expected_outcomes)
    prob_matrix = prob_matrix.fillna(uniform_prob)

    return prob_matrix


def get_cached_probability_matrix(df: pd.DataFrame, instrument_key: str, date_str: str, strategy_params: dict):
    """Retrieves a cached matrix, or builds and caches a new one if missing/updated."""
    
    param_string = json.dumps(strategy_params, sort_keys=True)
    param_hash = hashlib.md5(param_string.encode()).hexdigest()[:6]
    
    safe_key = instrument_key.replace("|", "_").replace(" ", "_")
    cache_path = MATRIX_CACHE_DIR / f"{safe_key}_{date_str}_{param_hash}.joblib"
    
    if cache_path.exists():
        return joblib.load(cache_path)
        
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
