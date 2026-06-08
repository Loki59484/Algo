from collections import defaultdict
from types import SimpleNamespace
from functools import partial
import datetime as dt
import numpy as np
import joblib
from pathlib import Path
import pandas_ta as ta
from tqdm import tqdm
import shutil
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core import anatomy as ana
from core.datatypes import *
from core.upstox_methods import *

# from core.planner import Planner
ustox = UpstoxClient()

print(
    "--------------------SIMULATOR--------------------".center(
        shutil.get_terminal_size().columns
    )
)

# DEFINING BUY-SELL PARAMETERS

def generate_prob_matrix(client:UpstoxClient,instrument:Instrument,lookback, force_rebuild=False):
    back_data = Instrument.load_previous(client=client,ins=instrument,prev_trading_day=instrument.date-timedelta(days=lookback), isexpired=True)
    return back_data



def buy_signal(df, probability_matrix, **kwargs):
    try:
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
    ce_signal = (df["close"] > df["SUPERT_14_2.0"]) & (
        df["SUPERT_14_2.0"] == df["SUPERT_14_2.0"].shift(1)
    ) | square_off
    pe_signal = (df["close"] < df["SUPERT_14_2.0"]) & (
        df["SUPERT_14_2.0"] == df["SUPERT_14_2.0"].shift(1)
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


def procedure(trader: ana.Trader, strategy: ana.Strategy, **kwargs):
    # planner : Planner = Planner()
    # args = SimpleNamespace()
    # plan = planner.create()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.colheader_justify", "center")

    flat_subjects = [
        subject
        for bucket in trader.buckets
        for subject in bucket.legs.values()
        if subject
    ]
    subject_spots = [bucket.spot for bucket in trader.buckets]

    raw_spot_dfs = [
        pd.DataFrame([asdict(candle) for candle in subject.historical_candles])
        .assign(key=subject.key)
        .set_index("timestamp")
        for subject in subject_spots
    ]

    raw_option_dfs = [
        pd.DataFrame([asdict(candle) for candle in subject.historical_candles])
        .assign(key=subject.key)
        .set_index("timestamp")
        for subject in flat_subjects
    ]

    worker_func = partial(_worker, study=strategy.indicators, kwargs=kwargs)
    logger.info("Computing indicators")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        finished_dfs = list(
            tqdm(
                executor.map(worker_func, raw_spot_dfs),
                total=len(raw_spot_dfs),
                desc="Calculating TA & Signals",
            )
        )

    # EXTRACT TRADING DAY DATA, DROPPING WARM UP CANDLES
    for spot, enriched_df in zip(subject_spots, finished_dfs):
        try:
            if enriched_df.empty:
                continue
            if "key" in enriched_df.columns:
                df_key = enriched_df["key"].iloc[0]
                if spot.key != df_key:
                    raise ValueError(
                        f"CRITICAL MISMATCH: Spot key {spot.key} does not match DF key {df_key}"
                    )
            # ------------------------
            truncated_df = enriched_df[enriched_df.index.date >= spot.date]
            truncated_df = truncated_df.rename(
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
                    "EMA_200": "EMA",
                    "RSI_14": "RSI",
                }
            )
            spot.historical_df = truncated_df.reset_index()
        except Exception as e:
            breakpoint(header=f"{e}")
    logger.info("Technical analysis completed")

    for subject, df in zip(flat_subjects, raw_option_dfs):

        if "key" in df.columns:
            df_key = df["key"].iloc[0]
            if subject.key != df_key:
                raise ValueError(
                    f"CRITICAL MISMATCH: Subject key {subject.key} does not match DF key {df_key}"
                )
        # ------------------------

        subject.historical_df = df[df.index.date >= subject.date].reset_index()

    trader.buckets.sort(key=lambda b: b.date)

    def execute_sell(spot_row: tuple, row: tuple, option, trader, stoploss, target=None, side=None):
        stoploss_hit = row.low <= stoploss if stoploss is not None else False
        target_hit = row.high >= target if target is not None else False
        signal_hit = spot_row.sell_signal == side

        if stoploss_hit or target_hit or signal_hit:
            
            if pd.isna(row.next_open):
                return # End of dataset
                
            if stoploss_hit:
                # Clamp the worst-case fill to 0.05
                base_price = max(0.05, min(stoploss, row.open))
                exec_time = row.timestamp
                remark = "SL"
                buffer_points = 5.00 
            elif target_hit:
                base_price = max(target, row.open)
                exec_time = row.timestamp
                remark = "Target"
                buffer_points = 2.00
            else:
                # Clamp the open price to 0.05
                base_price = max(0.05, row.next_open if pd.notna(row.next_open) else row.close)
                exec_time = row.next_time if pd.notna(row.next_time) else row.timestamp
                remark = "Signal"
                buffer_points = 2.00

            # Clamp the limit price sent to the broker to 0.05
            buffered_sell_price = max(0.05, round(base_price - buffer_points, 2))

            # 3. Fire the Limit IOC Order
            status = trader.broker.sell_order(
                key=option.key,
                qty=trader.portfolio.report[-1].Buy_qty,
                price=buffered_sell_price, # Sent to broker as the Limit Price
            )
            
            if status == -1:
                logger.error(f"Sell order not placed for {option.key}. Gapped down past limit.")
                return

            # 4. Log the trade (assuming the exchange matched us at the base_price)
            report: Trade = trader.portfolio.report[-1]
            report.Sell_conditions = spot_row._asdict() # Log Spot conditions, not option conditions
            report.Sell_qty = report.Buy_qty
            report.Sell_timestamp = exec_time 
            report.Sell_price = base_price    # Log the actual executed market price
            report.Remark = remark
            report.Movement = report.Sell_price - report.Buy_price
            report.PnL = (report.Sell_price * report.Sell_qty) - (report.Buy_price * report.Buy_qty)
            report.total = trader.portfolio.funds.total
            
            bucket.open_position = None
            logger.info(f"Trade executed successfully for sell side ({remark}).")

    def execute_buy(spot_row,row: tuple, option: Instrument, trader: ana.Trader):
        if (
            dt.time(12, 00)
            > row.timestamp.time()
            > dt.time(10, 00)  # Block 10 AM to 12 PM
            or dt.time(14, 00) > row.timestamp.time() > dt.time(13, 00)
            or row.timestamp.time() > dt.time(15, 0)
        ):
            return -1
        buy_cons = (
            strategy.buy_constraints(bucket=bucket, **kwargs)
            if strategy.buy_constraints is not None
            else True
        )
        if buy_cons:
            
            exec_price = row.next_open if pd.notna(row.next_open) else row.close
            exec_time = row.next_time if pd.notna(row.next_time) else row.timestamp 
            if exec_price < 50.0:
                return -1 # Ignore cheap, decaying lotto options
            
            qty = trader.calculate_units(
                close=exec_price,
                lot_size=option.lot_size,
            )
            logger.info(f"lot_size = {option.lot_size}, qty = {qty}")
            if qty == 0:
                return -1
            status = trader.broker.buy_order(
                key=option.key,
                price=exec_price,
                qty=qty,
                order_type='LIMIT'
            )

            if status is None:
                return -1
            pos, ord_id = status[0], status[1]
            bucket.open_position = pos
            bucket.open_position.stoploss = max(0.05, exec_price - (1.5 * row.ATR))
            bucket.open_position.target =  exec_price + (22  * row.ATR)
            trader.portfolio.report.append(
                Trade(
                    Trade_id=ord_id,
                    Instrument_key=option.key,
                    Buy_timestamp=exec_time,
                    Side=option.type,
                    Buy_price=exec_price,
                    Buy_qty=qty,
                    Buy_conditions=spot_row._asdict(),
                )
            )

            logger.info("Trade executed successfully for buy side.")
            return 1

    for bucket in trader.buckets:
        call_option = bucket.legs.get("CE")
        put_option = bucket.legs.get("PE")
        call_option.historical_df["next_open"] = call_option.historical_df["open"].shift(-1)
        call_option.historical_df["next_time"] = call_option.historical_df["timestamp"].shift(-1)
        put_option.historical_df["next_open"] = put_option.historical_df["open"].shift(-1)
        put_option.historical_df["next_time"] = put_option.historical_df["timestamp"].shift(-1)
        call_option.historical_df["ATR"] = call_option.historical_df.ta.atr()
        put_option.historical_df["ATR"] = put_option.historical_df.ta.atr()
        spot = bucket.spot
                
        if call_option is None or put_option is None:
            logger.warning(f"NoneType option found for bucket {bucket.date}")
            continue
        try:
            if (
                not hasattr(spot, "historical_df")
                or not hasattr(put_option, "historical_df")
                or not hasattr(spot, "historical_df")
            ):
                logger.info(
                    f"Missing historical_df for bucket {bucket.date}. Skipping."
                )
                continue

            for row_ce, row_pe, row_spot in zip(
                call_option.historical_df.itertuples(),
                put_option.historical_df.itertuples(),
                spot.historical_df.itertuples(),
            ):
                if bucket.open_position is None:
                    if row_spot.buy_signal == "CE":
                        status = execute_buy(row_spot,row_ce, call_option, trader)
                        if status == 1:
                            continue

                    if row_spot.buy_signal == "PE":
                        status = execute_buy(row_spot,row_pe, put_option, trader)
                        if status == 1:
                            continue
                else:
                    stoploss = bucket.open_position.stoploss
                    target = bucket.open_position.target
                    if bucket.open_position.instrument_token == call_option.key:
                        execute_sell(
                            row_spot,
                            row_ce,
                            call_option,
                            trader,
                            stoploss,
                            target,
                            side="CE",
                        )
                    elif bucket.open_position.instrument_token == put_option.key:
                        execute_sell(
                            row_spot,
                            row_pe,
                            put_option,
                            trader,
                            stoploss,
                            target,
                            side="PE",
                        )
        except Exception as e:
            breakpoint(header=f"{e}")
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
# files = [file for file in files if "INDEX" not in str(file)]

files.sort()

if INSTRUMENT_CACHE.exists() and not args.no_cache:
    print("Loading instruments from cache...", end="\r")
    insts = joblib.load(INSTRUMENT_CACHE)
    print(f"Loaded {len(insts)} instruments from cache.", end="\r")
else:
    insts = Instrument.load_multiple(client=None, source=files, lookback=0)
    joblib.dump(insts, INSTRUMENT_CACHE)
insts_dict = {(item.key, item.date): item for item in insts}
trader.add_instrument(insts_dict)
# CREATE BUCKETS FOR EACH DAY
daily_buckets = defaultdict(dict)
for instrument in tqdm(
    trader.instruments.values(), desc="Filtering instruments", leave=False
):
    breakpoint()
    leg_type = (
        "CE"
        if "CE" in instrument.type
        else "PE" if "PE" in instrument.type else "INDEX"
    )
    daily_buckets[instrument.date][leg_type] = instrument
for trade_date, legs in tqdm(
    daily_buckets.items(), desc="Loading Buckets", leave=False
):
    options = {k: v for k, v in legs.items() if k in ["CE", "PE"]}
    bucket = Bucket(trade_date, legs=options, spot=legs.get("INDEX", None))
    trader.buckets.append(bucket)

trader.run()
