from collections import defaultdict
from functools import partial
import datetime as dt
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

# ustox = UpstoxClient()
INSTRUMENT_CACHE = DATA_DIR / "cache" / "instruments_cache.joblib"

print(
    "--------------------SIMULATOR--------------------".center(
        shutil.get_terminal_size().columns
    )
)

# DEFINING BUY-SELL PARAMETERS


# DEFINING BUY-SELL PARAMETERS
def buy_signal(df, **kwargs):
    try:
        cond_1 = (df["close"] > df["SUPERT_14_2.0"]) & (
            df["SUPERT_14_2.0"] > df["SUPERT_14_2.0"].shift(1)
        )
        cond_2 = (25 < (df["ADXR_14_2"])) & ((df["ADXR_14_2"]) < 30)
        cond_3 = df["DMP_14"] > df["DMN_14"]
        cond_4 = (abs(df["DMP_14"] - df["DMN_14"]) > 2) & (
            abs(df["DMP_14"] - df["DMN_14"]) <= 10
        )
        cond_6 = df["SUPERT_14_2.0"] < df["VWAP_D"]
        cond_7 = df["RSI_14"] < 61
        return (cond_1) & (cond_2) & (cond_3) & (cond_4) & (cond_6) & cond_7
    except Exception as e:
        logger.exception(e)

def sell_signal(df, **kwargs):
    cond_1 = (df["close"] > df["SUPERT_14_2.0"]) & (
        df["SUPERT_14_2.0"] == df["SUPERT_14_2.0"].shift(1)
    )
    square_off = pd.Series(df.index == df.index[-1], index=df.index)
    return cond_1 | square_off


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

    flat_subjects = [
        subject for bucket in trader.buckets for subject in bucket.legs.values()
    ]
    raw_dfs = [
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
                executor.map(worker_func, raw_dfs),
                total=len(raw_dfs),
                desc="Calculating TA & Signals",
            )
        )

    # EXTRACT TRADING DAY DATA, DROPPING WARM UP CANDLES
    for subject, enriched_df in zip(flat_subjects, finished_dfs):
        try:
            if enriched_df.empty:
                continue
            truncated_df = enriched_df[enriched_df.index.date >= subject.date]
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
            subject.historical_df = truncated_df.reset_index()
        except Exception as e:
            breakpoint(header=f"{e}")
    logger.info("Technical analysis completed")
    trader.buckets.sort(key=lambda b: b.date)

    def execute_sell(row: tuple, option, trader, stoploss, target=None):
        stoploss_hit = row.low <= stoploss if stoploss is not None else False
        target_hit = row.high >= target if target is not None else False
        if stoploss_hit or target_hit or row.sell_signal:
            sell_cons = (
                strategy.sell_constraints(bucket=bucket, **kwargs)
                if strategy.sell_constraints is not None
                else True
            )
            if stoploss_hit or target_hit or sell_cons:
                status = trader.broker.sell_order(
                    key=option.key,
                    qty=trader.portfolio.report[-1].Buy_qty,
                    price=(
                        stoploss
                        if stoploss_hit
                        else target if target_hit else row.close
                    ),
                )
                if status == -1:
                    logger.error("Sell order not placed.")
                    return

                report: Trade = trader.portfolio.report[-1]
                report.Sell_conditions = row._asdict()
                report.Sell_qty = report.Buy_qty
                report.Sell_timestamp = row.timestamp
                report.Sell_price = stoploss if stoploss_hit else row.close
                report.Remark = "SL" if stoploss_hit else "T" if target_hit else "-"
                report.Movement = report.Sell_price - report.Buy_price
                report.PnL = (report.Sell_price * report.Sell_qty) - (
                    report.Buy_price * report.Buy_qty
                )
                report.total = trader.portfolio.funds.total
                bucket.open_position = None
                logger.info("Trade executed successfully for sell side.")

    def execute_buy(row: tuple, option: Instrument, trader: ana.Trader):
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

            qty = trader.calculate_units(
                close=row.close,
                lot_size=option.lot_size,
            )
            if qty == 0:
                return -1
            status = trader.broker.buy_order(
                key=option.key,
                price=row.close,
                qty=qty,
            )

            if status is None:
                return -1
            pos, ord_id = status[0], status[1]
            bucket.open_position = pos
            bucket.open_position.stoploss = row.close - (0.5 * row.ATR)
            bucket.open_position.target = None  # row.close + (5 * row.ATR)
            trader.portfolio.report.append(
                Trade(
                    Trade_id=ord_id,
                    Instrument_key=option.key,
                    Buy_timestamp=row.timestamp,
                    Side=option.type,
                    Buy_price=row.close,
                    Buy_qty=qty,
                    Buy_conditions=row._asdict(),
                )
            )

            logger.info("Trade executed successfully for buy side.")
            return 1

    for bucket in trader.buckets:
        call_option = bucket.legs.get("CE")
        put_option = bucket.legs.get("PE")

        if call_option is None or put_option is None:
            logger.warning(f"NoneType option found for bucket {bucket.date}")
            continue
        try:
            if not hasattr(call_option,"historical_df") or not hasattr(put_option,"historical_df"):
                continue
            for row_ce, row_pe in zip(
                call_option.historical_df.itertuples(),
                put_option.historical_df.itertuples(),
            ):
                if bucket.open_position is None:
                    if row_ce.buy_signal:
                        status = execute_buy(row_ce, call_option, trader)
                        if status == 1:
                            continue

                    if row_pe.buy_signal:
                        status = execute_buy(row_pe, put_option, trader)
                        if status == 1:
                            continue
                else:
                    stoploss = bucket.open_position.stoploss
                    target = bucket.open_position.target
                    if bucket.open_position.instrument_token == call_option.key:
                        execute_sell(row_ce, call_option, trader, stoploss, target)
                    elif bucket.open_position.instrument_token == put_option.key:
                        execute_sell(row_pe, put_option, trader, stoploss, target)
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

prtf = Portfolio(funds=Funds(starting_capital=300000))

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
files = list((DATA_DIR / "historical" / args.bulk[0]).rglob("*.parquet"))
files = [file for file in files if "INDEX" not in str(file)]
files.sort()

if INSTRUMENT_CACHE.exists():
    print("Loading instruments from cache...", end="\r")
    insts = joblib.load(INSTRUMENT_CACHE)
    print(f"Loaded {len(insts)} instruments from cache.", end="\r")
else:
    insts = Instrument.load_multiple(client=None, source=files, lookback=0)
    #joblib.dump(insts, INSTRUMENT_CACHE)
insts_dict = {(item.key, item.date): item for item in insts}
trader.add_instrument(insts_dict)
# CREATE BUCKETS FOR EACH DAY
daily_buckets = defaultdict(dict)

for instrument in tqdm(
    trader.instruments.values(), desc="Filtering instruments", leave=False
):
    leg_type = "CE" if "CE" in instrument.type else "PE"
    daily_buckets[instrument.date][leg_type] = instrument
for trade_date, legs in tqdm(
    daily_buckets.items(), desc="Loading Buckets", leave=False
):
    bucket = Bucket(trade_date, legs=legs)
    trader.buckets.append(bucket)

trader.run()
