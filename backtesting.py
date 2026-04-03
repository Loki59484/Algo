from collections import defaultdict
from functools import partial
from datetime import datetime
import joblib
from pathlib import Path
import pandas_ta as ta
from tqdm import tqdm
import shutil
from datetime import time
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core import anatomy as ana
from core.datatypes import *
from core import upstox_methods as ustox
from ui import tui

INSTRUMENT_CACHE = ustox.DATA_DIR / "cache" / "instruments_cache.joblib"

print(
    "-------------TRADING SIMULATOR-------------".center(
        shutil.get_terminal_size().columns
    )
)

# SETTING UP TRADER
trader = ana.Trader()
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
trader.strategy = strat

# LOADING INSTRUMENTS

files = list((ustox.DATA_DIR / "historical").rglob("*.parquet"))
files = [file for file in files if "INDEX" not in str(file)]
files.sort()

if INSTRUMENT_CACHE.exists():
    print("Loading instruments from cache...", end="\r")
    insts = joblib.load(INSTRUMENT_CACHE)
    print(f"Loaded {len(insts)} instruments from cache.", end="\r")
else:
    insts = Instrument.load_multiple(source=files, lookback=2)
    joblib.dump(insts, INSTRUMENT_CACHE)
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

# DEFINING BUY-SELL PARAMETERS


def buy_signal(df, **kwargs):
    cond_1 = (df["close"] > df["SUPERT_14_2.0"]) & (
        df["SUPERT_14_2.0"] > df["SUPERT_14_2.0"].shift(1)
    )
    cond_2 = (25 < (df["ADXR_14_2"])) & ((df["ADXR_14_2"]) < 30)
    cond_3 = df["DMP_14"] > df["DMN_14"]
    cond_4 = abs(df["DMP_14"] - df["DMN_14"]) > 1
    cond_6 = df["SUPERT_14_2.0"] < df["VWAP_D"]
    return (cond_1) & (cond_2) & (cond_3) & (cond_4)  # & (cond_6)


def sell_signal(df, **kwargs):
    cond_1 = (df["close"] > df["SUPERT_14_2.0"]) & (
        df["SUPERT_14_2.0"] == df["SUPERT_14_2.0"].shift(1)
    )
    square_off = pd.Series(df.index == df.index[-1], index=df.index)
    return cond_1 | square_off


def buy_cons(**kwargs):
    target = kwargs.get("instrument", None)
    return not target.position.open


def sell_cons(**kwargs):
    target = kwargs.get("instrument", None)
    return target.position.open


def _worker(df: pd.DataFrame, study: ta.Study, kwargs):
    return ana.Strategy.apply_study(df, study=study, **kwargs)


def procedure(trader: ana.Trader, strategy: ana.Strategy, **kwargs):

    flat_subjects = [
        subject for bucket in trader.buckets for subject in bucket.legs.values()
    ]
    raw_dfs = [subject.to_dataframe() for subject in flat_subjects]
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
        truncated_df = enriched_df[enriched_df.index.date >= subject.date]
        truncated_df = truncated_df.rename(
            columns={
                "SUPERT_14_2.0": "SUPERT_14_2",
                "SUPERTl_14_2.0": "SUPERTs_14_2",
                "SUPERTs_14_2.0": "SUPERTs_14_2",
            }
        )
        subject.historical_df = truncated_df.reset_index()
    logger.info("Technical analysis completed")
    trader.buckets.sort(key=lambda b: b.date)
    for bucket in trader.buckets:
        call_option = bucket.legs.get("CE")
        put_option = bucket.legs.get("PE")
        app = tui.TradingTUI(bucket=bucket, simulate=True)
        app.run()
        exit()
        if call_option is None or put_option is None:
            logger.warning(f"NoneType option found for bucket {bucket.date}")
            continue
        for row_ce, row_pe in zip(
            call_option.historical_df.itertuples(),
            put_option.historical_df.itertuples(),
        ):
            if time(11, 00) > row_ce.timestamp.time() > time(10, 00) or time(
                14, 00
            ) > row_ce.timestamp.time() > time(13, 00):
                continue

            # BUY SELL CONSTRAINTS
            if bucket.open_position is None:
                if row_ce.buy_signal:
                    call_buy_cons = (
                        strategy.buy_constraints(instrument=call_option, **kwargs)
                        if strategy.buy_constraints is not None
                        else True
                    )
                    if call_buy_cons:
                        trader.execute_buy(
                            tick=row_ce,
                            instrument=call_option,
                            funds=trader.portfolio.funds,
                        )
                        bucket.open_position = call_option.position
                        bucket.open_position.open_trade.stoploss = row_ce.close - (
                            1.5 * row_ce.ATRr_14
                        )
                        continue

                if row_pe.buy_signal:
                    put_buy_cons = (
                        strategy.buy_constraints(instrument=put_option, **kwargs)
                        if strategy.buy_constraints is not None
                        else True
                    )
                    if put_buy_cons:
                        trader.execute_buy(
                            tick=row_pe,
                            instrument=put_option,
                            funds=trader.portfolio.funds,
                        )
                        bucket.open_position = put_option.position
                        bucket.open_position.open_trade.stoploss = row_pe.close - (
                            0.2 * row_pe.ATRr_14
                        )
                        continue

            else:
                stoploss = bucket.open_position.open_trade.stoploss
                if bucket.open_position.key == call_option.key:
                    call_stoploss_hit = (
                        row_ce.low <= stoploss if stoploss is not None else False
                    )
                    if row_ce.sell_signal or call_stoploss_hit:
                        call_sell_cons = (
                            strategy.sell_constraints(
                                instrument=call_option, **kwargs
                            )
                            if strategy.sell_constraints is not None
                            else True
                        )
                        if call_sell_cons or call_stoploss_hit:
                            trader.execute_sell(
                                tick=row_ce,
                                trade=bucket.open_position.open_trade,
                                instrument=call_option,
                                exit_price=stoploss if call_stoploss_hit else None,
                                funds=trader.portfolio.funds,
                                remark="SL" if call_stoploss_hit else "-",
                            )
                            bucket.open_position = None

                elif bucket.open_position.key == put_option.key:
                    put_stoploss_hit = (
                        row_pe.low <= stoploss if stoploss is not None else False
                    )
                    if row_pe.sell_signal or put_stoploss_hit:
                        put_sell_cons = (
                            strategy.sell_constraints(
                                instrument=put_option, **kwargs
                            )
                            if strategy.sell_constraints is not None
                            else True
                        )
                        if put_sell_cons or put_stoploss_hit:
                            trader.execute_sell(
                                tick=row_pe,
                                trade=bucket.open_position.open_trade,
                                instrument=put_option,
                                exit_price=stoploss if put_stoploss_hit else None,
                                funds=trader.portfolio.funds,
                                remark="SL" if put_stoploss_hit else "-",
                            )
                            bucket.open_position = None
        trader.portfolio.funds.starting_capital = (
            trader.portfolio.funds.available
        ) = trader.portfolio.funds.total

trader.strategy.buy_conditon = buy_signal
trader.strategy.sell_condition = sell_signal
trader.strategy.buy_constraints = buy_cons
trader.strategy.sell_constraints = sell_cons
trader.strategy.custom_test = partial(
    procedure,
    trader=trader,
    strategy=trader.strategy,
    buy_condition=buy_signal,
    sell_condition=sell_signal,
)

report = trader.test()
logger.info("\n-------------BACKTESTING COMPLETE-------------")

report_file = (
    ustox.LOG_DIR
    / "reports"
    / f"trade_report_{datetime.now().strftime("%d%m%Y_%H%M%S")}.csv"
)
report_file.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(report_file)
