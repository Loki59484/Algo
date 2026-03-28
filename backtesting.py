from collections import defaultdict
import shutil
from pathlib import Path
import pandas_ta as ta
from tqdm import tqdm
import sys

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core import anatomy as ana
from core.datatypes import *
from core.upstox_func import DATA_DIR

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
    ]
)
trader.strategy = strat

# LOADING INSTRUMENTS
files = list((DATA_DIR / "historical").rglob("*.parquet"))
files = [file for file in files if "INDEX" not in str(file)]
files.sort()
insts = Instrument.load_multiple(source=files, lookback=1)
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
    bucket = ana.Bucket(trade_date, legs=legs, margin=300000)
    trader.buckets.append(bucket)

# DEFINING BUY-SELL PARAMETERS


def buy_signal(df, **kwargs):
    cond_1 = (df["close"] > df["SUPERT_14_2.0"]) & (
        df["SUPERT_14_2.0"] > df["SUPERT_14_2.0"].shift(1)
    )
    cond_2 = (25 < (df["ADXR_14_2"])) & ((df["ADXR_14_2"]) < 30)
    cond_3 = df["DMP_14"] > df["DMN_14"]
    cond_4 = abs(df["DMP_14"] - df["DMN_14"]) > 1
    return (cond_1) & (cond_2) & (cond_3) & (cond_4)


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
        subject.historical_df = enriched_df.loc[subject.date :].reset_index()
    logger.info("Technical analysis completed")
    trader.buckets.sort(key=lambda b: b.date)
    for bucket in tqdm(trader.buckets, desc="Testing buckets", position=0, leave=True):
        call_option = bucket.legs.get("CE")
        put_option = bucket.legs.get("PE")

        if call_option is None or put_option is None:
            logger.warning(f"NoneType option found for bucket {bucket.date}")
            continue

        for row_ce, row_pe in zip(
            call_option.historical_df.itertuples(),
            put_option.historical_df.itertuples(),
        ):
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
                            tick=row_ce, instrument=call_option, margin=bucket.margin
                        )
                        bucket.open_position = call_option.position
                        bucket.open_position.open_trade.stoploss = row_ce.close - (1.5*row_ce.ATRr_14)
                        continue

                if row_pe.buy_signal:
                    put_buy_cons = (
                        strategy.buy_constraints(instrument=put_option, **kwargs)
                        if strategy.buy_constraints is not None
                        else True
                    )
                    if put_buy_cons:
                        trader.execute_buy(
                            tick=row_pe, instrument=put_option, margin=bucket.margin
                        )
                        bucket.open_position = put_option.position
                        bucket.open_position.open_trade.stoploss = row_pe.close - (1.5*row_pe.ATRr_14)
                        continue

            else:
                stoploss = bucket.open_position.open_trade.stoploss
                if bucket.open_position.key == call_option.key:
                    call_stoploss_hit = (
                        row_ce.low <= stoploss
                        if stoploss is not None
                        else False
                    )
                    if row_ce.sell_signal or call_stoploss_hit:
                        call_sell_cons = (
                            strategy.sell_constraints(instrument=call_option, **kwargs)
                            if strategy.sell_constraints is not None
                            else True
                        )
                        if call_sell_cons or call_stoploss_hit:
                            trader.execute_sell(
                                tick=row_ce,
                                trade=bucket.open_position.open_trade,
                                instrument=call_option,
                                exit_price = stoploss if call_stoploss_hit else None,
                                remark = "Stoploss hit" if call_stoploss_hit else ''
                            )
                            bucket.open_position = None

                elif bucket.open_position.key == put_option.key:
                    put_stoploss_hit = (
                        row_pe.low <= stoploss
                        if stoploss is not None
                        else False
                    )
                    if row_pe.sell_signal or put_stoploss_hit:
                        put_sell_cons = (
                            strategy.sell_constraints(instrument=put_option, **kwargs)
                            if strategy.sell_constraints is not None
                            else True
                        )
                        if put_sell_cons or put_stoploss_hit:
                            trader.execute_sell(
                                tick=row_pe,
                                trade=bucket.open_position.open_trade,
                                instrument=put_option,
                                exit_price = stoploss if put_stoploss_hit else None,
                                remark = "Stoploss hit" if put_stoploss_hit else ''
                            )
                            bucket.open_position = None


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

trader.test()
logger.info("\n-------------BACKTESTING COMPLETE-------------")
