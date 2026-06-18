from collections import defaultdict
from dataclasses import asdict
from functools import partial
from typing import Literal
from copy import deepcopy
from pathlib import Path
import pandas_ta as ta
from tqdm import tqdm
import asyncio
import datetime as dt
import pandas as pd
import logging
import shutil
import json
import zmq
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
logger = logging.getLogger(__name__)

# IMPORTING CUSTOM MODULES
from core.datatypes import Instrument, Trade, Portfolio, Bucket, Funds, Tick
from core.upstox_methods import UpstoxClient, DATA_DIR
from core.methods import setup_cli, to_ist
from core import anatomy as ana
from ui import tui

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


BEST_PARAMS = {
    'rsi_min': 47,
    'adx_min': 61,
    'use_ema': True,
    'use_supertrend': False,
    'req_active_slope': True,
    'use_macd': False,
    'bb_max_width': 0.014498753585377278,
    'sl_atr': 1.5,
    'target_atr': 8.0
}

# DEFINING BUY-SELL PARAMETERS
def buy_signal(df, **kwargs):
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


# EXECUTOR FUNCTION FOR THE TRADER
async def executor(
    data: pd.DataFrame,
    bucket: Bucket,
    strategy: ana.Strategy,
    trader: ana.Trader,
    **kwargs,
):

    data = data.rename(
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

    latest_tick = next(data.iloc[[-1]].itertuples())
    ui_socket: zmq.asyncio.Socket = kwargs.get("ui_socket", None)

    if ui_socket:
        last_row = data.iloc[-1]
        last_row["timestamp"] = last_row.name
        payload = {latest_tick.key[0]: last_row.to_json(date_format="iso")}
        await ui_socket.send_string(json.dumps(payload))

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
                    qty=trader.portfolio.report[-1].buy_qty,
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
                report.sell_conditions = row._asdict()
                report.sell_qty = report.buy_qty
                report.sell_timestamp = row.Index
                report.sell_price = stoploss if stoploss_hit else row.close
                report.remark = "SL" if stoploss_hit else "T" if target_hit else "-"
                report.movement = report.sell_price - report.buy_price
                report.pnl = (report.sell_price * report.sell_qty) - (
                    report.buy_price * report.buy_qty
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
            funds_bf = trader.portfolio.funds.available_margin
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
            bucket.open_position.stoploss = None
            trader.portfolio.report.append(
                Trade(
                    trade_id=ord_id,
                    instrument_key=option.key,
                    buy_timestamp=row.Index,
                    side=option.type,
                    buy_price=row.close,
                    buy_qty=qty,
                    buy_conditions=row._asdict(),
                )
            )

            logger.info("Trade executed successfully for buy side.")
            return 1

    call_option = bucket.legs.get("CE")
    put_option = bucket.legs.get("PE")

    if call_option is None or put_option is None:
        logger.warning(f"NoneType option found for bucket {bucket.date}")
        return

    if bucket.open_position is None:
        if latest_tick.buy_signal:
            # Check which instrument this tick belongs to before buying!
            if latest_tick.key == call_option.key:
                status = execute_buy(latest_tick, call_option, trader)
            elif latest_tick.key == put_option.key:
                status = execute_buy(latest_tick, put_option, trader)

            if status == 1:
                return
        trader.portfolio.funds.settle()
    else:
        stoploss = bucket.open_position.stoploss
        if bucket.open_position.instrument_token == call_option.key:
            execute_sell(latest_tick, call_option, trader, stoploss)
        elif bucket.open_position.instrument_token == put_option.key:
            execute_sell(latest_tick, put_option, trader, stoploss)
        trader.portfolio.funds.settle()
    

# PROCESSOR TO HANDLE INCOMING TICKS
async def processor(
    trader: ana.Trader, bucket: list[ana.Bucket], stopevent: asyncio.Event, **kwargs
):
    trading_items = [data.key for data in bucket.legs.values()]
    logger.debug(f"Custom tick processor started.")
    prev_ticks = {}
    while not stopevent.is_set():
        ticks: dict[str, Tick] = await trader.datafeed.get()

        if ticks == prev_ticks:
            continue
        prev_ticks = ticks
        keys = [key for key in ticks.keys()]
        if not any([key in [item[0] for item in keys] for key in trading_items]):
            logger.info(f"Setting stop event since no keys.")
            stopevent.set()
            break

        for key, tick in ticks.items():
            try:
                if key in trader.instruments.keys():
                    trader.instruments[key].historical_candles.append(
                        tick.ohlc_1m if isinstance(tick, Tick) else tick
                    )
            except Exception as e:
                logger.exception(f"Error in processor : {e}")
            logger.debug(f"Generating tasks")
        tasks = [
            asyncio.to_thread(trader.strategy.apply, val.historical_candles, key=key)
            for key, val in trader.instruments.items()
            if key[0] in trading_items
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            await trader.executor(trader=trader, data=result)


def main():
 
    # GETTING INSTRUMENTS/BUCKETS TO BE SIMULATED
    args = setup_cli()
    # SETUP TRADER INSTANCE
    capital: Funds = Funds.update_from_json(ustox.get_funds(),)
    #capital: Funds = Funds(starting_capital=20000)
    prtf = Portfolio(funds=capital)
    feeder_queue = asyncio.Queue(maxsize=10)
    strat = ana.Strategy()

    trader = ana.Trader(
        portfolio=prtf,
        strategy=strat,
        #broker=ana.SimBroker(portfolio=prtf),
        broker=ana.LiveBroker(ustox),
        datafeed=feeder_queue,
    )

    def setup_mode(mode: Literal["sim", "live"], key: str | list[str]):
        if mode == "live":
            _, insts = ustox.get_options_with_expiry(
                ustox.get_all_options(instrument_key=key), return_df=True
            )
            market_quote = ustox.get_marketquote(
                instrument_key=insts["instrument_key"].to_list()
            )
            best_ce = None
            max_ce_volume = -1

            best_pe = None
            max_pe_volume = -1

            for _, quote in market_quote.items():
                opt_type = insts.loc[
                    insts["instrument_key"] == quote["instrument_token"],
                    "instrument_type",
                ].item()
                volume = quote.get("volume", 0)
                if opt_type == "CE":
                    if volume > max_ce_volume:
                        max_ce_volume = volume
                        best_ce = quote.get("instrument_token")

                elif opt_type == "PE":
                    if volume > max_pe_volume:
                        max_pe_volume = volume
                        best_pe = quote.get("instrument_token")

            options = insts[
                (insts["instrument_key"] == best_ce)
                | (insts["instrument_key"] == best_pe)
            ]

            insts = Instrument.parse_options(client=ustox, options=options, lookback=2)

        elif mode == "sim":
            files = []
            if args.tickwise:
                dir = Path(args.tickwise).resolve().absolute()
                files.extend(
                    [file for file in dir.iterdir() if "INDEX" not in str(file)]
                )

            if args.bulk:
                for dir in args.bulk:
                    dir = Path(dir).resolve().absolute()
                    files.extend(
                        [file for file in dir.iterdir() if "INDEX" not in str(file)]
                    )
            files.sort()
            insts = Instrument.load_multiple(client=ustox, source=files, lookback=2)

        insts_dict = {(item.key, item.date): item for item in insts}
        sim_dict = deepcopy(insts_dict)
        trader.add_instrument(insts_dict)
        return sim_dict

    # CREATE BUCKETS FOR EACH DAY
    tradable_insts = setup_mode(args.command,args.index)
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
        break
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
    trader.strategy.buy_conditon = buy_signal
    trader.strategy.sell_condition = sell_signal
    trader.strategy.buy_constraints = buy_cons
    trader.strategy.sell_constraints = sell_cons

    trader.set_executor(
        partial(
            executor,
            bucket=trader.buckets[-1],
            strategy=trader.strategy,
            buy_condition=buy_signal,
            sell_condition=sell_signal,
        )
    )

    trader.set_tick_processor(partial(processor, trader, trader.buckets[-1]))

    async def starter():
        try:
            stopevent = asyncio.Event()
            trading_items = [data.key for data in bucket.legs.values()]
            trading_insts = {
                k: v for k, v in tradable_insts.items() if k[0] in trading_items
            }
            if args.command == "live":
                logger.info("Live streamer created.")
                simfeeder = ana.LivefeedStreamer(ustox, trading_insts, feeder_queue)
            elif args.command == "sim":
                logger.info("Simulation streamer created.")
                simfeeder = ana.SimfeedStreamer(trading_insts, feeder_queue, stopevent)
            tasks = [simfeeder.start(), trader.processor(stopevent)]
            if args.tui:
                port = "tcp://127.0.0.1:5556"
                context = zmq.asyncio.Context()
                socket = context.socket(zmq.PUB)
                socket.bind(port)
                logger.info("Broadcasting UI data to port 5555")
                trader.set_executor(
                    partial(
                        executor,
                        bucket=trader.buckets[-1],
                        strategy=trader.strategy,
                        buy_condition=buy_signal,
                        sell_condition=sell_signal,
                        ui_socket=socket,
                    )
                )
                app = tui.TradingTUI(
                    trader, trader.buckets[-1], simulate=True, client=ustox, port=port
                )
            tasks.append(app.run_async())
            tasks_to_run = [asyncio.create_task(task) for task in tasks]

            await asyncio.gather(*tasks_to_run, return_exceptions=True)
        except Exception as e:
            logger.exception(f" Exception while running livetrader |\n {e}")
        finally:
            for task in tasks_to_run:
                task.cancel()
                logger.info(f"Cancelled async tasks: {task.get_coro()}.")
            if args.tui:
                socket.close()
                context.term()

    try:
        asyncio.run(starter())
    except Exception as e:
        logger.exception(f"Exception {e}")


main()
