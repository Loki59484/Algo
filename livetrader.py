from collections import defaultdict
from functools import partial
from typing import Literal
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from datetime import date
import datetime as dt
import pandas as pd
import numpy as np
import zmq.asyncio
import asyncio
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
from core.upstox_methods import UpstoxClient
from core.methods import setup_cli, start_heartbeat, ZMQErrorLogger, calculate_trade_charges
from core import anatomy as ana
from ui import tui


zmq_handler = ZMQErrorLogger(component_name="Live Trader", port=5568)
zmq_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(zmq_handler)

ustox = UpstoxClient()
MATRIX_CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache" / "matrices"
MATRIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(
    "--------------------HYBRID LIVE TRADING ENGINE (5M Strategy / 1M Execution)--------------------".center(
        shutil.get_terminal_size().columns
    )
)

# =====================================================================
# THE GEAR BOX: OPTUNA VALIDATED PARAMETERS (5-MIN TIMEFRAME)
# =====================================================================
SNIPE_BEST_PARAMS = {
    "rsi_min": 55,
    "adx_min": 35,
    "use_ema": True,
    "use_supertrend": False,
    "req_active_slope": True,
    "use_macd": False,
    "bb_max_width": 0.025,
    "sl_atr": 1.5,
    "target_atr": 6.0,
    "trailing_sl_atr": 1.5,
}
SCALP_BEST_PARAMS = {
    "rsi_min": 60,
    "adx_min": 25,
    "use_ema": True,
    "use_supertrend": False,
    "req_active_slope": False,
    "use_macd": False,
    "bb_max_width": 0.035,
    "sl_atr": 1.5,
    "target_atr": 3.0,
    "trailing_sl_atr": 0.8,
    "max_daily_trades": 5,
    "max_daily_profit": 15000.0,
}

# =====================================================================
# MARKET FRICTION & LATENCY VARIABLES
# =====================================================================
SLIPPAGE = 0.25  # Slippage per market order fill

# =====================================================================
# HELPER FUNCTIONS & MULTI-TIMEFRAME RESAMPLER
# =====================================================================
def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregates 1-minute OHLCV data into 5-minute candles."""
    if df_1m.empty:
        return df_1m

    resampled = df_1m.resample("5min", origin="start_day").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    
    return resampled


def place_buy_order(
    opt_leg: Instrument,
    opt_row: pd.Series,
    bucket: Bucket,
    trader: ana.Trader,
    active_params,
    spot_row,
    gear_name,
    timestamp,
):
    try:
        atr = opt_row.get("ATRr_14", 0)
        if atr == 0:
            return -1

        close_price = opt_row.get("close", 0)
        buy_price = close_price + SLIPPAGE

        qty = trader.calculate_units(close=buy_price, lot_size=opt_leg.lot_size)
        if qty == 0:
            return -1

        status = trader.broker.buy_order(key=opt_leg.key, price=buy_price, qty=qty)
        
        if status is None or status.get("status") != "success":
            logger.error(f"Order failed or rejected by API: {status}")
            return -1

        ord_id = status["data"]["order_ids"][0]
        
        from types import SimpleNamespace
        pos = SimpleNamespace(instrument_key=opt_leg.key)
        
        bucket.open_position = pos
        bucket.open_position.target = buy_price + (active_params["target_atr"] * atr)
        bucket.open_position.stoploss = buy_price - (active_params["sl_atr"] * atr)
        bucket.open_position.trail_dist = (
            active_params.get("trailing_sl_atr", 1.5) * atr
        )
        bucket.open_position.highest_seen = buy_price

        trader.portfolio.report.append(
            Trade(
                trade_id=ord_id,
                instrument_key=opt_leg.key,
                buy_timestamp=timestamp,
                side=opt_leg.type,
                buy_price=buy_price,
                buy_qty=qty,
                buy_conditions=spot_row.to_dict(),
            )
        )
        logger.info(
            f"[{gear_name}] MARKET ORDER FILLED: Buy {opt_leg.type} @ ₹{buy_price:.2f} (Base LTP: ₹{close_price:.2f}) # Qty {qty}"
        )
        return 1
    except Exception as e:
        logger.exception(f"Error in place_buy_order: {e}")
        return -1


def _change_gears(trader, timestamp):
    try:
        today = timestamp.date()
        daily_trades = []
        for t in trader.portfolio.report:
            ts = getattr(t, "buy_timestamp", getattr(t, "Buy_timestamp", None))
            if ts and pd.Timestamp(ts).date() == today:
                daily_trades.append(t)

        daily_realized_pnl = sum(
            getattr(t, "pnl", getattr(t, "PnL", 0))
            for t in daily_trades
            if getattr(t, "remark", getattr(t, "Remark", "")) in ["T", "SL", "EOD", "TARGET", "STOPLOSS"]
        )

        num_trades_today = len(daily_trades)

        if (
            daily_realized_pnl >= SCALP_BEST_PARAMS["max_daily_profit"]
            or num_trades_today >= SCALP_BEST_PARAMS["max_daily_trades"]
        ):
            return SNIPE_BEST_PARAMS, "SNIPER"
        return SCALP_BEST_PARAMS, "SCALP"
    except Exception as e:
        logger.exception(f"Error in _change_gears: {e}")
        return SCALP_BEST_PARAMS, "SCALP"


def _close_trade(trader, bucket, timestamp, exit_price, remark, gear_name):
    try:
        report = trader.portfolio.report[-1]

        buy_qty = getattr(report, "buy_qty", getattr(report, "Buy_qty", 0))
        buy_price = getattr(report, "buy_price", getattr(report, "Buy_price", 0))

        report.sell_qty = buy_qty
        report.sell_timestamp = timestamp
        report.sell_price = exit_price
        report.remark = remark
        report.movement = exit_price - buy_price
        report.pnl = (exit_price * buy_qty) - (buy_price * buy_qty)
        report.total = trader.portfolio.funds.total

        bucket.open_position = None
        trader.portfolio.funds.settle()

        pnl = getattr(report, "pnl", getattr(report, "PnL", 0))
        logger.info(
            f"[{gear_name}] Position Closed: {timestamp} | {remark} | Fill Price: ₹{exit_price:.2f} | Net PnL: ₹{pnl:.2f}"
        )
    except Exception as e:
        logger.exception(f"Error in _close_trade: {e}")


def _evaluate_signals(spot_row, spot_prev, params):
    rsi = spot_row.get("RSI_14", 50)
    adx = spot_row.get("ADX_14", 0)
    spot_close = spot_row.get("close", 0)

    supert_slope = spot_row.get("SUPERT_14_2.0", 0) - spot_prev.get("SUPERT_14_2.0", 0)
    bbu, bbl = spot_row.get("BBU_20_2.0", 0), spot_row.get("BBL_20_2.0", 0)
    bb_width = ((bbu - bbl) / spot_close) if spot_close > 0 else 1.0

    ce_cond = (rsi > params["rsi_min"]) and (adx > params["adx_min"])
    pe_cond = (rsi < (100 - params["rsi_min"])) and (adx > params["adx_min"])

    if params["use_ema"]:
        ema_200 = spot_row.get("EMA_200", 0)
        ce_cond = ce_cond and (spot_close > ema_200)
        pe_cond = pe_cond and (spot_close < ema_200)

    if params["use_supertrend"]:
        supert_dir = spot_row.get("SUPERTd_14_2.0", 0)
        ce_cond = ce_cond and (supert_dir == 1)
        pe_cond = pe_cond and (supert_dir == -1)

    if params["req_active_slope"]:
        ce_cond = ce_cond and (supert_slope > 0)
        pe_cond = pe_cond and (supert_slope < 0)

    if params.get("use_macd", False):
        macd = spot_row.get("MACD_12_26_9", 0)
        macds = spot_row.get("MACDs_12_26_9", 0)
        ce_cond = ce_cond and (macd > macds)
        pe_cond = pe_cond and (macd < macds)

    ce_cond = ce_cond and (bb_width < params["bb_max_width"])
    pe_cond = pe_cond and (bb_width < params["bb_max_width"])

    return ce_cond, pe_cond


# =====================================================================
# LIVE EXECUTOR & POSITION MANAGER (1-MIN RESOLUTION FOR POSITIONS)
# =====================================================================
async def execute_live(
    trader: ana.Trader,
    bucket: Bucket,
    ui_socket: zmq.asyncio.Socket = None,
    data_frames: dict = None,
    **kwargs,
):
    spot: Instrument = bucket.spot
    call_option: Instrument = bucket.legs.get("CE")
    put_option: Instrument = bucket.legs.get("PE")

    if not spot or not call_option or not put_option or not data_frames:
        return

    def get_df(target_key):
        for k, df in data_frames.items():
            if k[0] == target_key:
                return df
        return None

    spot_df = get_df(spot.key)
    ce_df = get_df(call_option.key)
    pe_df = get_df(put_option.key)

    if spot_df is None or ce_df is None or pe_df is None:
        return

    if len(spot_df) < 5 or ce_df.empty or pe_df.empty:
        return

    spot_row = spot_df.iloc[-1]
    ce_row, pe_row = ce_df.iloc[-1], pe_df.iloc[-1]
    timestamp = spot_df.index[-1] if isinstance(spot_df.index, pd.DatetimeIndex) else pd.to_datetime(spot_df["timestamp"].iloc[-1])

    if ui_socket:
        payload = {
            spot.key: spot_row.to_json(date_format="iso"),
            call_option.key: ce_row.to_json(date_format="iso"),
            put_option.key: pe_row.to_json(date_format="iso"),
        }
        await ui_socket.send_string(json.dumps(payload))

    ACTIVE_PARAMS, gear_name = _change_gears(trader, timestamp)

    # 1. Intra-Candle Open Position Management (Evaluated every 1-Minute Candle)
    if bucket.open_position is not None:
        _manage_position_live(
            bucket,
            call_option,
            put_option,
            ce_df,
            pe_df,
            timestamp,
            ACTIVE_PARAMS,
            gear_name,
            trader,
        )
        return

    # 2. Strategy Signal Generation (Evaluated ONLY at 5-Minute Candle Closures)
    curr_time = timestamp.time()
    if curr_time > dt.time(15, 10):
        return

    # Check if the current 1-minute timestamp marks a 5-minute boundary completion
    if timestamp.minute % 5 == 4 or timestamp.minute % 5 == 0:
        current_5m_boundary = timestamp.floor("5min")
        
        if getattr(bucket, 'last_signal_5m', None) != current_5m_boundary:
            spot_df_5m = trader.strategy.apply(resample_to_5m(spot_df), key=spot.key)
            
            if len(spot_df_5m) >= 3:
                spot_closed_5m = spot_df_5m.iloc[-2]
                spot_prev_5m = spot_df_5m.iloc[-3]
                
                ce_cond, pe_cond = _evaluate_signals(spot_closed_5m, spot_prev_5m, ACTIVE_PARAMS)
                
                if ce_cond or pe_cond:
                    bucket.last_signal_5m = current_5m_boundary
                    
                    if ce_cond:
                        logger.info(f"[{gear_name}] 5M SIGNAL FIRED (CE)! Executing on 1m Open...")
                        place_buy_order(call_option, ce_row, bucket, trader, ACTIVE_PARAMS, spot_closed_5m, gear_name, timestamp)
                    elif pe_cond:
                        logger.info(f"[{gear_name}] 5M SIGNAL FIRED (PE)! Executing on 1m Open...")
                        place_buy_order(put_option, pe_row, bucket, trader, ACTIVE_PARAMS, spot_closed_5m, gear_name, timestamp)


def _manage_position_live(
    bucket, call_option, put_option, ce_df, pe_df, timestamp, active_params, gear_name, trader
):
    """Evaluates stoplosses and targets intra-candle using 1-minute OHLC data."""
    try:
        pos = bucket.open_position
        pos_key = getattr(
            pos, "instrument_token", getattr(pos, "instrument_key", getattr(pos, "key", None))
        )
        is_ce = pos_key == call_option.key

        active_opt = call_option if is_ce else put_option
        active_row = ce_df.iloc[-1] if is_ce else pe_df.iloc[-1]

        high = active_row.get("high", 0)
        low = active_row.get("low", 0)
        close = active_row.get("close", 0)

        if not hasattr(pos, "highest_seen"):
            pos.highest_seen = close
            atr = active_row.get("ATRr_14", 1.0)
            pos.trail_dist = active_params.get("trailing_sl_atr", 1.5) * atr
            if not hasattr(pos, "target"):
                pos.target = close + (active_params["target_atr"] * atr)
            if not hasattr(pos, "stoploss"):
                pos.stoploss = close - (active_params["sl_atr"] * atr)

        # Pessimistic Check: Stop loss evaluated first
        stoploss_hit = low <= pos.stoploss
        target_hit = high >= pos.target
        eod_square_off = timestamp.time() >= dt.time(15, 15)

        if stoploss_hit or target_hit or eod_square_off:
            remark = "STOPLOSS" if stoploss_hit else ("TARGET" if target_hit else "EOD")
            logger.info(f"[{gear_name}] {remark} TRIGGERED (Intra-Candle 1m resolution)...")

            last_trade = trader.portfolio.report[-1]
            qty_to_sell = getattr(last_trade, "buy_qty", getattr(last_trade, "Buy_qty", 0))

            exit_price = (pos.stoploss - SLIPPAGE) if stoploss_hit else (pos.target if target_hit else close - SLIPPAGE)

            status = trader.broker.sell_order(key=active_opt.key, qty=qty_to_sell, price=exit_price)

            if status != -1:
                _close_trade(trader, bucket, timestamp, exit_price, remark, gear_name)
        else:
            if high > pos.highest_seen:
                pos.highest_seen = high
                new_sl = pos.highest_seen - pos.trail_dist
                pos.stoploss = max(pos.stoploss, new_sl)

    except Exception as e:
        logger.exception(f"CRITICAL Error managing live position: {e}")


def _aggregate_tick(inst: Instrument, tick):
    try:
        new_candle = tick.ohlc_1m if hasattr(tick, "ohlc_1m") else tick
        if tick is None:
            return
        minute_ts = new_candle.timestamp.replace(second=0, microsecond=0)
        last_candle = inst.historical_candles[-1] if inst.historical_candles else None
        if last_candle is not None and last_candle.timestamp.tzinfo is not None:
            minute_ts = minute_ts.tz_localize(last_candle.timestamp.tz) 

        if last_candle and last_candle.timestamp == minute_ts:
            last_candle.high = max(last_candle.high, new_candle.high)
            last_candle.low = min(last_candle.low, new_candle.low)
            last_candle.close = new_candle.close
            last_candle.volume += new_candle.volume
        elif last_candle and minute_ts < last_candle.timestamp:
            pass
        else:
            c = deepcopy(new_candle)
            c.timestamp = minute_ts
            inst.historical_candles.append(c)
    except Exception as e:
        logger.info(f"{e}")


async def processor(
    trader: ana.Trader,
    bucket: Bucket,
    stopevent: asyncio.Event,
    ui_socket=None,
    **kwargs,
):
    logger.info("Processor initiated.")
    spot_key = next(
        (k for k in trader.instruments.keys() if "INDEX" in k[0] or trader.instruments[k].type == "SPOT"),
        None,
    )
    trading_items = [data.key for data in bucket.legs.values()]
    if spot_key:
        trading_items.append(spot_key[0])

    prev_ticks = {}

    while not stopevent.is_set():
        try:
            ticks: dict[str, Tick] = await asyncio.wait_for(trader.datafeed.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        if ticks == prev_ticks:
            continue
        prev_ticks = ticks

        try:
            keys = list(ticks.keys())
            if not any(k[0] in trading_items for k in keys):
                stopevent.set()
                break

            for key, tick in ticks.items():
                if key in trader.instruments.keys():
                    _aggregate_tick(trader.instruments[key], tick)

            tasks = []
            task_keys = []

            for key, val in trader.instruments.items():
                if key[0] in trading_items:
                    tasks.append(
                        asyncio.to_thread(trader.strategy.apply, val.historical_candles, key=key)
                    )
                    task_keys.append(key)

            results = await asyncio.gather(*tasks)
            data_frames = dict(zip(task_keys, results))

            await execute_live(
                trader=trader,
                bucket=bucket,
                ui_socket=ui_socket,
                data_frames=data_frames,
            )

        except Exception as e:
            logger.exception(f"CRITICAL ERROR in processor loop: {e}")


def _get_best_options(insts, market_quote):
    best_ce, best_pe = None, None
    max_ce_volume, max_pe_volume = -1, -1

    for _, quote in market_quote.items():
        opt_type = insts.loc[insts["instrument_key"] == quote["instrument_token"], "instrument_type"].item()
        volume = quote.get("volume", 0)
        if opt_type == "CE" and volume > max_ce_volume:
            max_ce_volume, best_ce = volume, quote.get("instrument_token")
        elif opt_type == "PE" and volume > max_pe_volume:
            max_pe_volume, best_pe = volume, quote.get("instrument_token")

    return best_ce, best_pe


def _setup_sim_dummy(spot, trading_date, spot_file, ce_file, pe_file):
    _, insts = ustox.get_options_with_expiry(
        ustox.get_all_options(instrument_key=spot), return_df=True
    )
    market_quote = ustox.get_marketquote(instrument_key=insts["instrument_key"].to_list())
    best_ce, best_pe = _get_best_options(insts, market_quote)

    def _prep_csv(filepath):
        df = pd.read_csv(filepath)
        if "time" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        if "vol" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"vol": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    spot_data = _prep_csv(spot_file)
    ce_data = _prep_csv(ce_file)
    pe_data = _prep_csv(pe_file)

    options = insts[(insts["instrument_key"] == best_ce) | (insts["instrument_key"] == best_pe)]
    parsed_insts = []

    ce_meta = options[options["instrument_key"] == best_ce].iloc[0].to_dict()
    ce_meta["date"] = trading_date
    parsed_insts.append(
        Instrument.load_instrument(client=ustox, data=ce_data, metadata=ce_meta, lookback=2, load_history=True, is_expired=False)
    )

    pe_meta = options[options["instrument_key"] == best_pe].iloc[0].to_dict()
    pe_meta["date"] = trading_date
    parsed_insts.append(
        Instrument.load_instrument(client=ustox, data=pe_data, metadata=pe_meta, lookback=1, load_history=True, is_expired=False)
    )

    spot_meta = {"instrument_key": spot, "instrument_type": "INDEX", "exchange": "NSE", "date": trading_date}
    parsed_insts.append(
        Instrument.load_instrument(client=ustox, data=spot_data, metadata=spot_meta, lookback=2, load_history=True, is_expired=False)
    )

    for inst in parsed_insts:
        if not inst.historical_candles:
            continue
        unique_c = {}
        for c in inst.historical_candles:
            c.timestamp = c.timestamp.replace(second=0, microsecond=0)
            unique_c[c.timestamp] = c
        sorted_ts = sorted(unique_c.keys())
        inst.historical_candles.clear()
        inst.historical_candles.extend([unique_c[ts] for ts in sorted_ts])

    return parsed_insts


def _setup_live(spot: str | list[str], trading_date: date):
    _, insts = ustox.get_options_with_expiry(ustox.get_all_options(instrument_key=spot), return_df=True)
    market_quote = ustox.get_marketquote(instrument_key=insts["instrument_key"].to_list())
    best_ce, best_pe = _get_best_options(insts, market_quote)
    options = insts[(insts["instrument_key"] == best_ce) | (insts["instrument_key"] == best_pe)]
    parsed_insts = Instrument.parse_options(client=ustox, options=options, lookback=2, is_expired=False)

    spot_data = ustox.get_historical(dtype="intraday", instrument_key=spot)
    spot_metadata = {"instrument_key": spot, "instrument_type": "INDEX", "exchange": "NSE", "date": trading_date}
    spot_inst = Instrument.load_instrument(client=ustox, data=spot_data, metadata=spot_metadata, lookback=2, load_history=True)
    parsed_insts.append(spot_inst)
    return parsed_insts


def _setup_sim(args):
    files = []
    if args.tickwise:
        data_dir = Path(args.tickwise).resolve().absolute()
        files.extend(list(data_dir.iterdir()))
    if args.bulk:
        for data_dir in args.bulk:
            data_dir = Path(data_dir).resolve().absolute()
            files.extend(list(data_dir.iterdir()))
    files.sort()
    return Instrument.load_multiple(client=ustox, source=files, lookback=2)


def setup_mode(args, trader: ana.Trader):
    trading_date = getattr(args, "date", date.today())
    if args.command == "live":
        insts = _setup_live(args.index, trading_date)
    elif args.command == "sim":
        if args.tickwise == "dummy":
            insts = _setup_sim_dummy(args.index, trading_date, args.call, args.put, args.spot)
        else:
            insts = _setup_sim(args)

    insts_dict = {(item.key, item.date): item for item in insts}
    trader.add_instrument(insts_dict)
    return insts_dict


def _print_trade_report(trader):
    print("\n" + "=" * 50)
    print("🎯 LIVE / SIM TRADING SESSION REPORT (5M/1M MTF) 🎯")
    print("=" * 50)

    gross_pnl, total_charges = 0.0, 0.0
    wins, losses, closed_trades = 0, 0, 0

    for t in trader.portfolio.report:
        remark = getattr(t, "remark", getattr(t, "Remark", None))
        if remark:
            closed_trades += 1
            b_price = getattr(t, "buy_price", getattr(t, "Buy_price", 0))
            s_price = getattr(t, "sell_price", getattr(t, "Sell_price", 0))
            qty = getattr(t, "buy_qty", getattr(t, "Buy_qty", 0))

            trade_gross = (s_price - b_price) * qty
            trade_charges = calculate_trade_charges(b_price, s_price, qty)

            gross_pnl += trade_gross
            total_charges += trade_charges

            if (trade_gross - trade_charges) > 0:
                wins += 1
            else:
                losses += 1

    print(f"Total Trades Taken  : {closed_trades}")
    if closed_trades > 0:
        win_rate = (wins / closed_trades) * 100
        print(f"Winning Trades      : {wins}")
        print(f"Losing Trades       : {losses}")
        print(f"Win Rate            : {win_rate:.2f}%")
        print("-" * 50)
        print(f"Gross PnL           : ₹{gross_pnl:,.2f}")
        print(f"Taxes & Brokerage   : ₹{total_charges:,.2f}")
        print(f"NET PROFIT (REAL)   : ₹{(gross_pnl - total_charges):,.2f}")
    else:
        print("No trades triggered in this session.")
    print("=" * 50 + "\n")


async def starter(trader, args, tradable_insts, feeder_queue):
    tasks_to_run = []
    ui_socket, context = None, None
    
    try:
        stopevent = asyncio.Event()
        trading_items = [data.key for data in trader.buckets[-1].legs.values()]
        trading_items.append(args.index)

        trading_insts = {k: v for k, v in tradable_insts.items() if k[0] in trading_items}
        
        if args.command == "live":
            simfeeder = ana.LivefeedStreamer(ustox, trading_insts, feeder_queue)
        elif args.command == "sim":
            simfeeder = ana.SimfeedStreamer(trading_insts, feeder_queue, stopevent)

        tasks = [simfeeder.start()]

        if args.tui:
            port = "tcp://127.0.0.1:5556"
            context = zmq.asyncio.Context().instance()
            ui_socket = context.socket(zmq.PUB)
            ui_socket.bind(port)
            app = tui.TradingTUI(trader, trader.buckets[-1], simulate=True, client=ustox, port=port)
            tasks.append(app.run_async())

        tasks.append(processor(trader, trader.buckets[-1], stopevent, ui_socket=ui_socket))

        tasks_to_run = [asyncio.create_task(task) for task in tasks]
        await asyncio.gather(*tasks_to_run, return_exceptions=True)

    except Exception as e:
        logger.exception(f"Exception while running livetrader |\n {e}")
    finally:
        for task in tasks_to_run:
            if not task.done():
                task.cancel()
        if args.tui:
            if ui_socket:
                ui_socket.close()
            if context:
                context.term()


def main():
    args = setup_cli()
    capital: Funds = Funds.update_from_json(ustox.get_funds())
    prtf = Portfolio(funds=capital)
    feeder_queue = asyncio.Queue(maxsize=10)
    strat = ana.Strategy()

    trader = ana.Trader(
        portfolio=prtf,
        strategy=strat,
        broker=ana.LiveBroker(ustox),
        datafeed=feeder_queue,
    )

    tradable_insts = setup_mode(args, trader)
    daily_buckets = defaultdict(dict)

    for instrument in tqdm(trader.instruments.values(), desc="Filtering instruments", leave=False):
        leg_type = "CE" if "CE" in instrument.type else ("PE" if "PE" in instrument.type else "INDEX")
        daily_buckets[instrument.date][leg_type] = instrument

    for trade_date, legs in tqdm(daily_buckets.items(), desc="Loading Buckets", leave=False):
        options = {k: v for k, v in legs.items() if k in ["CE", "PE"]}
        bucket = Bucket(trade_date, legs=options, spot=legs.get("INDEX", None))
        trader.buckets.append(bucket)

    strat.add_indicators([
        {"kind": "supertrend", "length": 14, "multiplier": 2.0},
        {"kind": "adx", "length": 14},
        {"kind": "atr", "length": 14},
        {"kind": "ema", "length": 200},
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "bbands", "length": 20, "std": 2.0},
        {"kind": "vwap"},
    ])

    try:
        asyncio.run(
            starter(trader=trader, args=args, tradable_insts=tradable_insts, feeder_queue=feeder_queue)
        )
        _print_trade_report(trader)
    except Exception as e:
        logger.exception(f"Exception {e}")


if __name__ == "__main__":
    start_heartbeat(component_name='Live Trader', port=5558)
    main()