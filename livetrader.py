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

REPORTS_DIR = ROOT_DIR / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(
    "--------------------HYBRID LIVE TRADING ENGINE (5M Strategy / 1M Execution)--------------------".center(
        shutil.get_terminal_size().columns
    )
)

# =====================================================================
# THE GEAR BOX: UNIFIED MTF PARAMETERS (SYNCHRONIZED WITH TESTER)
# =====================================================================
BEST_PARAMS = {
    "rsi_min": 60,
    "adx_min": 25,
    "adx_max": 75,
    "use_ema": True,
    "use_supertrend": False,
    "req_active_slope": False,
    "use_macd": False,
    "bb_max_width": 0.035,
    "sl_atr": 1.5,
    "target_atr": 5.0,
    "trailing_sl_atr": 1.2,
    "max_daily_trades": 3,
    "max_daily_profit": 12000, 
    "capital_allocation_pct": 0.50,
    "max_lots_cap": 5,
    "initial_capital": 70000.0,
}

SLIPPAGE = 0.25

def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregates 1-minute OHLCV data into 5-minute candles safely handling index mapping."""
    if df_1m.empty:
        return df_1m

    df_working = df_1m.copy()
    if not isinstance(df_working.index, pd.DatetimeIndex):
        if "timestamp" in df_working.columns:
            df_working["timestamp"] = pd.to_datetime(df_working["timestamp"])
            df_working.set_index("timestamp", inplace=True)
        elif "time" in df_working.columns:
            df_working["timestamp"] = pd.to_datetime(df_working["time"])
            df_working.set_index("timestamp", inplace=True)
        else:
            return pd.DataFrame()

    resampled = df_working.resample("5min", origin="start_day").agg({
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
    locked_limit_price,
    locked_atr
):
    try:
        # --- REALISTIC POSITION SIZING ---
        running_capital = trader.portfolio.funds.total
        allocated_cash = running_capital * active_params.get("capital_allocation_pct", 0.50)
        cost_per_lot = locked_limit_price * opt_leg.lot_size
        
        num_lots = int(allocated_cash // cost_per_lot)
        max_lots = active_params.get("max_lots_cap", 5)
        
        if num_lots > max_lots:
            num_lots = max_lots
            
        qty = num_lots * opt_leg.lot_size
        
        if qty == 0:
            logger.warning(f"[{gear_name}] Insufficient margin for 1 lot. Skipping trade.")
            return -1

        # Placing a limit order via broker
        status = trader.broker.buy_order(key=opt_leg.key, price=locked_limit_price, qty=qty, order_type="LIMIT")
        
        # --- ROBUST API SCHEMA PARSING (Matches Upstox API) ---
        ord_id = None
        if isinstance(status, dict):
            if status.get("status") != "success":
                logger.error(f"[{gear_name}] Broker API rejected order: {status}")
                return -1
            ord_id = status.get("data", {}).get("order_ids", ["sim_order_obj"])[0]
        elif status == -1 or status is None:
            logger.error(f"[{gear_name}] Order failed silently at broker level.")
            return -1
        else:
            ord_id = str(status) 
            
        from types import SimpleNamespace
        pos = SimpleNamespace(instrument_key=opt_leg.key)
        
        bucket.open_position = pos
        bucket.open_position.target = locked_limit_price + (active_params["target_atr"] * locked_atr)
        bucket.open_position.stoploss = locked_limit_price - (active_params["sl_atr"] * locked_atr)
        bucket.open_position.trail_dist = active_params.get("trailing_sl_atr", 1.2) * locked_atr
        bucket.open_position.highest_seen = locked_limit_price

        trader.portfolio.report.append(
            Trade(
                trade_id=ord_id,
                instrument_key=opt_leg.key,
                buy_timestamp=timestamp,
                side=opt_leg.type,
                buy_price=locked_limit_price,
                buy_qty=qty,
                buy_conditions=spot_row.to_dict(),
            )
        )
        logger.info(f"[{gear_name}] LIMIT ORDER ACCEPTED: Buy {opt_leg.type} @ ₹{locked_limit_price:.2f} | Qty {qty} ({num_lots} lots)")
        return 1
    except Exception as e:
        logger.exception(f"Error in place_buy_order: {e}")
        return -1


def _close_trade(trader, bucket, timestamp, exit_price, remark, gear_name):
    try:
        report = trader.portfolio.report[-1]

        buy_qty = getattr(report, "buy_qty", getattr(report, "Buy_qty", 0))
        buy_price = getattr(report, "buy_price", getattr(report, "Buy_price", 0))

        trade_gross = (exit_price - buy_price) * buy_qty
        trade_charges = calculate_trade_charges(buy_price, exit_price, buy_qty, instrument='O', trade_type='I')
        trade_net = trade_gross - trade_charges

        report.sell_qty = buy_qty
        report.sell_timestamp = timestamp
        report.sell_price = exit_price
        report.remark = remark
        report.movement = exit_price - buy_price
        report.pnl = trade_net
        report.charges = trade_charges
        
        bucket.open_position = None
        trader.portfolio.funds.total += trade_net
        report.total = trader.portfolio.funds.total

        logger.info(
            f"[{gear_name}] Position Closed: {timestamp} | {remark} | Fill: ₹{exit_price:.2f} | Net PnL: ₹{trade_net:.2f}"
        )
    except Exception as e:
        logger.exception(f"Error in _close_trade: {e}")


def _evaluate_signals(spot_row, spot_prev, params):
    rsi = spot_row.get("RSI_14", 50)
    adx = spot_row.get("ADX_14", 0)
    spot_close = spot_row.get("close", 0)

    if pd.isna(rsi) or pd.isna(adx):
        return False, False

    supert_slope = spot_row.get("SUPERT_14_2.0", 0) - spot_prev.get("SUPERT_14_2.0", 0)
    bbu, bbl = spot_row.get("BBU_20_2.0", 0), spot_row.get("BBL_20_2.0", 0)
    bb_width = ((bbu - bbl) / spot_close) if spot_close > 0 else 1.0

    adx_max = params.get("adx_max", 75)
    
    ce_cond = (rsi > params["rsi_min"]) and (params["adx_min"] < adx < adx_max)
    pe_cond = (rsi < (100 - params["rsi_min"])) and (params["adx_min"] < adx < adx_max)

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


_BUCKET_STATE = {}

async def execute_live(
    trader: ana.Trader,
    bucket: Bucket,
    ui_socket: zmq.asyncio.Socket = None,
    data_frames: dict = None,
    **kwargs,
):
    global _BUCKET_STATE
    
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

    ACTIVE_PARAMS = BEST_PARAMS
    gear_name = "MTF_ENGINE"

    # 1. PROCESS PENDING LIMIT ORDERS (1-Minute Queue)
    if getattr(bucket, "pending_entry", None) is not None:
        pending = bucket.pending_entry
        active_opt = call_option if pending["side"] == "CE" else put_option
        active_row = ce_row if pending["side"] == "CE" else pe_row
        
        # Penetration Logic: Must drop strictly BELOW limit price to trigger
        if active_row.get("low", float('inf')) < pending["limit_price"]:
            logger.info(f"[{gear_name}] QUEUE FILLED! Executing {pending['side']} @ ₹{pending['limit_price']}")
            place_buy_order(
                active_opt, active_row, bucket, trader, ACTIVE_PARAMS, 
                pending["spot_row"], gear_name, timestamp, 
                pending["limit_price"], pending["atr"]
            )
            bucket.pending_entry = None
        else:
            pending["bars_waiting"] += 1
            if pending["bars_waiting"] >= 3:
                logger.info(f"[{gear_name}] Limit Order Expired (Price never touched). Canceling.")
                bucket.pending_entry = None

    # 2. INTRA-CANDLE OPEN POSITION MANAGEMENT
    elif bucket.open_position is not None:
        _manage_position_live(
            bucket, call_option, put_option, ce_df, pe_df,
            timestamp, ACTIVE_PARAMS, gear_name, trader
        )
        
    # 3. STRATEGY SIGNAL DETECTION (Only at 5-Minute Boundaries)
    elif timestamp.minute % 5 == 4 or timestamp.minute % 5 == 0:
        if timestamp.time() > dt.time(15, 10):
            return

        bucket_id = str(bucket.date) 
        current_5m_boundary = timestamp.floor("5min")
        
        if _BUCKET_STATE.get(bucket_id) != current_5m_boundary:
            spot_df_5m = trader.strategy.apply(resample_to_5m(spot_df), key=spot.key)
            
            if len(spot_df_5m) >= 3:
                spot_closed_5m = spot_df_5m.iloc[-2]
                spot_prev_5m = spot_df_5m.iloc[-3]
                
                ce_cond, pe_cond = _evaluate_signals(spot_closed_5m, spot_prev_5m, ACTIVE_PARAMS)
                
                if ce_cond or pe_cond:
                    _BUCKET_STATE[bucket_id] = current_5m_boundary
                    
                    # Ensure daily limits aren't breached before queueing the order
                    today = timestamp.date()
                    daily_trades = [t for t in trader.portfolio.report if pd.to_datetime(getattr(t, "buy_timestamp")).date() == today]
                    daily_profit = sum(getattr(t, "pnl", 0) for t in daily_trades if getattr(t, "remark", None) is not None)

                    if len(daily_trades) >= ACTIVE_PARAMS["max_daily_trades"] or daily_profit >= ACTIVE_PARAMS["max_daily_profit"]:
                        return

                    if ce_cond:
                        logger.info(f"[{gear_name}] 5M SIGNAL (CE) -> Order Queued...")
                        bucket.pending_entry = {
                            "side": "CE", "limit_price": ce_row.get("close", 0), "bars_waiting": 0,
                            "spot_row": spot_closed_5m, "atr": ce_row.get("ATRr_14", 0)
                        }
                    elif pe_cond:
                        logger.info(f"[{gear_name}] 5M SIGNAL (PE) -> Order Queued...")
                        bucket.pending_entry = {
                            "side": "PE", "limit_price": pe_row.get("close", 0), "bars_waiting": 0,
                            "spot_row": spot_closed_5m, "atr": pe_row.get("ATRr_14", 0)
                        }


def _manage_position_live(
    bucket, call_option, put_option, ce_df, pe_df, timestamp, active_params, gear_name, trader
):
    try:
        pos = bucket.open_position
        pos_key = getattr(pos, "instrument_token", getattr(pos, "instrument_key", getattr(pos, "key", None)))
        is_ce = pos_key == call_option.key

        active_opt = call_option if is_ce else put_option
        active_row = ce_df.iloc[-1] if is_ce else pe_df.iloc[-1]

        high = active_row.get("high", 0)
        low = active_row.get("low", 0)
        close = active_row.get("close", 0)

        if not hasattr(pos, "highest_seen"):
            pos.highest_seen = close
            # Fallback if somehow ATR wasn't locked at the start (should not happen with queue)
            atr = active_row.get("ATRr_14", 1.0)
            pos.trail_dist = active_params.get("trailing_sl_atr", 1.2) * atr
            if not hasattr(pos, "target"):
                pos.target = close + (active_params["target_atr"] * atr)
            if not hasattr(pos, "stoploss"):
                pos.stoploss = close - (active_params["sl_atr"] * atr)

        stoploss_hit = low <= pos.stoploss
        target_hit = high >= pos.target
        eod_square_off = timestamp.time() >= dt.time(15, 15)

        if stoploss_hit or target_hit or eod_square_off:
            remark = "STOPLOSS" if stoploss_hit else ("TARGET" if target_hit else "EOD")
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
        new_candle = tick.ohlc_1m if hasattr(tick, "ohlm_1m") else tick
        if tick is None:
            return
            
        minute_ts = new_candle.timestamp.replace(second=0, microsecond=0)
        last_candle = inst.historical_candles[-1] if inst.historical_candles else None
        
        if last_candle is not None and last_candle.timestamp.tzinfo is not None:
            target_tz = last_candle.timestamp.tzinfo
            if minute_ts.tzinfo is None:
                minute_ts = minute_ts.tz_localize(target_tz)
            else:
                minute_ts = minute_ts.tz_convert(target_tz)

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
        logger.error(f"Error in _aggregate_tick: {e}")


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
        Instrument.load_instrument(client=ustox, data=ce_data, metadata=ce_meta, lookback=4, load_history=True, is_expired=False)
    )

    pe_meta = options[options["instrument_key"] == best_pe].iloc[0].to_dict()
    pe_meta["date"] = trading_date
    parsed_insts.append(
        Instrument.load_instrument(client=ustox, data=pe_data, metadata=pe_meta, lookback=4, load_history=True, is_expired=False)
    )

    spot_meta = {"instrument_key": spot, "instrument_type": "INDEX", "exchange": "NSE", "date": trading_date}
    parsed_insts.append(
        Instrument.load_instrument(client=ustox, data=spot_data, metadata=spot_meta, lookback=4, load_history=True, is_expired=False)
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
    parsed_insts = Instrument.parse_options(client=ustox, options=options, lookback=4, is_expired=False)

    spot_data = ustox.get_historical(dtype="intraday", instrument_key=spot)
    spot_metadata = {"instrument_key": spot, "instrument_type": "INDEX", "exchange": "NSE", "date": trading_date}
    spot_inst = Instrument.load_instrument(client=ustox, data=spot_data, metadata=spot_metadata, lookback=4, load_history=True)
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
    return Instrument.load_multiple(client=ustox, source=files, lookback=4)


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

def _export_livetrader_reports(trader):
    """Intercepts and dumps trade logs to CSV/Parquet for institutional reconciliation."""
    trade_rows = []
    for idx, t in enumerate(trader.portfolio.report):
        remark = getattr(t, "remark", getattr(t, "Remark", None))
        if remark:
            buy_price = float(getattr(t, "buy_price", 0))
            sell_price = float(getattr(t, "sell_price", 0))
            quantity = int(getattr(t, "buy_qty", 0))
            
            # Calculate gross PnL dynamically on export
            gross_pnl = (sell_price - buy_price) * quantity

            trade_rows.append({
                "trade_id": f"LIVE_{idx+1:04d}",
                "date": str(pd.to_datetime(getattr(t, "buy_timestamp")).date()),
                "side": getattr(t, "side", getattr(t, "Side", "")),
                "buy_timestamp": str(getattr(t, "buy_timestamp")),
                "sell_timestamp": str(getattr(t, "sell_timestamp")),
                "buy_price": round(buy_price, 2),
                "sell_price": round(sell_price, 2),
                "quantity": quantity,
                "gross_pnl": round(gross_pnl, 2),
                "charges": round(float(getattr(t, "charges", 0)), 2),
                "net_pnl": round(float(getattr(t, "pnl", 0)), 2),
                "remark": remark,
                "running_capital": round(float(getattr(t, "total", 0)), 2)
            })

    if trade_rows:
        df_export = pd.DataFrame(trade_rows)
        csv_path = REPORTS_DIR / "livetrader_trades.csv"
        parquet_path = REPORTS_DIR / "livetrader_trades.parquet"
        df_export.to_csv(csv_path, index=False)
        df_export.to_parquet(parquet_path, index=False)
        print(f"\n📊 Live Trader reports successfully saved to:\n  - {csv_path}\n  - {parquet_path}")
    else:
        print("\n⚠️ No closed trades found to export.")

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
    
    # Capital constraints initialized securely here
    prtf = Portfolio(funds=Funds(starting_capital=BEST_PARAMS["initial_capital"]))
    feeder_queue = asyncio.Queue(maxsize=10)
    strat = ana.Strategy()

    trader = ana.Trader(
        portfolio=prtf,
        strategy=strat,
        broker=ana.LiveBroker(ustox) if args.command == "live" else ana.SimBroker(portfolio=prtf),
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
        _export_livetrader_reports(trader)
    except Exception as e:
        logger.exception(f"Exception {e}")


if __name__ == "__main__":
    start_heartbeat(component_name='Live Trader', port=5558)
    main()