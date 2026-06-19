from collections import defaultdict
from functools import partial
from typing import Literal
from copy import deepcopy
from pathlib import Path
import asyncio
import datetime as dt
import pandas as pd
import logging
import shutil
import json
import zmq
import zmq.asyncio
import sys
from tqdm import tqdm

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
MATRIX_CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache" / "matrices"
MATRIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(
    "--------------------HYBRID LIVE TRADING ENGINE--------------------".center(
        shutil.get_terminal_size().columns
    )
)

# =====================================================================
# THE GEAR BOX: OPTUNA VALIDATED PARAMETERS
# =====================================================================
SNIPE_BEST_PARAMS = {
    'rsi_min': 47,
    'adx_min': 61,
    'use_ema': True,
    'use_supertrend': False,
    'req_active_slope': True,
    'use_macd': False,
    'bb_max_width': 0.014498753585377278,
    'sl_atr': 1.5,
    'target_atr': 8.0,
    'trailing_sl_atr': 1.5
}

SCALP_BEST_PARAMS = {
    'rsi_min': 45,
    'adx_min': 20,
    'use_ema': True,
    'use_supertrend': False,
    'req_active_slope': False,
    'use_macd': False,
    'bb_max_width': 0.06897211941654376,
    'sl_atr': 1.3,
    'target_atr': 2.0,
    'trailing_sl_atr': 0.3,
    'max_daily_trades': 50,
    'max_daily_profit': 18301.0
}


# =====================================================================
# THE EXECUTOR
# =====================================================================
async def executor(trader: ana.Trader, bucket: Bucket, ui_socket: zmq.asyncio.Socket = None, **kwargs):
    spot_key = next((k for k in trader.instruments.keys() if "INDEX" in k[0] or trader.instruments[k].type == "SPOT"), None)
    call_option = bucket.legs.get("CE")
    put_option = bucket.legs.get("PE")

    if not spot_key or not call_option or not put_option: 
        return

    spot_inst = trader.instruments[spot_key]
    ce_inst = trader.instruments[call_option.key]
    pe_inst = trader.instruments[put_option.key]

    spot_df = spot_inst.historical_candles
    ce_df = ce_inst.historical_candles
    pe_df = pe_inst.historical_candles

    if len(spot_df) < 2 or ce_df.empty or pe_df.empty: 
        return

    spot_row = spot_df.iloc[-1]
    spot_prev = spot_df.iloc[-2]
    timestamp = spot_df.index[-1] if isinstance(spot_df.index, pd.DatetimeIndex) else pd.Timestamp.now()

    if ui_socket:
        payload = {spot_key[0]: spot_row.to_json(date_format="iso")}
        await ui_socket.send_string(json.dumps(payload))

    # ---------------------------------------------------------
    # GEAR SHIFTER LOGIC
    # ---------------------------------------------------------
    today = timestamp.date()
    daily_trades = [t for t in trader.portfolio.report if pd.Timestamp(t.Buy_timestamp).date() == today]
    
    daily_realized_pnl = sum(getattr(t, 'PnL', 0) for t in daily_trades if getattr(t, 'Remark', '') in ["T", "SL", "EOD"])
    num_trades_today = len(daily_trades)

    if daily_realized_pnl >= SCALP_BEST_PARAMS['max_daily_profit'] or num_trades_today >= SCALP_BEST_PARAMS['max_daily_trades']:
        ACTIVE_PARAMS = SNIPE_BEST_PARAMS
        gear_name = "SNIPER"
    else:
        ACTIVE_PARAMS = SCALP_BEST_PARAMS
        gear_name = "SCALP"

    # ---------------------------------------------------------
    # POSITION MANAGER (Ratchet Trailing Stop)
    # ---------------------------------------------------------
    if bucket.open_position is not None:
        pos = bucket.open_position
        is_ce = pos.instrument_token == call_option.key
        active_opt = call_option if is_ce else put_option
        active_df = ce_df if is_ce else pe_df
        active_row = active_df.iloc[-1]
        
        high, low, close = active_row.get("high", 0), active_row.get("low", 0), active_row.get("close", 0)

        # Initialize tracking metrics on the fly if not preset
        if not hasattr(pos, 'highest_seen'):
            pos.highest_seen = close
            atr = active_row.get("ATRr_14", 1.0)
            pos.trail_dist = ACTIVE_PARAMS.get('trailing_sl_atr', 1.5) * atr
            
        target_hit = high >= pos.target
        stoploss_hit = low <= pos.stoploss
        eod_square_off = timestamp.time() >= dt.time(15, 15)

        # The Pessimistic Check Order (SL first, Target second)
        if stoploss_hit or target_hit or eod_square_off:
            exit_price = pos.stoploss if stoploss_hit else (pos.target if target_hit else close)
            remark = "SL" if stoploss_hit else ("T" if target_hit else "EOD")
            
            status = trader.broker.sell_order(
                key=active_opt.key,
                qty=trader.portfolio.report[-1].Buy_qty,
                price=exit_price,
            )
            
            if status != -1:
                report: Trade = trader.portfolio.report[-1]
                report.Sell_qty = report.Buy_qty
                report.Sell_timestamp = timestamp
                report.Sell_price = exit_price
                report.Remark = remark
                report.Movement = report.Sell_price - report.Buy_price
                report.PnL = (report.Sell_price * report.Sell_qty) - (report.Buy_price * report.Buy_qty)
                report.total = trader.portfolio.funds.total
                bucket.open_position = None
                trader.portfolio.funds.settle()
                logger.info(f"[{gear_name}] Position Closed: {remark} | PnL: ₹{report.PnL:.2f}")
        else:
            # Shift trailing stop upwards
            if high > pos.highest_seen:
                pos.highest_seen = high
                new_sl = pos.highest_seen - pos.trail_dist
                pos.stoploss = max(pos.stoploss, new_sl)
        return

    # ---------------------------------------------------------
    # DYNAMIC SIGNAL GENERATOR
    # ---------------------------------------------------------
    curr_time = timestamp.time()
    if (dt.time(10, 0) < curr_time < dt.time(12, 0)) or (curr_time > dt.time(15, 0)):
        return

    rsi = spot_row.get("RSI_14", 50)
    adx = spot_row.get("ADX_14", 0)
    spot_close = spot_row.get("close", 0)
    
    supert_slope = spot_row.get("SUPERT_14_2.0", 0) - spot_prev.get("SUPERT_14_2.0", 0)
    bbu, bbl = spot_row.get("BBU_20_2.0", 0), spot_row.get("BBL_20_2.0", 0)
    bb_width = ((bbu - bbl) / spot_close) if spot_close > 0 else 1.0

    ce_cond = (rsi > ACTIVE_PARAMS['rsi_min']) and (adx > ACTIVE_PARAMS['adx_min'])
    pe_cond = (rsi < (100 - ACTIVE_PARAMS['rsi_min'])) and (adx > ACTIVE_PARAMS['adx_min'])

    if ACTIVE_PARAMS['use_ema']:
        ema_200 = spot_row.get("EMA_200", 0)
        ce_cond = ce_cond and (spot_close > ema_200)
        pe_cond = pe_cond and (spot_close < ema_200)

    if ACTIVE_PARAMS['use_supertrend']:
        supert_dir = spot_row.get("SUPERTd_14_2.0", 0)
        ce_cond = ce_cond and (supert_dir == 1)
        pe_cond = pe_cond and (supert_dir == -1)

    if ACTIVE_PARAMS['req_active_slope']:
        ce_cond = ce_cond and (supert_slope > 0)
        pe_cond = pe_cond and (supert_slope < 0)
        
    if ACTIVE_PARAMS.get('use_macd', False):
        macd = spot_row.get("MACD_12_26_9", 0)
        macds = spot_row.get("MACDs_12_26_9", 0)
        ce_cond = ce_cond and (macd > macds)
        pe_cond = pe_cond and (macd < macds)

    ce_cond = ce_cond and (bb_width < ACTIVE_PARAMS['bb_max_width'])
    pe_cond = pe_cond and (bb_width < ACTIVE_PARAMS['bb_max_width'])

    # ---------------------------------------------------------
    # ORDER EXECUTION
    # ---------------------------------------------------------
    def place_buy_order(opt_leg: Instrument, opt_row: pd.Series):
        atr = opt_row.get("ATRr_14", 0)
        if atr == 0: return -1
        
        close_price = opt_row.get("close", 0)
        qty = trader.calculate_units(close=close_price, lot_size=opt_leg.lot_size)
        if qty == 0: return -1

        status = trader.broker.buy_order(key=opt_leg.key, price=close_price, qty=qty)
        if status is None: return -1

        pos, ord_id = status[0], status[1]
        bucket.open_position = pos
        
        # Inject custom Optuna Exits into position block
        bucket.open_position.target = close_price + (ACTIVE_PARAMS['target_atr'] * atr)
        bucket.open_position.stoploss = close_price - (ACTIVE_PARAMS['sl_atr'] * atr)
        bucket.open_position.trail_dist = ACTIVE_PARAMS.get('trailing_sl_atr', 1.5) * atr
        bucket.open_position.highest_seen = close_price

        trader.portfolio.report.append(
            Trade(
                Trade_id=ord_id, 
                Instrument_key=opt_leg.key, 
                Buy_timestamp=timestamp,
                Side=opt_leg.type, 
                Buy_price=close_price, 
                Buy_qty=qty, 
                Buy_conditions=spot_row.to_dict()
            )
        )
        logger.info(f"[{gear_name}] GEAR TRIGGERED: Buy {opt_leg.type} @ {close_price:.2f}")
        return 1

    if ce_cond:
        place_buy_order(call_option, ce_df.iloc[-1])
    elif pe_cond:
        place_buy_order(put_option, pe_df.iloc[-1])


# =====================================================================
# THE PROCESSOR
# =====================================================================
async def processor(trader: ana.Trader, bucket: Bucket, stopevent: asyncio.Event, ui_socket=None, **kwargs):
    spot_key = next((k for k in trader.instruments.keys() if "INDEX" in k[0] or trader.instruments[k].type == "SPOT"), None)
    trading_items = [data.key for data in bucket.legs.values()]
    if spot_key: 
        trading_items.append(spot_key[0])

    logger.debug(f"Hybrid tick processor started.")
    prev_ticks = {}
    
    while not stopevent.is_set():
        ticks: dict[str, Tick] = await trader.datafeed.get()

        if ticks == prev_ticks:
            continue
        prev_ticks = ticks
        
        keys = list(ticks.keys())
        if not any(k[0] in trading_items for k in keys):
            logger.info("Setting stop event since no keys match.")
            stopevent.set()
            break

        # Append incoming data to local cache
        for key, tick in ticks.items():
            if key in trader.instruments.keys():
                trader.instruments[key].historical_candles.append(
                    tick.ohlc_1m if isinstance(tick, Tick) else tick
                )

        # Apply strategy calculations in parallel
        tasks = [
            asyncio.to_thread(trader.strategy.apply, val.historical_candles, key=key)
            for key, val in trader.instruments.items() if key[0] in trading_items
        ]
        await asyncio.gather(*tasks)
        
        # Fire unified executor
        await executor(trader=trader, bucket=bucket, ui_socket=ui_socket)


# =====================================================================
# MAIN LOOP
# =====================================================================
def main():
    args = setup_cli()
    capital: Funds = Funds.update_from_json(ustox.get_funds())
    prtf = Portfolio(funds=capital)
    feeder_queue = asyncio.Queue(maxsize=10)
    strat = ana.Strategy()

    trader = ana.Trader(
        portfolio=prtf,
        strategy=strat,
        broker=ana.SimBroker(portfolio=prtf),
        datafeed=feeder_queue,
    )

    def setup_mode(mode: Literal["sim", "live"], key: str | list[str]):
        if mode == "live":
            _, insts = ustox.get_options_with_expiry(ustox.get_all_options(instrument_key=key), return_df=True)
            market_quote = ustox.get_marketquote(instrument_key=insts["instrument_key"].to_list())
            best_ce, best_pe = None, None
            max_ce_volume, max_pe_volume = -1, -1

            for _, quote in market_quote.items():
                opt_type = insts.loc[insts["instrument_key"] == quote["instrument_token"], "instrument_type"].item()
                volume = quote.get("volume", 0)
                if opt_type == "CE" and volume > max_ce_volume:
                    max_ce_volume, best_ce = volume, quote.get("instrument_token")
                elif opt_type == "PE" and volume > max_pe_volume:
                    max_pe_volume, best_pe = volume, quote.get("instrument_token")

            options = insts[(insts["instrument_key"] == best_ce) | (insts["instrument_key"] == best_pe)]
            insts = Instrument.parse_options(client=ustox, options=options, lookback=2, is_expired=False)
            
            # Mount Spot Data explicitly in Live Mode
            spot_inst = Instrument(key=key, type="SPOT", name="Nifty 50")
            spot_inst.historical_candles = ustox.get_historical_candle(key, "1minute")
            insts.append(spot_inst)

        elif mode == "sim":
            files = []
            if args.tickwise:
                dir = Path(args.tickwise).resolve().absolute()
                files.extend([file for file in dir.iterdir()])
            if args.bulk:
                for dir in args.bulk:
                    dir = Path(dir).resolve().absolute()
                    files.extend([file for file in dir.iterdir()])
            files.sort()
            # The load_multiple handles the INDEX file now that we removed the regex block
            insts = Instrument.load_multiple(client=ustox, source=files, lookback=2)
            
        insts_dict = {(item.key, item.date): item for item in insts}
        trader.add_instrument(insts_dict)
        return deepcopy(insts_dict)

    tradable_insts = setup_mode(args.command, args.index)
    daily_buckets = defaultdict(dict)

    for instrument in tqdm(trader.instruments.values(), desc="Filtering instruments", leave=False):
        if "INDEX" in instrument.key[0] or instrument.type == "SPOT": continue
        leg_type = "CE" if "CE" in instrument.type else "PE"
        daily_buckets[instrument.date][leg_type] = instrument
        
    for trade_date, legs in tqdm(daily_buckets.items(), desc="Loading Buckets", leave=False):
        bucket = Bucket(trade_date, legs=legs)
        trader.buckets.append(bucket)
        break

    # Require all parameters for the AI evaluations
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

    async def starter():
        try:
            stopevent = asyncio.Event()
            
            # Fetch Keys to broadcast into datafeed
            trading_items = [data.key for data in trader.buckets[-1].legs.values()]
            spot_key = next((k for k in trader.instruments.keys() if "INDEX" in k[0] or trader.instruments[k].type == "SPOT"), None)
            if spot_key: trading_items.append(spot_key[0])
                
            trading_insts = {k: v for k, v in tradable_insts.items() if k[0] in trading_items}
            
            if args.command == "live":
                logger.info("Live streamer created.")
                simfeeder = ana.LivefeedStreamer(ustox, trading_insts, feeder_queue)
            elif args.command == "sim":
                logger.info("Simulation streamer created.")
                simfeeder = ana.SimfeedStreamer(trading_insts, feeder_queue, stopevent)
                
            tasks = [simfeeder.start()]
            ui_socket = None

            if args.tui:
                port = "tcp://127.0.0.1:5556"
                context = zmq.asyncio.Context()
                ui_socket = context.socket(zmq.PUB)
                ui_socket.bind(port)
                logger.info("Broadcasting UI data to port 5556")
                
                app = tui.TradingTUI(trader, trader.buckets[-1], simulate=True, client=ustox, port=port)
                tasks.append(app.run_async())

            # Start processor (which automatically executes trades)
            tasks.append(processor(trader, trader.buckets[-1], stopevent, ui_socket=ui_socket))
            
            tasks_to_run = [asyncio.create_task(task) for task in tasks]
            await asyncio.gather(*tasks_to_run, return_exceptions=True)
            
        except Exception as e:
            logger.exception(f" Exception while running livetrader |\n {e}")
        finally:
            for task in tasks_to_run:
                task.cancel()
                logger.info(f"Cancelled async tasks: {task.get_coro()}.")
            if args.tui:
                ui_socket.close()
                context.term()

    try:
        asyncio.run(starter())
    except Exception as e:
        logger.exception(f"Exception {e}")

if __name__ == "__main__":
    main()