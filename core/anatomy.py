# ------Import Libraries------
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from tqdm import tqdm
from collections import deque
from functools import partial
from datetime import datetime, date, timedelta
from core import upstox_func as ustox
from pathlib import Path
import pandas_ta as ta
import pandas as pd
import numpy as np
import logging
import traceback
import asyncio
import weakref
import config
import magic
import time
import json
import csv
import os

global call_data, put_data, scrip_data
ticks_ready = asyncio.Event()
pool = ThreadPoolExecutor()
logger = logging.getLogger(__name__)


@dataclass
class Plotdata:
    key: str
    stoploss: float = 0.0
    target: float = 0.0
    buys: list = field(default_factory=list)
    sells: list = field(default_factory=list)
    open: list = field(default_factory=list)
    high: list = field(default_factory=list)
    low: list = field(default_factory=list)
    close: list = field(default_factory=list)
    timestamp: deque = field(default_factory=lambda: deque(maxlen=1000))
    ema: list = field(default_factory=lambda: [[], []])  # emashort, emalong
    supertrend: list = field(default_factory=list)


@dataclass
class Tick:
    key: str
    vol: int = 0.0
    oi: int = 0
    close: float = np.nan
    open: float = np.nan
    high: float = np.nan
    low: float = np.nan
    timestamp: str = "0"
    depth: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.tick_lock = asyncio.Lock()

    async def update(self, buffer):
        async with self.tick_lock:
            try:
                if isinstance(buffer, dict):
                    if "marketFF" in buffer.keys():
                        self.vol = int(buffer["marketFF"]["vtt"])
                        self.oi = int(buffer["marketFF"]["oi"])
                        self.close = float(buffer["marketFF"]["ltpc"]["ltp"])
                        self.timestamp = buffer["marketFF"]["ltpc"]["ltt"]
                        self.depth = pd.DataFrame(
                            buffer["marketFF"]["marketLevel"]["bidAskQuote"]
                        )
                        for cols in self.depth.columns:
                            self.depth[cols] = pd.to_numeric(
                                self.depth[cols], errors="coerce"
                            )
                    elif "indexFF" in buffer.keys():
                        self.close = float(buffer["indexFF"]["ltpc"]["ltp"])
                        self.timestamp = buffer["indexFF"]["ltpc"]["ltt"]

                elif isinstance(buffer, pd.DataFrame):
                    if not buffer["close"]:
                        raise ValueError(
                            "The input data must contain OHLC data or close values at the least."
                        )
                    self.open = buffer["open"]
                    self.close = buffer["close"]
                    self.oi = buffer["oi"] if buffer["oi"] else 0
                    self.high = buffer["high"] if buffer["high"] else 0
                    self.low = buffer["low"] if buffer["low"] else 0
                    self.timestamp = buffer["timestamp"] if buffer["time"] else 0
                    self.vol = buffer["vol"]

                elif isinstance(buffer, pd.Series):

                    if not buffer["close"]:
                        raise ValueError(
                            "The input data must contain OHLC data or close values at the least."
                        )
                    self.close = buffer["close"]
                    self.open = buffer["open"]
                    self.low = buffer["low"]
                    self.high = buffer["high"]
                    self.oi = buffer["oi"]
                    self.timestamp = buffer["timestamp"]
                    self.vol = buffer["vol"]
                else:
                    raise Exception(
                        f"Error while reading data into ticks.Buffer is of type {type(buffer)}"
                    )
            except Exception as e:
                logger.exception(e)

    def copy(self):
        return dict(
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            vol=self.vol,
            oi=self.oi,
            key=self.key,
            depth=self.depth.copy(),
        )


class Position:
    key: str = ""
    symbol: str = ""
    open: bool = False
    cost: float = 0
    charges: float = 0
    buy_price: float = 0
    sell_price: float = 0
    quantity: int = 0
    unrealised: float = 0
    pnl: float = 0
    value: float = 0
    history: list = []

    def __init__(self):
        pass

    @classmethod
    def open_position(cls):
        pos = ustox.get_positions()
        if len(pos) > 0:
            lastpos = pos[-1]
            cls.open = True
            cls.quantity = lastpos["day_buy_quantity"]
            cls.value = lastpos["value"]
            cls.buy_price = lastpos["buy_price"]
            cls.cost = (cls.buy_price * cls.quantity) + cls.charges
            cls.unrealised = sum(float(item["unrealised"]) for item in pos)
            cls.key = lastpos["instrument_token"]
            cls.symbol = lastpos["trading_symbol"]
            cls.pnl = sum(float(item["realised"]) for item in pos)
        else:
            return

    @classmethod
    def snapshot(cls):
        return {
            "key": cls.key,
            "symbol": cls.symbol,
            "open": cls.open,
            "cost": cls.cost,
            "buy_price": cls.buy_price,
            "sell_price": cls.sell_price,
            "quantity": cls.quantity,
            "unrealised": cls.unrealised,
            "pnl": cls.pnl,
            "value": cls.value,
        }

    @classmethod
    def close_position(cls):
        pos = ustox.get_positions()
        if sum([item["unrealised"] for item in pos]) > 0:
            open_pos = [
                item["instrument_token"]
                for item in pos
                if float(item["unrealised"]) > 0
            ]
            logger.info(f"Position still open. Cannot close position.: {open_pos}")
            return
        if len(pos) > 0:
            pos = pos[-1]
            cls.sell_price = pos["sell_price"]
            cls.history.append(cls.snapshot())
            cls.key = ""
            cls.symbol = ""
            cls.open = False
            cls.cost = 0
            cls.buy_price = 0
            cls.sell_price = 0
            cls.quantity = 0
            cls.unrealised = 0
            cls.value = 0


class Funds:
    opening: int = 0
    balance: int = 0
    used: int = 0
    totalpnl: int = 0

    def __init__(self):
        pass

    @classmethod
    def update_funds(cls):
        funds = ustox.get_funds()
        if funds:
            funds = funds["equity"]
            cls.balance = funds["available_margin"]
            cls.used = funds["used_margin"]
            cls.totalpnl = int(Position.pnl + Position.unrealised)
        else:
            return


class Trader:
    _instances = weakref.WeakSet()
    lock = asyncio.Lock()
    scrip_pointer = None

    def __init__(self, key, trades, gui, tag, simdate=None, underlying=None) -> None:
        try:
            self.simdate = simdate
            self.gui = gui
            self.tag = tag
            if self.tag == "underlying":
                Trader.scrip_pointer = self
            self._instances.add(self)
            self.key = key
            if "INDEX" in self.key:
                self.trade_flag = False
            else:
                self.trade_flag = True
            self.trades = trades
            self.tick = Tick(self.key)
            self.last_ts = "0"
            self.rsi_flag = False
            self._tickqueue = None
            self._history_lock = None
            self.history = None
            self.prev_trading_day = simdate - timedelta(days=1)
            while ustox.is_nse_holiday(self.prev_trading_day, ustox.holidays):
                logger.info(f"Skipping holiday/weekend : {self.prev_trading_day}")
                self.prev_trading_day -= timedelta(days=1)

            self.cookie_path = (
                ustox.directory + f"/cookies/{self.key}_{self.prev_trading_day}"
            )
            simdir = (
                ustox.directory
                + f"/sim_database/{self.prev_trading_day}/{self.key if not 'INDEX' in self.key else 'NSE_INDEX'}.json"
            )

            if os.path.exists(self.cookie_path):
                self.history = pd.read_json(self.cookie_path, convert_dates=False)
            elif os.path.exists(simdir):
                self.history = pd.read_json(simdir,convert_dates=False)
            else:
                self.history = ustox.get_historical(
                    dtype="historical",
                    instrument_key=self.key,
                    from_date=self.prev_trading_day,
                    to_date=(
                        self.prev_trading_day
                        if config.market_close.is_set()
                        else simdate
                    ),
                    interval=15,
                )
                self.history = (
                    self.history
                    if (not self.history is None and not self.history.empty)
                    else ustox.get_historical(
                        dtype="historical",
                        instrument_key=self.key,
                        to_date=self.prev_trading_day,
                        from_date=self.prev_trading_day,
                        is_expired=True,
                        interval=15,
                    )
                )
                if self.history is None or self.history.empty:
                    logger.error(
                        f"Empty Dataframe for the {self.key} for simdate : {self.simdate}"
                    )
                else:
                    self.history.to_json( 
                        self.cookie_path
                    )

            self.ohlc = self.history[["open", "high", "low", "close", "vol", "oi"]].copy()
            if self.history is None or self.history.empty:
                logger.error(
                    f"Failed to retrieve historical data for {self.key} for date : {self.prev_trading_day}.| Simulation date : {self.simdate} Cannot proceed with analysis."
                )

        except KeyError as e:
            logger.critical(
                f"KeyError while setting up analyse function | \n{e} | current date {simdate} | previous trading day {self.prev_trading_day}",
                stack_info=True,exc_info=True
            )
        except Exception as e:
            logger.exception(
                f"Exception while setting up analyse function | \n{e} | current date {simdate} | previous trading day {self.prev_trading_day} "
            )

    @property
    def tickqueue(self):
        """Lazy initializer for the asyncio Queue"""
        if self._tickqueue is None:
            self._tickqueue = asyncio.Queue(maxsize=10)
        return self._tickqueue

    @property
    def history_lock(self):
        """Lazy initializer for the asyncio Lock"""
        if self._history_lock is None:
            self._history_lock = asyncio.Lock()
        return self._history_lock

    """Indicator Functions"""

    def compute_supertrend(
        data: pd.DataFrame, length: int = 7, multiplier: float = 3.0, to_list=False
    ):
        supertrend = ta.supertrend(
            high=data["high"],
            low=data["low"],
            close=data["close"],
            length=length,
            multiplier=multiplier,
        )
        if supertrend is None:
            return [None] * 4

        if to_list:
            main = supertrend[f"SUPERT_{length}_{multiplier}"].values.tolist()
            direction = supertrend[f"SUPERTd_{length}_{multiplier}"].values.tolist()
            long_band = supertrend[f"SUPERTl_{length}_{multiplier}"].values.tolist()
            short_band = supertrend[f"SUPERTs_{length}_{multiplier}"].values.tolist()
            return main, direction, long_band, short_band
        main = supertrend[f"SUPERT_{length}_{multiplier}"].iloc[-1]
        main = main if not np.isnan(main) else None
        direction = supertrend[f"SUPERTd_{length}_{multiplier}"].iloc[-1]
        direction = direction if not np.isnan(direction) else None
        long_band = supertrend[f"SUPERTl_{length}_{multiplier}"].iloc[-1]
        long_band = long_band if not np.isnan(long_band) else None
        short_band = supertrend[f"SUPERTs_{length}_{multiplier}"].iloc[-1]
        short_band = short_band if not np.isnan(short_band) else None
        return main, direction, long_band, short_band

    @staticmethod
    def superindicator(close: pd.Series, depth_history: pd.DataFrame):
        if len(depth_history) < 2:
            return False, False
        qty_df = pd.concat(
            [item for item in depth_history["depth"].iloc[-100:]]
        ).reset_index(drop=True)
        quantities = pd.concat([qty_df["bidQ"], qty_df["askQ"]], axis=0)
        avg_size = np.percentile(quantities.iloc[-100:], 98)

        def depth_handler(depth):
            bids = depth[["bidP", "bidQ"]].rename(
                columns={"bidP": "Price", "bidQ": "Bid"}
            )
            asks = depth[["askP", "askQ"]].rename(
                columns={"askP": "Price", "askQ": "Ask"}
            )
            bids.set_index("Price", inplace=True)
            asks.set_index("Price", inplace=True)
            bids["Rank"] = bids["Bid"].rank(pct=True) * 100
            asks["Rank"] = asks["Ask"].rank(pct=True) * 100
            return bids, asks

        def depth_wall(bid, ask, range=0):
            if range > 0:
                ask_wall = ask.iloc[:range]["Ask"].sum()
                bid_wall = bid.iloc[:range]["Bid"].sum()
            else:
                ask_wall = ask["Ask"].sum()
                bid_wall = bid["Bid"].sum()

            return bid_wall, ask_wall

        prev_ask_walls = []
        prev_bid_walls = []

        ask_volume = []
        bid_volume = []

        current_bids, current_asks = depth_handler(depth_history["depth"].iloc[-1])

        for item in depth_history["depth"].iloc[-25:]:
            bids, asks = depth_handler(item)
            bid_wall, ask_wall = depth_wall(bids, asks, 25)
            prev_bid_walls.append(bid_wall)
            prev_ask_walls.append(ask_wall)

        for item in depth_history["depth"].iloc[-25:]:
            bids, asks = depth_handler(item)
            bid_wall, ask_wall = depth_wall(bids, asks)
            bid_volume.append(bid_wall)
            ask_volume.append(ask_wall)

        avg_bid_wall_change = np.mean(np.diff(np.array(prev_bid_walls)))
        avg_ask_wall_change = np.mean(np.diff(np.array(prev_ask_walls)))

        avg_bid_change = np.mean(np.diff(np.array(bid_volume)))
        avg_ask_change = np.mean(np.diff(np.array(ask_volume)))

        supports = current_bids[current_bids["Bid"] > avg_size]
        resistances = current_asks[current_asks["Ask"] > avg_size]

        if (bid_volume[-1] - bid_volume[-2]) > 3 * avg_bid_change and (
            prev_bid_walls[-2] - prev_bid_walls[-1]
        ) > 3 * avg_bid_wall_change:
            predicted_val = resistances.index.values.tolist()

        elif (ask_volume[-1] - ask_volume[-2]) > 3 * avg_ask_change and (
            prev_ask_walls[-2] - prev_ask_walls[-1]
        ) > 3 * avg_ask_wall_change:
            predicted_val = supports.index.values.tolist()

        else:
            predicted_val = None

        buy_cond = False
        sell_cond = False
        return buy_cond, sell_cond, predicted_val

    def compute_ema_cross(close: pd.Series, fast, slow, to_list=False):
        ema_fast = ta.ema(close=close, length=fast)
        ema_slow = ta.ema(close=close, length=slow)
        if to_list:
            return ema_fast.values.tolist() if not ema_fast is None else [np.nan], (
                ema_slow.values.tolist() if not ema_slow is None else [np.nan]
            )
        ema_fast = ema_fast.dropna().iloc[-1] if ema_fast is not None else None
        ema_slow = ema_slow.dropna().iloc[-1] if ema_slow is not None else None
        return ema_fast, ema_slow

    @staticmethod
    def resample_data(data: pd.DataFrame | pd.Series, period: int):
        if len(data) < period:
            raise ValueError("Not enough data to resample")
        open = []
        low = []
        high = []
        close = []
        if isinstance(data, pd.DataFrame):
            while len(data) >= period:
                tempdata = data.iloc[:period]
                open.append(tempdata["close"].iloc[0])
                low.append(min(tempdata["close"]))
                high.append(max(tempdata["close"]))
                close.append(tempdata["close"].iloc[-1])
                data = data.iloc[period:]
        elif isinstance(data, pd.Series):
            tempdata = data.iloc[:period]
            open.append(tempdata.iloc[0])
            low.append(tempdata.min())
            high.append(tempdata.max())
            close.append(tempdata.iloc[-1])
            data = data.iloc[period:]

        return pd.DataFrame(
            {
                "open": open,
                "high": high,
                "low": low,
                "close": close,
            }
        )

    @staticmethod
    def compute_adx(data: pd.DataFrame, window, to_list=False):
        adx = ta.adx(
            high=data["high"], low=data["low"], close=data["close"], length=window
        )
        if adx is None:
            return [None, None, None] if not to_list else [np.nan, np.nan, np.nan]
        if to_list:
            return [
                adx[f"ADXR_{window}_2"].values.tolist(),
                adx[f"DMP_{window}"].values.tolist(),
                adx[f"DMN_{window}"].values.tolist(),
            ]
        adx = adx.dropna().iloc[-1] if adx.dropna().size > 0 else None
        return (
            [adx[f"ADXR_{window}_2"], adx[f"DMP_{window}"], adx[f"DMN_{window}"]]
            if adx is not None
            else [None, None, None]
        )

    @staticmethod
    def logic_check(data: pd.DataFrame, flag, step=0):
        if flag:
            return
        vols = pd.Series(data["vol"].dropna())
        vol_median = vols.rolling(20).median()
        mad = (vols - vol_median).abs().rolling(20).median()
        mod_z = 0.6745 * (vols - vol_median) / mad
        if not pd.isna(mod_z.iloc[-2]):
            if mod_z.iloc[-2] > 3.5:
                print(vols.values.tolist()[-5:])

    @staticmethod
    def compute_rsi(data: pd.Series, length=14, to_list=False, simple_rsi=False):
        import pandas_ta as ta

        if not simple_rsi and data.shape[0] <= 101:
            return [np.nan] if to_list else None
        if simple_rsi:
            rsi = ta.rsi(data, length=length)
        else:
            rsi = ta.crsi(data, length=length)
        if to_list:
            return rsi.values.tolist() if rsi is not None else [np.nan]
        rsi = rsi.iloc[-1] if rsi is not None else None
        return rsi if rsi is not None else None

    """Main Analysis Function"""

    def sell_order(self, hit_stoploss=False):
        self.units = self.bought_units
        sell_ts = datetime.fromtimestamp((int(self.tick.timestamp) / 1000)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        if Position.key == "":
            self.gui.logconsole.log_error("No open orders. Sell order not executed.")
            return 1
        if Position.key != self.key:
            self.gui.logconsole.log_error(
                "Discrepancy in Open Position key and Trader key"
            )
            return 0
        order_status = ustox.place_order(
            self.key, "SELL", quantity=self.units, sandbox=config.sandbox_orders
        )
        if order_status.status_code == 200 or self.gui.simulation:
            try:
                order_id = order_status.json()["data"]["order_ids"][-1]
            except KeyError:
                order_id = "111"
            details = (
                ustox.get_order_details(order_id=order_id)
                if not config.sandbox_orders
                else {"status": "complete"}
            )
            if config.market_close.is_set() or details["status"] == "complete":
                self.gui.logconsole.log_info(
                    f"{sell_ts} | {self.key} : Sell order placed for {self.units} units at {self.tick.close if not hit_stoploss else self.stoploss}: {order_id}"
                )
            else:
                self.gui.logconsole.log_error(
                    f"Failed to place order. {details["status_message"]}"
                )
                if self.newtick:
                    self.trades.sells.append(np.nan)
                return 1
        else:
            self.gui.logconsole.log_error(
                f"Failed to place order. {order_status.json()}"
            )
            return 1
        movement = (
            (self.tick.close - self.last_buy)
            if not hit_stoploss
            else (self.stoploss - self.last_buy)
        )
        self.gui.logconsole.log_debug(
            f"Movement : {movement}",
            f"{self.buy_ts} - {sell_ts},{self.simdate},{self.tag},{self.last_buy},{self.tick.close if not hit_stoploss else self.stoploss},{self.stoploss},{self.tick.close},{movement},{self.units},{movement*self.units},{hit_stoploss},{self.adx_str.replace('(','').replace(")",'')}",
        )
        if config.market_close.is_set():
            Position.open = False
            pl = (self.tick.close - self.last_buy) * self.units
            Funds.balance = round(Funds.balance + pl, 2)
            Position.key = ""
            Funds.totalpnl = +round(pl, 2)
            self.gui.updatefunds(val=Funds.balance)
        else:
            Position.close_position()
            Funds.update_funds()
            self.gui.updatepos()
            self.gui.updatefunds()
        if self.newtick:
            self.trades.sells.append(self.tick.close)
        else:
            self.trades.sells[-1] = self.tick.close

        self.trades.stoploss = self.stoploss = 0.0
        self.target = self.trades.target = 0.0
        self.last_buy = None

        return 0

    def buy_order(self, adx_str):
        if not Position.open and Position.charges < 3:
            order_status = ustox.place_order(
                self.key, "BUY", quantity=self.units, sandbox=config.sandbox_orders
            )
            if order_status.status_code == 200 or self.gui.simulation:
                try:
                    order_id = order_status.json()["data"]["order_ids"][-1]
                except KeyError:
                    order_id = "111"
                details = (
                    ustox.get_order_details(order_id=order_id)
                    if not config.sandbox_orders
                    else {"status": "complete"}
                )
                if config.market_close.is_set() or details["status"] == "complete":
                    try:
                        self.buy_ts = datetime.fromtimestamp(
                            (int(self.tick.timestamp) / 1000)
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        logger.error(
                            f"Error occurred while setting buy timestamp: {e} for timestamp {self.tick.timestamp} | {self.simdate} | {self.key}"
                        )
                        self.gui.logconsole.log_error(
                            f"Error occurred while setting buy timestamp: {e} for timestamp {self.tick.timestamp}"
                        )

                    self.gui.logconsole.log_info(
                        f"{self.buy_ts} | {self.key} : Buy order placed for {self.units} units at {self.tick.close}: {order_id}"
                    )
                    self.last_buy = self.tick.close
                else:
                    self.gui.logconsole.log_error(
                        f"Failed to place buy order. {details["status_message"]}"
                    )
                    if self.newtick:
                        self.trades.buys.append(self.tick.close)
                    return 1

            else:
                self.gui.logconsole.log_error(
                    f"Failed to place buy order. {order_status.json()}]"
                )
                if self.newtick:
                    self.trades.buys.append(self.tick.close)
                return 1
            self.stoploss = self.trades.stoploss = self.tick.close * 0.99
            self.target = self.trades.target = self.tick.close * 1.1
            self.gui.logconsole.log_info(f"STOPLOSS : {self.stoploss}")
            if self.newtick:
                self.trades.buys.append(self.tick.close)
            else:
                self.trades.buys[-1] = self.tick.close
            # brokerage = ustox.get_brokerage(
            #    price=self.tick.close,
            #    instrument_key=self.key,
            #    quantity=self.units,
            #    product="I",
            # )

            # Position.charges = Position.charges + (brokerage if brokerage else 90)
            Position.charges = Position.charges + 1
            self.gui.logconsole.log_info(f"{Position.charges}")
            if config.market_close.is_set() or self.gui.simulation:
                Position.open = True
                Position.key = self.key
                self.gui.updatepos()
                Funds.update_funds()
            else:
                Position.open_position()
                self.gui.updatefunds()
            self.adx_str = adx_str
            self.bought_units = self.units
            return 0
        elif Position.open:
            self.gui.logconsole.log_error(
                "A position is already open, cannot place a buy order."
            )
            return 1
        else:
            return 1

    async def analyse(self):
        loop = asyncio.get_running_loop()
        first = True
        count = 1
        temp_history = []
        self.prev_high = 0
        self.last_buy = None
        self.bought_units = None
        self.units = 0
        while not config.stopevent.is_set():
            try:
                #                if (Funds.balance - Funds.opening) >= config.target_profit:
                #                    self.gui.logconsole.log_info(
                #                        f"Target {config.target_profit} reached. Trading stopped."
                #                    )
                #                    config.stopevent.set()
                #                    self.gui.shutdown()
                if not config.stopevent.is_set():
                    await config.looprun.wait()
                if not config.stopevent.is_set():
                    try:
                        tick = await asyncio.wait_for(self.tickqueue.get(), timeout=3)
                    except TimeoutError:
                        if config.stopevent.is_set():
                            print(f"{self.tag} trader shutting down.")
                            break
                        else:
                            continue
                if not self.trade_flag:
                    if isinstance(tick, dict) and not tick:
                        continue
                if not config.stopevent.is_set():
                    await self.tick.update(tick)
                if self.last_ts == self.tick.timestamp:
                    continue
                self.last_ts = self.tick.timestamp
                self.units = config.units(300000, self.tick.close)
                temp_history.append(self.tick.close)
                try:
                    tick_condition = (
                        6 if self.ohlc.shape[0] < 60 or config.market_close.is_set() else 61
                    )
                except Exception as e:
                    logger.error(
                        f"Error occurred while setting tick condition: {e} for timestamp {self.tick.timestamp} | previous day {self.prev_trading_day} | current day {self.simdate} | {self.key}"
                    )
                    tick_condition = 6
                self.newtick = count == tick_condition
                if self.newtick:
                    count = 1
                    if self.ohlc.empty:
                        self.ohlc = Trader.resample_data(
                            pd.Series(temp_history), len(temp_history)
                        )
                    else:
                        self.ohlc = pd.concat(
                            [
                                self.ohlc,
                                Trader.resample_data(
                                    pd.Series(temp_history), len(temp_history)
                                ),
                            ],
                            axis=0,
                        )
                    temp_history = []
                self.ohlc["vol"] = self.ohlc["vol"].fillna(0)
                self.ohlc["oi"] = self.ohlc["oi"].fillna(0)
                async with self.history_lock:
                    if self.newtick:
                        self.history = pd.concat(
                            [self.history, self.ohlc], axis=0, ignore_index=True
                        )
                        if first:
                            self.trades.open = self.history["open"].values.tolist()
                            self.trades.high = self.history["high"].values.tolist()
                            self.trades.low = self.history["low"].values.tolist()
                            self.trades.close = self.history["close"].values.tolist()
                        else:
                            self.trades.open.append(self.history["open"].iloc[-1])
                            self.trades.high.append(self.history["high"].iloc[-1])
                            self.trades.low.append(self.history["low"].iloc[-1])
                            self.trades.close.append(self.history["close"].iloc[-1])

                    else:
                        common_cols = self.history.columns.intersection(
                            self.ohlc.columns
                        )
                        if self.history.empty:
                            continue
                        self.history.loc[self.history.index[-1], common_cols] = (
                            self.ohlc[common_cols].iloc[0]
                        )
                        if first:
                            self.trades.open = self.history["open"].values.tolist()
                            self.trades.high = self.history["high"].values.tolist()
                            self.trades.low = self.history["low"].values.tolist()
                            self.trades.close = self.history["close"].values.tolist()
                        else:
                            if self.tick.close > self.trades.high[-1]:
                                self.trades.high[-1] = self.tick.close
                            if self.tick.close < self.trades.low[-1]:
                                self.trades.low[-1] = self.tick.close
                            self.trades.close[-1] = self.tick.close
                if len(self.history) > 1000:
                    self.history = self.history.iloc[-1000:]

                async with self.tick.tick_lock, self.history_lock:
                    if first:
                        indicator_tasks = [
                            partial(
                                Trader.compute_ema_cross,
                                self.history["close"],
                                9,
                                26,
                                to_list=True,
                            ),
                            partial(
                                Trader.compute_supertrend,
                                self.ohlc,
                                14,
                                2,
                                to_list=True,
                            ),
                            partial(
                                Trader.compute_adx,
                                self.ohlc,
                                14,
                                to_list=True,
                            ),
                        ]
                        results = await asyncio.gather(
                            *[
                                loop.run_in_executor(pool, func)
                                for func in indicator_tasks
                            ]
                        )
                        ema, supertrend, adx = results
                        self.trades.buys.extend([np.nan] * len(self.history))
                        self.trades.sells.extend([np.nan] * len(self.history))
                        self.trades.ema[0].extend(ema[0])
                        self.trades.ema[1].extend(ema[1])
                        self.trades.supertrend.extend(supertrend[0])
                        ticks_ready.set()
                        first = False
                    else:
                        indicator_tasks = [
                            partial(
                                Trader.compute_ema_cross, self.history["close"], 9, 26
                            ),
                            partial(
                                Trader.compute_supertrend,
                                self.ohlc,
                                14,
                                2,
                                to_list=True,
                            ),
                        ]

                        results = await asyncio.gather(
                            *[
                                loop.run_in_executor(pool, func)
                                for func in indicator_tasks
                            ]
                        )

                        ema, supertrend = results
                        if self.newtick:
                            self.trades.ema[0].append(
                                ema[0] if not ema[0] is None else np.nan
                            )
                            self.trades.ema[1].append(
                                ema[1] if not ema[1] is None else np.nan
                            )
                            self.trades.supertrend.append(
                                supertrend[0][-1]
                                if not supertrend[0] is None
                                else np.nan
                            )
                        else:
                            self.trades.ema[0][-1] = (
                                ema[0] if not ema[0] is None else np.nan
                            )
                            self.trades.ema[1][-1] = (
                                ema[1] if not ema[1] is None else np.nan
                            )
                            self.trades.supertrend[-1] = (
                                supertrend[0][-1]
                                if not supertrend[0] is None
                                else np.nan
                            )
                        if (
                            all(
                                item is not None
                                for item in [ema[0], ema[1], supertrend[0]]
                            )
                            and not Position.open
                        ):
                            buy_cond = [
                                self.tick.close
                                > self.trades.supertrend[-1]
                                > self.trades.supertrend[-2],
                                20 < adx[0][-1] < 35,
                                adx[1][-1] > adx[2][-1],
                                abs(adx[1][-1] - adx[2][-1]) > 1,
                            ]
                        else:
                            buy_cond = [False]

                        if not self.last_buy is None:
                            if Position.open and Position.key == self.key:
                                sell_cond = [
                                    self.tick.close
                                    > self.trades.supertrend[-1]
                                    == self.trades.supertrend[-2],
                                    self.tick.close <= self.stoploss,
                                ]

                        else:
                            sell_cond = [False, False]
                        async with Trader.lock:
                            if not Position.open and not self.tag == "underlying":
                                if (
                                    all(buy_cond)
                                    # and self.tick.close * self.units <= 300000
                                ):
                                    adx_str = f"{adx[0][-1],adx[1][-1],adx[2][-1]}"
                                    self.buy_order(adx_str)
                                else:
                                    if self.newtick:
                                        self.trades.buys.append(np.nan)
                                        self.trades.sells.append(np.nan)
                            else:
                                if sell_cond[1] and Position.key == self.key:
                                    self.sell_order(hit_stoploss=True)
                                elif sell_cond[0] and Position.key == self.key:
                                    self.sell_order()
                                else:
                                    if self.newtick:
                                        self.trades.buys.append(np.nan)
                                        self.trades.sells.append(np.nan)
                self.gui.data_queue.put(True)
                count += 1
            except asyncio.CancelledError as e:
                config.stopevent.set()
                return
            except Exception as e:
                logger.exception(f"{e}\nStopevent set.", stack_info=True, exc_info=True)
                config.stopevent.set()
                return

            finally:
                if config.stopevent.is_set():
                    for task in self.gui.async_tasks:
                        if not task.done():
                            task.cancel()
                            logger.info("async tasks cancelled")


async def sim_live(
    buffer,
    keys,
    current_index,
    total_indices,
):
    if config.stopevent.is_set():
        logger.info("Returning because stop event is set.")
        return
    tqdm_lock = tqdm.get_lock()
    if tqdm_lock is not None:
        tqdm.set_lock(tqdm_lock)
    try:
        await config.looprun.wait()
        while current_index < total_indices or current_index == total_indices == 0:
            if config.stopevent.is_set():
                logger.info("Returning because stop event is set.")
                return
            outputdata = {}
            for opt, key in zip([call_data, put_data, scrip_data], keys):
                if isinstance(opt, list):
                    outputdata[key] = opt[current_index]
                elif isinstance(opt, pd.DataFrame) or isinstance(opt, pd.Series):
                    outputdata[key] = opt.iloc[current_index]
                else:
                    raise TypeError(
                        "The data for simulation is invalid. Expected a list or a Pandas DataFrame of OHLC data."
                    )

            await buffer.put(outputdata)
            current_index += 1
            config.current_index = current_index
            config.tick_ready.set()
            await asyncio.sleep(config.latency)
        else:
            print("#------Simulation Complete------#")
            await asyncio.sleep(2)  # Allow the final ticks to be processed
            config.stopevent.set()  # Signal all other loops to stop
            return
    except asyncio.CancelledError:
        logger.info("Coroutines have been stopped!")
    except IndexError as e:
        logger.error(f"IndexError {e} at index {current_index}", exc_info=True)
        return
    except Exception as e:
        logger.exception(f"Exception {e} at index {current_index}", exc_info=True)
        logger.info("Exiting simulation.")
        exit()


async def publisher(buffer: asyncio.Queue):
    try:
        while True:
            if config.stopevent.is_set():
                print("stopevent set for publisher")
                break
            try:
                data = await asyncio.wait_for(buffer.get(), timeout=6)
            except TimeoutError as e:
                logger.error("Timed out while getting publishable data from buffer.")
                if config.stopevent.is_set():
                    logger.info(
                        "Simulation finished naturally. Publisher shutting down."
                    )
                    return
                else:
                    logger.error(
                        f"Unexpected Timeout! | Index : {config.current_index} | Buffer state: {buffer}"
                    )
                    continue
            for subs in Trader._instances:
                if subs.key in data.keys():
                    await asyncio.wait_for(
                        subs.tickqueue.put(data[subs.key]), timeout=5
                    )
                else:
                    pass
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.exception(e, stack_info=True, exc_info=True)


def file_handler(filepath):
    filetype = magic.from_file(filepath, mime=True)
    if filetype == "text/plain":
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                with open(filepath, "r") as f:
                    data = f.read(2048)
                    csv.Sniffer().sniff(data)
                    return pd.read_csv(filepath)
            except (csv.Error, UnicodeEncodeError):
                raise Exception(f"Error while reading file: {filepath}.")
    elif (
        filetype == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        data = pd.read_excel(filepath)
        return data
    elif filetype == "application/json":
        return pd.read_json(filepath, convert_dates=False)
    else:
        raise TypeError("Invalid file type. Only JSON, CSV or Excel(.xlsx) accepted.")


def setup_data(simulation=False, dateidx=None):
    global call_data, put_data, scrip_data
    live_index = 0
    livetotal = 0
    try:
        if config.market_close.is_set() or simulation:
            dateidx = 144 if dateidx is None else dateidx
            scrip, suitable_calls, suitable_puts, keys = (
                "",
                [],
                [],
                [None, None, "NSE_INDEX|Nifty 50"],
            )
            testdata = os.listdir(ustox.directory + "sim_database/")
            print("Simulating date: ", testdata[dateidx])
            simdate = datetime.strptime(testdata[dateidx], "%Y-%m-%d").date()
            sim_datadir = ustox.directory + "sim_database/" + testdata[dateidx] + "/"
            files = os.listdir(sim_datadir)
            keynames = [Path(file).stem for file in files]

            local_database = pd.read_parquet(ustox.DATA_DIR / "NSE_DATABASE.parquet")

            call_filter = (local_database["segment"] == "NSE_FO") & (
                local_database["instrument_type"] == "CE"
            )
            put_filter = (local_database["segment"] == "NSE_FO") & (
                local_database["instrument_type"] == "PE"
            )

            call_instrument_keys = set(
                local_database.loc[call_filter, "instrument_key"]
            )

            put_instrument_keys = set(local_database.loc[put_filter, "instrument_key"])

            for filename, keyname in zip(files, keynames):
                if keyname in call_instrument_keys:
                    suitable_calls.append(filename)
                elif keyname in put_instrument_keys:
                    suitable_puts.append(filename)
                elif keyname == "NSE_INDEX":
                    scrip = filename
            if len(suitable_puts) == 0 or len(suitable_calls) == 0:
                option_files = [file for file in files if not "INDEX" in file]
                if len(option_files) >= 0 and scrip:
                    # 1. Load the data temporarily to analyze price action
                    idx_data = file_handler(sim_datadir + scrip)
                    opt0_data = file_handler(sim_datadir + option_files[0])
                    opt1_data = file_handler(sim_datadir + option_files[1])

                    # 2. Truncate to the shortest length to ensure perfect alignment
                    min_len = min(len(idx_data), len(opt0_data), len(opt1_data))

                    if min_len > 0:
                        # 3. Calculate Pearson correlation between option close & index close
                        corr0 = (
                            opt0_data["close"]
                            .iloc[:min_len]
                            .corr(idx_data["close"].iloc[:min_len])
                        )
                        corr1 = (
                            opt1_data["close"]
                            .iloc[:min_len]
                            .corr(idx_data["close"].iloc[:min_len])
                        )

                        # 4. The one with the higher correlation is the Call (CE)
                        if corr0 > corr1:
                            suitable_calls.append(option_files[0])
                            suitable_puts.append(option_files[1])
                        else:
                            suitable_calls.append(option_files[1])
                            suitable_puts.append(option_files[0])
                    else:
                        raise ValueError(
                            f"Empty files found on {testdata[dateidx]}. Cannot determine CE/PE."
                        )
                else:
                    raise ValueError(
                        f"Missing required option/index files on {testdata[dateidx]} to perform fallback."
                    )
                scrip = next((file for file in files if "INDEX" in file), None)

            for idx in suitable_calls:
                if os.path.exists(sim_datadir + idx):
                    call_data = file_handler(sim_datadir + idx)
                    call_data["date"] = pd.to_datetime(call_data["timestamp"]).dt.date
                    call_grouped = call_data.groupby("date")
                    if call_grouped.ngroups != 1:
                        logger.warning(
                            "Data for call option failed integrity check.",
                            extra={
                                "dates": call_data["date"],
                                "groups": call_grouped.ngroups,
                            },
                        )
                    call_data.drop(columns=["date"])

                    idx = Path(idx).stem
                    keys[0] = idx
                    call_plotdata = Plotdata(idx)
                else:
                    print(f"Call path {sim_datadir + idx} does not exist.")
            for idx in suitable_puts:
                if os.path.exists(sim_datadir + idx):
                    put_data = file_handler(sim_datadir + idx)
                    put_data["date"] = pd.to_datetime(put_data["timestamp"]).dt.date
                    put_grouped = put_data.groupby("date")
                    if put_grouped.ngroups != 1:
                        logger.warning(
                            "Data for put option failed integrity check.",
                            extra={
                                "dates": put_data["date"],
                                "groups": put_grouped.ngroups,
                            },
                        )
                    put_data.drop(columns=["date"])
                    idx = Path(idx).stem
                    keys[1] = idx
                    put_plotdata = Plotdata(idx)
                else:
                    print(f"Put path {sim_datadir + idx} does not exist.")
            if os.path.exists(sim_datadir + scrip):
                scrip_data = file_handler(sim_datadir + scrip)
            else:
                print("Scrip data does not exist.")
                scrip_data = None

            scrip_plotdata = Plotdata(keys[2])
            livetotal = min(len(call_data), len(put_data))
            live_index = 0
            latency = 0.5

        else:
            simdate = datetime.today().date()
            latency = 0.1
            expiry_date = ustox.get_expiry(ustox.get_options())
            print("Trading options with expiry on: ", expiry_date)
            suitable_puts, suitable_calls = ustox.get_suitable(
                expiry_date,
                funds=Funds.balance,
            )
            if len(suitable_calls) == 0 or len(suitable_puts) == 0:
                gap = 2
                while len(suitable_calls) == 0 or len(suitable_puts) == 0:
                    try:
                        print(
                            f"No suitable options found, trying again in {gap} seconds."
                        )
                        suitable_puts, suitable_calls = ustox.get_suitable(
                            ustox.get_expiry(ustox.get_options()), funds=Funds.balance
                        )
                    except KeyboardInterrupt:
                        exit()
                    time.sleep(gap)
                    gap += 2

            keys = [
                suitable_calls[0]["instrument_key"],
                suitable_puts[0]["instrument_key"],
                "NSE_INDEX|Nifty 50",
            ]
            call_plotdata = Plotdata(keys[0])
            put_plotdata = Plotdata(keys[1])
            scrip_plotdata = Plotdata(keys[2])
            livetotal = 0
            live_index = 0

        return (
            keys,
            simdate,
            call_plotdata,
            put_plotdata,
            scrip_plotdata,
            live_index,
            livetotal,
            latency,
        )

    except Exception as e:
        logger.exception(e, stack_info=True, exc_info=True)
    except FileNotFoundError as e:
        logger.error(e, stack_info=True, exc_info=True)
