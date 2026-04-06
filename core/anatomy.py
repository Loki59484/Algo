"""
Structural class containing classes and function definitions for backtesting and live trading.
Function defined here are used to load and backtest strategies.
The Position and Funds classes are used to store the state of the open position and the available funds, respectively.
The Tick class is used to store the current tick data for each scrip.
The Plotdata class is used to store the data for plotting the charts in the GUI.
"""

# ------Import Libraries------
from abc import ABC, abstractmethod
from typing import Any, Callable
from functools import partial
from pathlib import Path
import pandas_ta as ta
import pandas as pd
import asyncio
import logging
import sys


# Setting up paths to manage imports
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core.datatypes import *
from core.methods import *
from core.upstox_methods import UpstoxClient


logger = logging.getLogger(__name__)
ustox = UpstoxClient()


class Broker(ABC):
    """
    Abstract Base class for defining methods of an executor that manages order placement and modification.
    """

    @abstractmethod
    def buy_order(key: str, price: float, qty: int, **kwargs):
        pass

    @abstractmethod
    def sell_order(key: str, price: float, qty: int, **kwargs):
        pass

    @abstractmethod
    def cancel_order(id: str):
        pass

    @abstractmethod
    def modify_order(id: str, **kwargs):
        pass


class Streamer(ABC):
    """
    Abstract Base class for defining methods of a streamer that provides a stream of ticks
    """

    @abstractmethod
    async def start():
        pass


class LivefeedStreamer(Streamer):

    def __init__(
        self,
        client: UpstoxClient,
        instruments: dict[str, Instrument],
        buffer: asyncio.Queue,
    ):
        super().__init__()
        self.instruments = instruments
        self.keys: list = [i.key for i in instruments.values()]
        self.buffer = buffer
        self.client: UpstoxClient = client

    async def start(self):
        await self.client.subscribe_ticks(instrument_key=self.keys, buffer=self.buffer)


class SimfeedStreamer(Streamer):
    def __init__(self, instruments: dict[str, Instrument], buffer: asyncio.Queue):
        super().__init__()
        self.instruments: dict[str, Instrument] = instruments
        self.keys: list = [i.key for i in instruments.values()]
        self.buffer = buffer

    async def start(self):
        await self.simulator(buffer=self.buffer)

    async def simulator(self, buffer: asyncio.Queue):
        pass


class Strategy:
    """
    Base class for defining strategies.

    Parameters
    --------
        name: str | None , optional
            A custom name for the strategy, Defaults to None
        buy_condition : Callable[[pd.DataFrame], pd.Series], optional
            A callable to check condition(s) to set a buy signal that
            accepts Dataframe or Series and returns a boolean series.
            Should also accepts kwargs for specifying conditions requiring
            parameters other than the DataFrame.
        sell_condition : Callable[[pd.DataFrame], pd.Series], optional
            A callable to check condition(s) to set a sell signal that
            accepts Dataframe or Series and returns a boolean series.
            Should also accepts kwargs for specifying conditions requiring
            parameters other than the DataFrame.
        indicators: ta.Study | None, optional
            A pandas Study class object to add indicators to the dataset.
            A Study can directly be assigned to the indicators or the `add_indicators()`
            function can be to assign indicators a built-in study.

    Returns
    --------
    Strategy
        An object of `Strategy` class.
    """

    def __init__(self, name=None, buy_cond=None, sell_cond=None):
        self.name: str | None = name
        self.buy_conditon: Callable[[pd.DataFrame, Any], pd.Series] | None = buy_cond
        self.sell_condition: Callable[[pd.DataFrame, Any], pd.Series] | None = sell_cond
        self.indicators: ta.Study | None = None

    def add_indicators(self, indicators: dict | list[dict]):
        """
        Adds new indicators as a dict to the pandas_ta Study used for indicators.
        It accepts a dict for a new indicator or a list of dicts for multiple indicators.
        """
        if self.indicators is None:
            self.indicators = ta.Study(
                name="Stategy_indicators",
                ta=([indicators] if isinstance(indicators, dict) else indicators),
                cores=0,
            )
            return
        if isinstance(indicators, dict):
            self.indicators.ta.append(indicators)
            return
        if isinstance(indicators, list):
            self.indicators.ta.extend(indicators)
            return

    @classmethod
    def apply_study(
        cls, target: pd.DataFrame | deque[Candle], study: ta.Study, **kwargs
    ):
        """
        Applies the given study to the provided target.

        Parameters
        ---------
        target : pd.DataFrame | deque[Candles]
            The target dataframe containing candle data or a deque containing Candle objects which are converted to a DataFrame internally.
            The assumptions made by pandas_ta for columns being named `open`, `high`, `low`, `close` and `volume` expected.
        kwargs : Any
            The kwargs provided are passed directly to the callables for buy and sell condition.

        Returns
        ---------
        pd.DataFrame
        """
        if isinstance(target, deque):
            if not target:
                return None
            target = pd.DataFrame([{**asdict(candle), **kwargs} for candle in target])

        if target.empty:
            return None

        target.ta.study(study)
        buy_cond = kwargs.get("buy_condition", None)
        sell_cond = kwargs.get("sell_condition", None)

        target["buy_signal"] = buy_cond(target, **kwargs) if buy_cond else False
        target["sell_signal"] = sell_cond(target, **kwargs) if sell_cond else False

        return target

    def apply(self, target: pd.DataFrame | deque[Candle], **kwargs):
        """
        Applies the self strategy instance to the provided target.

        Parameters
        ---------
        target : pd.DataFrame
            The target dataframe containing candle data.
            The assumptions made by pandas_ta for columns being named `open`, `high`, `low`, `close` and `volume` expected.
        kwargs : Any
            The kwargs provided are passed directly to the callables for buy and sell condition.

        Return
        ---------
        pd.DataFrame
        """
        if isinstance(target, deque):
            if not target:
                return None
            target = pd.DataFrame([{**asdict(candle), **kwargs} for candle in target])

        if target.empty or not (self.indicators and self.indicators.ta):
            return None

        target.ta.study(self.indicators)
        target["buy_signal"] = (
            self.buy_conditon(target, **kwargs) if self.buy_conditon else False
        )
        target["sell_signal"] = (
            self.sell_condition(target, **kwargs) if self.sell_condition else False
        )

        return target


class SimBroker(Broker):

    def __init__(self):
        super().__init__()
        self.order_history: dict[str, Order] = {}

    def buy_order(instrument, price, qty):
        """
        Function to acknowledge buy requests while simulating

        Returns
        -------
        Order
        """
        return super().buy_order(qty)

    def sell_order(instrument, price, qty):
        """
        Function to acknowledge sell requests while simulating

        Returns
        -------
        Order
        """
        return super().sell_order(qty)

    def cancel_order(id):
        return super().cancel_order()

    def modify_order():
        pass


class LiveBroker(Broker):

    def __init__(self, client):
        super().__init__()
        self.order_history: dict[str, Order] = {}
        self.client = client

    def buy_order(
        self, key: str, price: float, qty: int, sandbox: bool = False, **kwargs
    ):
        return self.client.place_order(
            instrument_token=key,
            transaction_type="BUY",
            quantity=qty,
            price=price,
            sandbox=sandbox,
            **kwargs,
        )

    def sell_order(
        self, key: str, price: float, qty: int, sandbox: bool = False, **kwargs
    ):
        return self.client.place_order(
            instrument_token=key,
            transaction_type="SELL",
            quantity=qty,
            price=price,
            sandbox=sandbox,
            **kwargs,
        )

    def cancel_order(self, id: str, sandbox: bool = False):
        return self.client.cancel_order(id, sandbox=sandbox)

    def modify_order(self, id: str, sandbox: bool = False, **kwargs):
        return self.client.modify_order(id=id, sandbox=sandbox, **kwargs)

    def update_order_history(self, source: str | dict | Order):
        if isinstance(source, id):
            self.order_history.append(self.client.get_order_details(order_id=id))
        elif isinstance(source, dict):
            self.order_history.append(Order.parse(source))
        elif isinstance(source, Order):
            self.order_history.append(Order)


class Trader:
    """
    The trading engine built for backtesting multiple instruments all at once
    using pandas powerful technical analysis tools.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        strategy: Strategy,
        broker: LiveBroker | SimBroker,
        datafeed: asyncio.Queue,
    ):
        """_summary_

        Args:
            portfolio (Portfolio): A `Portfolio` object to store positions, orders and other portfolio realted details.
            strategy (Strategy): A strategy object to store indicators and produce buy-sell signals.
            broker (LiveBroker | SimBroker): Broker object for handling order placement and their modification.
            datafeed (asyncio.Queue): Queue for holding live ticks
            executor (Callable[[Any],None]): A custom function to execute trades in a complex strategic manner. Defaults to a built-in `_executor` function.
        """
        self.instruments: dict[str, Instrument] = {}
        self.portfolio: Portfolio = portfolio
        self.buckets: list[Bucket] = []
        self.strategy: Strategy = strategy
        self.broker: SimBroker | LiveBroker = broker
        self.datafeed: asyncio.Queue = datafeed
        self.executor: Callable[[Any], None] = self._executor

    
    def add_executor(self,executor: Callable[[Any], None]):
        self.executor = executor
        

    def _executor(self, data: pd.Series | pd.DataFrame,sandbox=True,**kwargs):
        """
        Built-in executor function to execute a basic strategy of buying on buy signals and seeling on sell signals.
        Sandbox is enabled while placing order by default.
        Orders are placed for a single lot size of the subject instrument. 

        Args:
            data (pd.Series | pd.DataFrame):

        Returns:
        None
        """

        if data is None or data.empty:
            return
        last_tick = data.iloc[-1]
        if last_tick.get(["buy_signal"]) == True:
            self.broker.buy_order(
                key=last_tick.get("key"),
                price=last_tick.get("close"),
                qty=self.instruments[last_tick.get("key")].lot_size,
                sandbox=sandbox,
                **kwargs
            )
        elif last_tick.get(["sell_signal"]) == True:
            self.broker.sell_order(
                key=last_tick.get("key"),
                price=last_tick.get("close"),
                qty=self.instruments[last_tick.get("key")].lot_size,
                sandbox=sandbox,
                **kwargs
            )
        else:
            return {}

    def calculate_units(self, balance, close, lot_size):
        return min(
            max(0, int((balance / close) - ((balance / close) % lot_size))),
            (32000 - (32000 % lot_size)),
        )

    def add_instrument(self, item: Instrument | dict[str, Instrument]):
        if isinstance(item, dict):
            self.instruments = {**self.instruments, **item}
            logger.debug(f"{len(item)} instruments added to Trader instance.")
            return
        if isinstance(item, Instrument):
            self.instruments[item.key] = item
            logger.debug(f"Instrument {item.key} added to Trader instance.")
            return

    def add_bucket(self, buckets: Bucket | list):
        if isinstance(buckets, Bucket):
            self.buckets.append(buckets)
        elif isinstance(buckets, list):
            self.buckets.extend(buckets)

    async def tick_processor(self, **kwargs):
        while True:
            
            ticks: dict[str, Tick] = await self.datafeed.get()

            for key, tick in ticks.items():
                self.instruments[key].historical_candles.append(tick)

            tasks = [
                asyncio.to_thread(self.strategy.apply, val.historical_candles, key=key)
                for key, val in self.instruments.items()
            ]
            
            results = await asyncio.gather(*tasks)
            