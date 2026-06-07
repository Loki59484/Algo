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
from random import randint
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
from core.upstox_methods import UpstoxClient, LOG_DIR


logger = logging.getLogger(__name__)


class Broker(ABC):
    """
    Abstract Base class for defining methods of an executor that manages order placement and modification.
    """

    def __init__(self):
        """
        Initializes the Broker with an empty order history.
        """
        self.order_history: dict[str, Order] = {}

    
    @abstractmethod
    def buy_order(self, key: str, price: float, qty: int, **kwargs):
        pass

    @abstractmethod
    def sell_order(self, key: str, price: float, qty: int, **kwargs):
        pass

    @abstractmethod
    def cancel_order(self, id: str):
        pass

    @abstractmethod
    def modify_order(self, id: str, **kwargs):
        pass


class Streamer(ABC):
    """
    Abstract Base class for defining methods of a streamer that provides a stream of ticks
    """

    @abstractmethod
    async def start():
        pass


class BaseEngine(ABC):
    """
    Abstract Base class for defining structure of Trading Engine

    """

    def __init__(self, portfolio, strategy, broker):
        self.instruments: dict[str, Instrument] = {}
        self.portfolio: Portfolio = portfolio
        self.buckets: list[Bucket] = []
        self.strategy: Strategy = strategy
        self.broker: SimBroker | LiveBroker = broker

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

    @abstractmethod
    def run(self):
        pass


class LivefeedStreamer(Streamer):
    """
    Streamer to subscribe and stream tick data for specified instruments and pass them to output buffer queue.
    
    Args:
        client (UpstoxClient): Agent to subscribe the tick data.
        instruments (dict[str, Instrument]): Instruments to be subscribed
        buffer (asyncio.Queue): Output queue to push data into.
    
    """
    def __init__(
        self,
        client: UpstoxClient,
        instruments: dict[str, Instrument],
        buffer: asyncio.Queue,
    ):
        """
        Args:
            client (UpstoxClient)
            instruments (dict[str, Instrument])
            buffer (asyncio.Queue)
        """
        super().__init__()
        self.instruments = instruments
        self.keys: list = [i.key for i in instruments.values()]
        self.buffer = buffer
        self.client: UpstoxClient = client

    async def start(self):
        logger.info("Logger started, subscribing to ticks.")
        await self.client.subscribe_ticks(instrument_key=self.keys, buffer=self.buffer)


class SimfeedStreamer(Streamer):
    """
    Streamer to subscribe and stream tick data for specified instruments and pass them to output buffer queue.
    
    Args:
        instruments (dict[str, Instrument]): Instruments for which tick data is to be streamed. 
        buffer (asyncio.Queue): Output queue to push tick data into.
        stopevent (asyncio.Event): Event to stop streaming tick data.
    """
    def __init__(self, instruments: dict[str, Instrument], buffer: asyncio.Queue,stopevent:asyncio.Event):
        """
        Args:
            instruments (dict[str, Instrument])
            buffer (asyncio.Queue) 
            stopevent (asyncio.Event)
        """
        super().__init__()
        self.instruments: dict[str, Instrument] = instruments
        self.keys: list = [i.key for i in instruments.values()]
        self.buffer = buffer
        self.stopevent = stopevent

    def add_instrument(self, item: Instrument | dict[str, Instrument]):
        if isinstance(item, dict):
            self.instruments = {**self.instruments, **item}
            logger.debug(f"{len(item)} instruments added to Trader instance.")
            return
        if isinstance(item, Instrument):
            self.instruments[item.key] = item
            logger.debug(f"Instrument {item.key} added to Trader instance.")
            return

    async def start(self):
        await self.simulator(buffer=self.buffer)

    async def simulator(self, buffer: asyncio.Queue):
        from copy import deepcopy
        i = 0
        items_to_simulate = deepcopy(self.instruments)

        sim_data = {key:[sim_candle for sim_candle in items.historical_candles if sim_candle.timestamp.date()==items.date] for key,items in items_to_simulate.items()}
        #total_idx = max([len(data.historical_candles) for data in items_to_simulate.values()])
        while True:
            try: 
                if not self.stopevent.is_set():                    
                    for key,data in self.instruments.items():
                        if i==len(data.historical_candles):
                            items_to_simulate.pop(key)
                    if not items_to_simulate:
                        logger.info("Simulation completed.")
                        self.stopevent.set()
                        return
                    tick_data = {key:data[i] for key,data in sim_data.items()}
                    await buffer.put(tick_data)
                    i+=1
                    await asyncio.sleep(0.01)
                else:
                    logger.info("Simulation Stopped")
                    break
            except Exception as e:
                logger.exception(f"Exception while simulating.")



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

        target["buy_signal"] = buy_cond(target, **kwargs) if buy_cond is not None else None
        target["sell_signal"] = sell_cond(target, **kwargs) if sell_cond is not None else None

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
            target.set_index('timestamp',inplace=True,drop=True)

        if target.empty or not (self.indicators and self.indicators.ta):
            return None

        target.ta.study(self.indicators)
        target["buy_signal"] = (
            self.buy_conditon(target, **kwargs) if self.buy_conditon is not None else None
        )
        target["sell_signal"] = (
            self.sell_condition(target, **kwargs) if self.sell_condition is not None else None
        )
        return target


class SimBroker(Broker):

    def __init__(self, portfolio):
        super().__init__()
        self.portfolio: Portfolio = portfolio

    def buy_order(self, key: str, price: float, qty: int, **kwargs):
        """
        Function to acknowledge buy requests while simulating

        Returns
        -------
        Order
        """

        cost = price * qty
        status = self.portfolio.funds.debit(cost)
        if status == -1:
            logger.warning("Order failed due to insufficient funds.")
            return None
        order_id = str(randint(1000000, 9999999))

        if not key in self.portfolio.positions.keys():
            self.portfolio.positions[key] = Position(
                instrument_token=key, buy_price=price, day_buy_quantity=qty
            )

        else:
            qty += self.portfolio.positions[key].day_buy_quantity
            self.portfolio.positions[key].update_position(
                buy_price=price, day_buy_quantity=qty
            )
        logger.info(f"{key} | Buy order placed succesfully for {qty} at {price}.")
        return self.portfolio.positions[key], order_id

    def sell_order(self, key: str, price: float, qty: int, **kwargs):
        """
        Function to acknowledge sell requests while simulating

        Returns
        -------
        Order
        """
        amount = price * qty
        self.portfolio.funds.credit(amount)
        if not key in self.portfolio.positions.keys():
            logger.warning(f"Sell order not placed as no positions are open for {key}.")
            return -1
        qty += self.portfolio.positions[key].day_sell_quantity
        self.portfolio.positions[key].update_position(
            sell_price=price, day_sell_quantity=qty
        )
        logger.info(f"{key} | Sell order placed succesfully for {qty} at {price}.")

        return 1

    def cancel_order(self, id):
        return super().cancel_order()

    def modify_order(self):
        pass


class LiveBroker(Broker):

    def __init__(self, client):
        super().__init__()
        self.client : UpstoxClient = client

    def buy_order(
        self, key: str, price: float, qty: int, sandbox: bool = False, **kwargs
    ):
        return self.client.place_order(
            instrument_token=key,
            transaction_type="BUY",
            quantity=qty,
            price=price,
            sandbox=sandbox,
            order_type='LIMIT',
            validity='IOC'
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
            order_type="LIMIT",
            validity="IOC"
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


class Trader(BaseEngine):
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
        """
        Initates a trader instance for event based tick by tick trading or simulation.
        Args:
            portfolio (Portfolio): A `Portfolio` object to store positions, orders and other portfolio realted details.
            strategy (Strategy): A strategy object to store indicators and produce buy-sell signals.
            broker (LiveBroker | SimBroker): Broker object for handling order placement and their modification.
            datafeed (asyncio.Queue): Queue for holding live ticks

        """

        super().__init__(portfolio=portfolio, strategy=strategy, broker=broker)
        self.datafeed: asyncio.Queue = datafeed
        self.executor: Callable[[Any], None] = (
            self._executor
        )  # for executing buy-sell logic
        self.processor: Callable[[Any], None] = (
            self._default_processor
        )  # for proccessing incoming ticks

    def set_executor(self, executor: Callable[[Any], None]):
        self.executor = executor

    def set_tick_processor(self, processor_func: Callable[[Any], None]):
        self.processor = processor_func

    def _executor(self, data: pd.Series | pd.DataFrame, sandbox=True, **kwargs):
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
                **kwargs,
            )
        elif last_tick.get(["sell_signal"]) == True:
            self.broker.sell_order(
                key=last_tick.get("key"),
                price=last_tick.get("close"),
                qty=self.instruments[last_tick.get("key")].lot_size,
                sandbox=sandbox,
                **kwargs,
            )
        else:
            return {}

    def calculate_units(self, close, lot_size):
        balance = self.portfolio.funds.available_margin
        return min(
            max(0, int((balance / close) - ((balance / close) % lot_size))),
            (32000 - (32000 % lot_size)),
        )


    async def _default_processor(self, **kwargs):
        while True:
            ticks: dict[str, Tick] = await self.datafeed.get()
            for key, tick in ticks.items():
                self.instruments[key].historical_candles.append(tick)
            tasks = [
                asyncio.to_thread(self.strategy.apply, val.historical_candles, key=key)
                for key, val in self.instruments.items()
            ]
            results = await asyncio.gather(*tasks)

            for result in results:
                self.executor(data=result)

    def run(self):
        """Kicks off the trading engine. Starts the tick_processor as an async task to start processing incoming ticks."""
        try:
            asyncio.run(self.processor())
            print("started")
        except KeyboardInterrupt:
            logger.info("Engine killed by user.")
        except Exception as e:
            logger.exception(f"Exception while processing live ticks.\n{e}")


class BulkSimulator(BaseEngine):

    def __init__(
        self,
        portfolio: Portfolio,
        strategy: Strategy,
        broker: LiveBroker | SimBroker,
    ):
        super().__init__(portfolio=portfolio, strategy=strategy, broker=broker)
        self.processor: Callable[[Any], None] = None

    def set_processor(self, processor_func: Callable[[Any], None]):
        self.processor = processor_func

    def calculate_units(self, close, lot_size):
        balance = self.portfolio.funds.available_margin
        logger.info(balance)
        return min(
            max(0, int((balance / close) - ((balance / close) % lot_size))),
            (32000 - (32000 % lot_size)),
        )

    def run(self):
        self.processor()
        report = self.portfolio.get_report()
        report_file = (
            LOG_DIR
            / "reports"
            / f"trade_report_{datetime.now().strftime("%d%m%Y_%H%M%S")}.csv"
        )
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_file)
