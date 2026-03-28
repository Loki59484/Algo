# ------Import Libraries------
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Literal
from datetime import datetime
import pyarrow.parquet as pq
from pathlib import Path
import pandas_ta as ta
from tqdm import tqdm
import pandas as pd
import logging
import asyncio
import random
import sys


"""
Structural class containing classes and function definitions for backtesting and live trading. 
Function defined here are used to load and backtest strategies.
The Position and Funds classes are used to store the state of the open position and the available funds, respectively. 
The Tick class is used to store the current tick data for each scrip. 
The Plotdata class is used to store the data for plotting the charts in the GUI.
"""

# Setting up paths to manage imports
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core import upstox_func as ustox
from config import config
from core.datatypes import *


global call_data, put_data, scrip_data
ticks_ready = asyncio.Event()
pool = ThreadPoolExecutor()
logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: Path, **kwargs):
    """
    Saves a DataFrame to a Parquet file with metadata. The metadata is passed as keyword arguments and stored in the Parquet file's schema metadata.
    """
    try:
        metadata_bytes = {
            key.encode("utf-8"): str(value).encode("utf-8")
            for key, value in kwargs.items()
        }
        table = pa.Table.from_pandas(df)
        existing_metadata = table.schema.metadata or {}
        final_metadata = {**existing_metadata, **metadata_bytes}
        table = table.replace_schema_metadata(final_metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
    except Exception as e:
        logger.exception(f"Error saving parquet file at {path}: {e}")


def load_parquet(path: Path):
    """
    Loads a DataFrame and metadata from a Parquet file. The metadata is returned as a dictionary with string keys and values.
    If there's an error during loading, it logs the exception and returns None for both data and metadata.
    """
    try:
        table = pq.read_table(path)
        df = table.to_pandas()
        metadata = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in (table.schema.metadata or {}).items()
        }
        return dict(data=df, metadata=metadata)
    except Exception as e:
        logger.exception(f"Error loading parquet file from {path}: {e}")
        return dict(data=None, metadata=None)


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
        buy_constraints: Callable[[Any], bool], optional
            Parameter to specify any other constraints to excercise while placing a buy order.
        sell_constraints: Callable[[Any], bool], optional
            Parameter to specify any other constraints to excercise while placing a sell order.
        self.custom_test 
            A custom test procedure provided as a functools.partial. This is called inside the `Trader.test()`
            instead of the internal test procedure.
        
    Yields
    --------
    Strategy
        An object of `Strategy` class.
    """

    def __init__(self):
        self.name: str | None = None
        self.buy_conditon: Callable[[pd.DataFrame, Any], pd.Series] | None = None
        self.sell_condition: Callable[[pd.DataFrame, Any], pd.Series] | None = None
        self.indicators: ta.Study | None = None
        self.buy_constraints: Callable[[Any], bool] | None = None
        self.sell_constraints: Callable[[Any], bool] | None = None
        self.custom_test: partial | None = None

    def add_indicators(self, indicators: dict | list[dict]):
        """
        Adds new indicators as a dict to the pandas_ta Study used for indicators.
        It accepts a dict for a new indicator or a list of dicts for multiple indicators.
        """
        if self.indicators is None:
            self.indicators = ta.Study(
                name="Stategy_indicators",
                ta=([indicators] if isinstance(indicators, dict) else indicators),
            )
            return
        if isinstance(indicators, dict):
            self.indicators.ta.append(indicators)
            return
        if isinstance(indicators, list):
            self.indicators.ta.extend(indicators)
            return

    @classmethod
    def apply_study(cls,target : pd.DataFrame,study : ta.Study,**kwargs):
        """
        Applies the given study to the provided target.

        Parameters
        ---------
        target : pd.DataFrame
            The target dataframe containing candle data.
            The assumptions made by pandas_ta for columns being named `open`, `high`, `low`, `close` and `volume` expected.
        kwargs : Any
            The kwargs provided are passed directly to the callables for buy and sell condition.

        Yields
        ---------
        None
        """
        target.ta.study(study)
        buy_cond = kwargs.get("buy_condition",None)
        sell_cond = kwargs.get("sell_condition",None)

        if buy_cond is not None:
            target["buy_signal"] = buy_cond(target, **kwargs)
        else:
            target["buy_signal"] = False

        if sell_cond:
            target["sell_signal"] = sell_cond(target, **kwargs)
        else:
            target["sell_signal"] = False
        
        return target


    def apply(self, target: pd.DataFrame, **kwargs):
        """
        Applies the strategy instance to the provided target.

        Parameters
        ---------
        target : pd.DataFrame
            The target dataframe containing candle data.
            The assumptions made by pandas_ta for columns being named `open`, `high`, `low`, `close` and `volume` expected.
        kwargs : Any
            The kwargs provided are passed directly to the callables for buy and sell condition.

        Yields
        ---------
        None
        """

        if self.indicators and self.indicators.ta:
            target.ta.study(self.indicators)
        
        if self.buy_conditon:
            target["buy_signal"] = self.buy_conditon(target, **kwargs)
        else:
            target["buy_signal"] = False

        if self.sell_condition:
            target["sell_signal"] = self.sell_condition(target, **kwargs)
        else:
            target["sell_signal"] = False
        
        return target


@dataclass(slots=True)
class Profile:
    pass


@dataclass
class Bucket:
    """A Bucket to collect instruments for being traded together."""

    date: datetime
    spot: Instrument | None = None
    legs: dict[str, Instrument] = field(default_factory=dict)
    margin: float = 0
    open_position : Position | None = None

    def add_leg(self, item: Instrument | dict[str, Instrument], leg_type):
        """Adds Instrument instances as legs to a bucket object.

        Args:
            item (Instrument | dict[str, Instrument]): A single Instrument instance or a dict of Instrument instances.
            If passing an Instrument instance, an option of `leg_type` can also be passed. `leg_type` defaults to `Instrument.type`.
            If passing as a dict, ensure key:value pair is of `{"leg_type":Instrument}` form.

        """
        if isinstance(item, dict):
            self.legs = {**self.legs, **item}
        elif isinstance(item, Instrument):
            key = leg_type if leg_type is not None else item.type
            self.legs[key] = item


class Trader:
    """
    The trading engine built for backtesting multiple instruments all at once
    using pandas powerful technical analysis tools.
    """

    def __init__(self):
        self.user: Profile | None = None
        self.strategy: Strategy = None
        self.instruments: dict[str, Instrument] = {}
        self.buckets: list[Bucket] = []
        self._trade_report: pd.DataFrame | None = None

    @property
    def trade_report(self, verbose: bool = True):
        if self._trade_report is None:
            df = pd.concat(
                [
                    subject.position.trade_report()
                    for subject in self.instruments.values()
                ],
                ignore_index=True,
            )
            self._trade_report = df.sort_values(by="Timestamp").reset_index(drop=True)
        if verbose:
            print(self._trade_report.round(2).to_markdown(tablefmt="pretty"))
            print(f"Total Movement: {self._trade_report['Movement'].sum():.2f}")
            print(f"Final PnL: {self._trade_report['PnL'].sum():.2f}")
            logger.info(f"Final Results:\n{self._trade_report.round(2).to_markdown(tablefmt="pretty")}")
            logger.info(f"Total Movement: {self._trade_report['Movement'].sum():.2f}")
            logger.info(f"Final PnL: {self._trade_report['PnL'].sum():.2f}")

        
        return self._trade_report

    def calculate_units(self, balance, close, lot_size):
        return max(
            0, int((balance / close) - ((balance / close) % lot_size))
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

    def execute_buy(self, tick: Candle, instrument: Instrument, margin: float):
        order_id = str(random.randint(10**11, 10**12 - 1))
        qty = self.calculate_units(balance=margin,close=tick.close,lot_size=instrument.lot_size)
        trade = Trade.from_candle(qty=qty, id=order_id, candle=tick)
        instrument.position.open_position(trade=trade)
        logger.info(f"Buy order placed for {qty} at {tick.close}")

    def execute_sell(
        self, tick: Tick, trade: Trade, instrument: Instrument, qty=None,**kwargs
    ):
        qty = trade.buy_qty if qty is None else qty
        trade.close_trade(
            tick.close, qty=trade.buy_qty, remark=kwargs.get("remark", "")
        )
        instrument.position.close_position()
        logger.info(f"Sell order placed for {trade.buy_price} at {tick.close}")

    def test(
        self,
        subjects: dict[str, Instrument] | None = None,
        strategy: Strategy | None = None,
        **kwargs,
    ):
        """
        Executes the hybrid vectorized/event-driven backtest across the portfolio.

        Calculates all technical indicators and signals in bulk via pandas-ta,
        then simulates chronological execution to enforce capital constraints.

        Parameters
        ----------
        subjects : dict[str, Instrument], optional
            A dictionary mapping instrument keys to their respective Instrument
            objects. If None, defaults to `self.instruments`.
        strategy : Strategy, optional
            The instantiated strategy blueprint containing the pandas-ta indicators
            `buy_condition` and `sell_condition` for execution. Defaults to `self.strategy`.
        kwargs: Any
            Kwargs are directly passed to the `apply()` and `test_instrument()` functions.

        """

        def test_instrument(subject: Instrument, **kwargs):

            strategy.apply(subject.to_dataframe(), **kwargs)

            # EXTRACT TRADING DAY DATA, DROPPING WARM UP CANDLES
            trading_data = subject.historical_df.loc[subject.date :]
            trading_data = trading_data.reset_index()
            for row in trading_data.itertuples():
                if row.buy_signal and (
                    strategy.buy_constraints(instrument=subject, **kwargs)
                    if strategy.buy_constraints is not None
                    else True
                ):
                    self.execute_buy(tick=row, instrument=subject,margin=30000)

                elif row.sell_signal and (
                    strategy.sell_constraints(instrument=subject, **kwargs)
                    if strategy.sell_constraints is not None
                    else True
                ):
                    self.execute_sell(
                        tick=row, trade=subject.position.open_trade, instrument=subject
                    )

        strategy = strategy or self.strategy

        if strategy is None:
            logger.warning("No strategy applied as none was provided.")
            return
        subjects = subjects or self.instruments

        if not subjects:
            logger.warning("No strategy applied as target(s) were not provided.")
            return
            
        logger.info("Starting simulations")
        if strategy.custom_test is not None:
            strategy.custom_test()
        else:
            for subject in tqdm(subjects.values(), desc="Backtesting"):
                test_instrument(subject, **kwargs)

        # PRINTING FINAL TRADE REPORT FOR THE SIMULATION
        self.trade_report

