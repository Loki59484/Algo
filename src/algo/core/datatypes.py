from __future__ import annotations
from dataclasses import field, fields, asdict
from typing import Literal, Any
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import deque
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import numpy as np
import logging
import sys
import json

"""
This module contains custom dataclasses for smooth handling of trading data.
"""

# Set up logging
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from algo.core.methods import to_ist, parse_obj_to_dataclass


class DatatypeBase:
    """Base class for inheriting object parsing ability"""

    @classmethod
    def parse(cls, obj: dict):
        """
        Base method used to parse dicts recieved from Upstox into relevent dataclass
        """

        required_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in obj.items() if k in required_fields}
        return cls(**filtered_dict)


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Funds:
    starting_capital: float
    pnl: float = 0.0
    total: float = 0.0
    used_margin: float = 0.0
    adhoc_margin: float = 0.0
    available_margin: float = 0.0
    exposure_margin: float = 0.0
    notional_cash: float = 0.0
    payin_amount: float = 0.0
    span_margin: float = 0.0

    def __post_init__(self):
        logger.info(f"Starting with capital: {self.starting_capital}")
        self.available_margin = self.total = self.starting_capital

    def credit(self, amount: float):
        """Called by the broker when an asset is sold."""
        self.total += amount
        self.used_margin = 0.0  # Clean reset. Open position is closed.

        # UPSTOX T+1 RULE: Realized profits are locked until settlement.
        # We cap the 'available margin' to starting_capital, but subtract any active used_margin.
        realized_equity = self.total + self.used_margin
        self.available_margin = (
            min(realized_equity, self.starting_capital) - self.used_margin
        )

        self.pnl = self.total - self.starting_capital

    def debit(self, amount: float):
        """Called by the broker when an asset is bought."""
        if amount > self.available_margin:
            logger.error(
                f"Insufficient funds. Required: {amount}, Available: {self.available_margin}"
            )
            return -1

        self.total -= amount
        self.used_margin += amount

        # Deduct the bought amount from available margin safely while respecting the T+1 cap
        realized_equity = self.total + self.used_margin
        self.available_margin = (
            min(realized_equity, self.starting_capital) - self.used_margin
        )

        self.pnl = self.total - self.starting_capital

    def settle(self, simulate: bool = True, client=None):
        """Called at the end of the day to finalize the ledger."""
        if not simulate and client is None:
            logger.error(
                "A client instance of `UpstoxClient` is required if not simulating."
            )
            return

        logger.info(
            f"Day settled | Starting: {self.starting_capital} | End Total: {self.total} | PnL: {self.pnl}"
        )

        if simulate:
            # T+1 SETTLEMENT: Today's profits are officially released into tomorrow's starting capital!
            self.starting_capital = self.total
            self.available_margin = self.total
            self.used_margin = 0.0
            self.pnl = 0.0
            return

        self.update_from_json(client.get_funds())

    @classmethod
    def update_from_json(cls, data: dict):
        """Instance method to update from live Upstox data, avoiding @classmethod bugs."""
        eq = data.get("equity", {})

        available_margin = eq.get("available_margin", 0.0)
        used_margin = eq.get("used_margin", 0.0)
        return cls(
            starting_capital=eq.get("available_margin", 0.0),
            adhoc_margin=eq.get("adhoc_margin", 0.0),
            available_margin=eq.get("available_margin", 0.0),
            exposure_margin=eq.get("exposure_margin", 0.0),
            notional_cash=eq.get("notional_cash", 0.0),
            payin_amount=eq.get("payin_amount", 0.0),
            span_margin=eq.get("span_margin", 0.0),
            used_margin=eq.get("used_margin", 0.0),
            total=available_margin + used_margin,
        )


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Greeks(DatatypeBase):
    """
    Dataclass to store greeks for a tick.
    """

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class LTPC(DatatypeBase):
    """
    Dataclass to store LTPC data of a tick.
    """

    ltp: float = np.nan
    ltt: str = "0"
    ltq: str = "0"
    cp: float = np.nan


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Candle:
    """
    Dataclass to store OHLC, vol, oi data of a tick.
    """

    timestamp: datetime = to_ist(0)
    open: float = np.nan
    high: float = np.nan
    low: float = np.nan
    close: float = np.nan
    volume: int = 0
    ltp: float | None = None

    # CUSTOM FIELDS
    buy_signal: bool = False
    sell_signal: bool = False

    @classmethod
    def load_ohlc(cls, ohlc: dict, ltpc: LTPC):
        return cls(
            timestamp=to_ist(ohlc.get("ts", "0")),
            open=ohlc.get("open", np.nan),
            high=ohlc.get("high", np.nan),
            low=ohlc.get("low", np.nan),
            close=ohlc.get("close", np.nan),
            volume=ohlc.get("vol", 0),
            ltp=getattr(ltpc, "ltp", None),
        )


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Tick:
    """
    Dataclass to store a tick data.
    """

    key: str = ""
    timestamp: datetime | None = None
    ohlc_1d: Candle | None = None
    ohlc_1m: Candle | None = None
    greeks: Greeks | None = None
    ltpc: LTPC | None = None
    depth: list[dict] | None = None
    oi: float = 0
    market_open: bool | None = False

    @classmethod
    def parse_tick(cls, key, feed: dict, timestamp: str, market_status: bool):
        try:
            logger.debug("Parsing tick")
            market_data = feed.get("marketFF", {})
            if not market_data:
                logger.warning(f"Market data not available for {key}.")
                return None

            # EXTRACTING MARKET DATA
            market_levels = market_data.get("marketLevel", {}).get("bidAskQuote", [])
            market_ohlc = market_data.get("marketOHLC", {}).get("ohlc", [])
            greeks = market_data.get("optionGreeks", {})
            ltpc = market_data.get("ltpc", {})
            oi = market_data.get("oi", np.nan)

            ohlc_1d_obj = None
            ohlc_1m_obj = None
            # INSTANTIATING CLASSES
            greeks = Greeks.parse(greeks)
            ltpc = LTPC.parse(ltpc)

            for item in market_ohlc:
                interval = item.get("interval")
                if interval == "1d":
                    ohlc_1d_obj = Candle.load_ohlc(ohlc=item, ltpc=ltpc)
                elif interval == "I1":
                    ohlc_1m_obj = Candle.load_ohlc(ohlc=item, ltpc=ltpc)
            logger.info(f"Tick parsed successfully for {key}")
            return cls(
                key=key,
                timestamp=to_ist(int(timestamp)),
                greeks=greeks,
                ltpc=ltpc,
                depth=market_levels,
                oi=oi,
                ohlc_1d=ohlc_1d_obj,
                ohlc_1m=ohlc_1m_obj,
                market_open=market_status,
            )
        except Exception as e:
            logger.exception(f"Exception while parsing tick: {e}")
            return None


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Order(DatatypeBase):
    """Class to hold order details recieved from upstox.

    Returns:
        Order
    """

    order_id: str = "0"
    transaction_type: str | None = None
    order_timestamp: datetime = to_ist(0)
    exchange: str | None = None
    product: str | None = None
    price: float | None = None
    quantity: int | None = None
    status: str | None = None
    tag: str | None = None
    instrument_token: str | None = None
    placed_by: str | None = None
    trading_symbol: str | None = None
    tradingsymbol: str | None = None
    order_type: str | None = None
    validity: str | None = None
    trigger_price: float | None = None
    disclosed_quantity: int | None = None
    average_price: float | None = None
    filled_quantity: int | None = None
    pending_quantity: int | None = None
    status_message: str | None = None
    status_message_raw: str | None = None
    exchange_order_id: str | None = None
    parent_order_id: str | None = None
    variety: str | None = None
    exchange_timestamp: str | None = None
    is_amo: bool | None = None
    order_request_id: str | None = None
    order_ref_id: str | None = None
    remark: str = ""


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Position(DatatypeBase):
    """
    Dataclass to hold positions retrieved from upstox
    """

    exchange: str | None = None
    multiplier: float = 0
    value: float = 0
    pnl: float = 0
    product: str | None = None
    instrument_token: str | None = None
    average_price: float = 0
    buy_value: float = 0
    overnight_quantity: int = 0
    day_buy_value: float = 0
    day_buy_price: float = 0
    overnight_buy_amount: float = 0
    overnight_buy_quantity: int = 0
    day_buy_quantity: int = 0
    day_sell_value: float = 0
    day_sell_price: float = 0
    overnight_sell_amount: float = 0
    overnight_sell_quantity: int = 0
    day_sell_quantity: int = 0
    quantity: int = 0
    last_price: float = 0
    unrealised: float = 0
    realised: float = 0
    sell_value: float = 0
    trading_symbol: str | None = None
    close_price: float = 0
    buy_price: float = 0
    sell_price: float = 0
    stoploss: float = 0
    target: float = 0

    def update_position(self, **kwargs):
        required_fields = {f.name for f in fields(self)}
        filtered_dict = {k: v for k, v in kwargs.items() if k in required_fields}
        for key, value in filtered_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)



@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Portfolio:
    funds: Funds = field(default_factory=Funds)
    positions: dict[str, Position] = field(default_factory=dict)
    report: list[Trade] = field(default_factory=list)

    def get_report(self, verbose=True) -> pd.DataFrame:

        data = [asdict(trade) for trade in self.report]
        df = pd.json_normalize(data)
        if df.empty:
            return pd.DataFrame()
        df.sort_values(by="buy_timestamp").reset_index(drop=True)
        if verbose:
            from num2words import num2words

        display_df = df.copy()
        if "trade_id" in display_df.columns:
            display_df = display_df.drop(columns=["trade_id"])

        # 1. Round the dataframe FIRST
        float_cols = display_df.select_dtypes(include=["float"]).columns
        display_df[float_cols] = display_df[float_cols].round(2)

        # 2. Print with floatfmt=".2f" to forcefully format all floats in the markdown table
        print(
            display_df[
                [
                    "buy_timestamp",
                    "pnl",
                    "remark",
                    "buy_qty",
                    "sell_qty",
                    "buy_price",
                    "sell_price",
                ]
            ].to_markdown(tablefmt="pretty", floatfmt=".2f")
        )

        print(f"Total movement: {display_df['movement'].sum():.2f}")
        print(f"Final pnl: {display_df['pnl'].sum():.2f}")
        print(
            f"Final pnl (words): {num2words(display_df['pnl'].sum().round(), lang='en_IN')}"
        )

        # 3. Use single quotes inside the method arguments to avoid breaking the f-string
        logger.info(
            f"Final Results:\n{display_df.to_markdown(tablefmt='pretty', floatfmt='.2f')}"
        )
        logger.info(f"Total movement: {display_df['movement'].sum():.2f}")
        logger.info(f"Final pnl: {display_df['pnl'].sum():.2f}")
        return df





@dataclass
class FinancialState:
    user: str = "USER"
    debt: float = 0.0
    trading_capital: float = 100000.0      # Maps to starting_amount
    savings: float = 0.0
    unrealized_profit: float = 0.0
    target: float = 50000000.0             # Maps to lifetime target
    base_pay: float = 0.0
    growth_factor: float = 0.1             # Maps to current_multiplier
    external_pnl: float = 468045.54        # Maps to past losses/profits 

    @classmethod
    def load_from_file(cls, path: Path):
        """Loads state from JSON. Gracefully handles missing files or extra keys."""
        if not path.exists() or path.stat().st_size == 0:
            return cls()
            
        with open(path, "r") as f:
            state_dict = json.load(f)
            return parse_obj_to_dataclass(cls,state_dict)            

    def save_to_file(self, path: Path):
        """Saves current state to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
