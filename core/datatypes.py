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
from core.methods import to_ist, parse_obj_to_dataclass


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


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Bucket:
    """A Bucket to collect instruments for being traded together."""

    date: datetime
    tag: str | None = None
    spot: Instrument | None = None
    legs: dict[str, Instrument] = field(default_factory=dict)
    open_position: Position | None = None
    probability_matrix: Any = None
    pending_entry: Any = None
    pending_exit: Any = None 
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


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Trade:
    trade_id: None | str = None
    instrument_key: None | str = None
    side: None | str = None
    buy_timestamp: None | datetime = None
    buy_price: None | float = None
    buy_qty: None | int = None
    sell_timestamp: None | datetime = None
    sell_price: None | float = None
    sell_qty: None | int = None
    movement: None | float = None
    pnl: None | float = None
    remark: None | str = ""
    buy_conditions: None | dict = None
    sell_conditions: None | dict = None
    total: None | float = None

class Instrument:
    """
    A class representing a financial instrument, such as a stock or option.
    The class is designed to handle both the data (candles) and associated metadata
    (like instrument key, date, expiry, lot size, strike price, and freeze quantity) in a structured way.
    """

    def __init__(
        self,
        instrument_key: str,
        date: str | datetime = None,
        expiry: str | datetime = None,
        data: pd.DataFrame | None = None,
        instrument_type: str | None = None,
    ):

        self.lot_size: int = 0
        self.key = instrument_key
        self.interval: str = "1"
        self.unit: str = "minute"
        self.freeze_qty: float = 0.0
        self.type: str | None = instrument_type
        self.strike_price: float = 0.0
        self.date = pd.to_datetime(date).date() if date else None
        self.historical_candles: deque[Candle] = (
            deque(iterable=self.df_to_candles(data), maxlen=100000)
            if data is not None
            else deque(maxlen=100000)
        )
        self.expiry = pd.to_datetime(expiry).date() if expiry else None
        self.exchange: Literal["NSE", "BSE"] = "NSE"

    def load_previous(
        self, client, from_date: datetime, to_date: datetime, is_expired: bool = True
    ):
        from core.upstox_methods import DATA_DIR
        from tools.download_historical import download_cache

        CACHE_DIR = DATA_DIR / "cache"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        CACHE_FILE = CACHE_DIR / f"{self.key}_{from_date}_{to_date}.parquet"
        data = None
        if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 0:

            data = pd.read_parquet(CACHE_FILE)
            logger.info(
                f"Data loaded for {self.key} | Date {from_date} - {to_date} from cache."
            )
        else:
            if not is_expired:
                logger.info("Loading historical data from Upstox.")
                data = client.get_historical(
                    instrument_key=self.key, from_date=from_date, to_date=to_date
                )
            else:
                logger.info(
                    f"Saving cache for expired instrument from Upstox at {CACHE_FILE}."
                )
                data = download_cache(
                    self.key,
                    is_expired=True,
                    expiry=self.expiry,
                    from_date=from_date,
                    to_date=to_date,
                    out_path=CACHE_FILE,
                )
        if data is None or data.empty:
            logger.warning(
                f"Could not load previous trading date data | Key : {self.key} | Date = {from_date}"
            )
        else:
            return data

    @classmethod
    def parse_options(
        cls, client, options: pd.DataFrame, lookback: int = 0, is_expired: bool = False
    ) -> list["Instrument"]:
        parsed_options = []
        try:
            for option in tqdm(
                options.itertuples(),
                desc="Parsing options",
                leave=False,
                total=len(options),
            ):

                today_data = client.get_historical(
                    dtype="intraday", instrument_key=option.instrument_key
                )

                ins = cls(instrument_key=option.instrument_key, expiry=option.expiry)
                ins.lot_size = getattr(option, "lot_size")
                ins.freeze_qty = getattr(option, "freeze_quantity")
                ins.type = getattr(option, "instrument_type")
                ins.strike_price = getattr(option, "strike_price")
                ins.date = getattr(option, "date", datetime.today().date())
                ins.exchange = getattr(option, "exchange", "NSE")

                data_dfs = []
                if lookback > 0:
                    hist_data = ins.load_historical_df(
                        client, lookback, is_expired=is_expired
                    )
                    if not hist_data.empty:
                        data_dfs.append(hist_data)

                if today_data is not None and not today_data.empty:
                    data_dfs.append(today_data)

                if not data_dfs:
                    continue

                # 4. Concatenate and clean
                clean_data = pd.concat(data_dfs, ignore_index=True)
                candles = ins.df_to_candles(clean_data)
                ins.historical_candles.extend(candles)
                parsed_options.append(ins)

        except Exception as e:
            logger.exception(f"Exception while parsing options: \n{e}")
        return parsed_options

    @classmethod
    def load_multiple(
        cls,
        client,
        source: list[Path] | list[dict],
        lookback: list | int = 0,
        load_history: bool = False,
    ):
        """Loads multiple instruments at once using `concurrent.futures.ProcessPoolExecutor`.

        Args:
            source (list[Path]  list[dict]):
            A list of file paths or a dict with `keys : pair` values as `data : pd.Dataframe` and `metadata : dict` is required.

            lookback (list | int): Days for which prior warm-up data is to be loaded into the instrument.
            A list corresponding to the lookback days can also be provided for assigning different values of lookback.

        Returns:
            list[Instrument]: A list of Instrument objects created using `load_instrument()` method.
        """
        logger.debug(f"Loading {len(source)} files into Instrument objects")
        if isinstance(lookback, int):
            lookbacks = [lookback] * len(source)
        elif len(lookback) != len(source):
            raise ValueError(
                "Length of lookback list must exactly match length of source list."
            )
        else:
            lookbacks = lookback

        loaded_instruments = []
        if not source:
            return loaded_instruments

        in_ram = isinstance(source[0], dict)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1  # os.cpu_count()
        ) as executor:
            if in_ram:
                futures = [
                    executor.submit(
                        cls.load_instrument(
                            client=client, **src, lookback=lb, load_history=load_history
                        )
                    )
                    for src, lb in zip(source, lookbacks)
                ]
            else:
                futures = [
                    executor.submit(
                        cls.load_instrument,
                        client=client,
                        path=src,
                        lookback=lb,
                        load_history=load_history,
                    )
                    for src, lb in zip(source, lookbacks)
                ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Loading Instruments",
                leave=False,
            ):
                try:
                    inst = future.result()
                    if inst is not None:
                        loaded_instruments.append(inst)
                except Exception as e:
                    logger.exception(f"Failed to load instrument in parallel: {e}")
        loaded_instruments.sort(key=lambda inst: inst.key)
        logger.debug("Instruments loaded successfully.")
        return loaded_instruments

    def _calculate_lookback_dates(self, client, lookback):
        """Walks backwards to calculate the start and end dates based on holidays."""
        end_date = self.date
        start_date = end_date
        days_found = 0

        while days_found < lookback:
            if not client.is_exchange_holiday(start_date, exchange=self.exchange):
                days_found += 1
            if days_found < lookback:
                start_date -= timedelta(days=1)

        return start_date, end_date

    def _build_date_chunks(self, start_date, end_date):
        """Pre-calculates monthly chunk boundaries to avoid API batch restrictions."""

        date_chunks = []
        curr_start = start_date

        while curr_start <= end_date:
            curr_end = min(
                curr_start + relativedelta(months=1) - timedelta(days=1), end_date
            )
            date_chunks.append((curr_start, curr_end))
            curr_start = curr_end + timedelta(days=1)

        return date_chunks

    def _fetch_historical_chunks(self, client, date_chunks, is_expired):
        """Queries the upstream API for each chunk, silently ignoring non-existent periods."""
        historical_dfs = []
        for c_start, c_end in date_chunks:
            chunk_data = self.load_previous(
                client, from_date=c_start, to_date=c_end, is_expired=is_expired
            )
            if chunk_data is not None and not chunk_data.empty:
                historical_dfs.append(chunk_data)
        return historical_dfs

    def load_historical_df(self, client, lookback, is_expired) -> pd.DataFrame:
        """
        Loads historical data chunks and applies cleanup.
        Guarantees a clean DataFrame return type even if the contract didn't exist yet.
        """
        if lookback <= 0:
            return pd.DataFrame()

        start_date, end_date = self._calculate_lookback_dates(client, lookback)
        date_chunks = self._build_date_chunks(start_date, end_date)
        data_dfs = self._fetch_historical_chunks(client, date_chunks, is_expired)

        if not data_dfs:
            logger.warning(
                f"No historical data available for {self.key}. Contract likely was not listed."
            )
            return pd.DataFrame()

        data = pd.concat(data_dfs, ignore_index=True)

        clean_data = data.copy()
        clean_data[["open", "high", "low", "close"]] = clean_data[
            ["open", "high", "low", "close"]
        ].ffill()
        clean_data["vol"] = (
            pd.to_numeric(clean_data["vol"], errors="coerce").fillna(0).astype(float)
        )

        return clean_data

    @classmethod
    def load_instrument(
        cls,
        client,
        path: Path | str | None = None,
        lookback: int = 0,
        data: pd.DataFrame | None = None,
        metadata: dict | None = None,
        load_history: bool = False,
        is_expired: bool = True
    ):
        from core.methods import load_parquet

        if path is None and data is None:
            raise ValueError(
                "Either a path to parquet file or pd.DataFrame and metadata must be provided."
            )

        if path is not None:
            data, metadata = load_parquet(path).values()

        ins = cls(
            instrument_key=metadata.get("instrument_key", "UNKNOWN"),
            date=metadata.get("date"),
            expiry=metadata.get("expiry"),
        )
        ins.lot_size = int(metadata.get("lot_size", 0))
        ins.strike_price = float(metadata.get("strike_price", 0.0))
        ins.freeze_qty = float(metadata.get("freeze_quantity", 0.0))
        ins.unit = str(metadata.get("unit", "minute"))
        ins.interval = str(metadata.get("interval", "1"))
        ins.type = str(metadata.get("instrument_type", "Index"))
        ins.exchange = str(metadata.get("exchange", "NSE"))

        data_dfs = []
        if load_history and lookback > 0:
            hist_data = ins.load_historical_df(client, lookback, is_expired=is_expired)
            if not hist_data.empty:
                data_dfs.append(hist_data)

        if data is not None and not data.empty:
            data_dfs.append(data)

        if not data_dfs:
            return ins

        final_data = pd.concat(data_dfs, ignore_index=True)
        candles = ins.df_to_candles(final_data)
        ins.historical_candles.extend(candles)
        return ins
        

    def df_to_candles(self, df: pd.DataFrame) -> list[Candle]:
        if not df.empty:
            clean_data = df.copy()
            clean_data[["open", "high", "low", "close"]] = clean_data[
                ["open", "high", "low", "close"]
            ].ffill()
            clean_data["vol"] = (
                pd.to_numeric(clean_data["vol"], errors="coerce")
                .fillna(0)
                .astype(float)
            )
            
            def _normalize_tz(ts):
                if pd.isna(ts): return ts
                parsed = pd.to_datetime(ts)
                return parsed.tz_localize("Asia/Kolkata") if parsed.tzinfo is None else parsed.tz_convert("Asia/Kolkata")
            
            clean_data["timestamp"] = clean_data["timestamp"].apply(_normalize_tz)
                                    
            candles = [
                Candle(timestamp=t, open=o, high=h, low=l, close=c, volume=v)
                for t, o, h, l, c, v in zip(
                    clean_data["timestamp"],  # Passed directly without to_ist()
                    clean_data["open"],
                    clean_data["high"],
                    clean_data["low"],
                    clean_data["close"],
                    clean_data["vol"],
                )
            ]
        return candles

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