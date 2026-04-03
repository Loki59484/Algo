from dataclasses import field, fields
from pydantic.dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from typing import Callable
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import numpy as np
import logging
import sys
import os

"""
This module contains custom dataclasses for smooth handling of trading data.
"""

# Set up logging
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core.methods import to_ist, parse_object


class Parsable:
    """ Base class for inheriting object parsing ability"""    

    def parse(cls,obj:dict):
        """
        Base method used to parse dicts recieved from Upstox into relevent dataclass
        """
        from dataclasses import fields
        
        required_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in obj.items() if k in required_fields}
        return cls(**filtered_dict)



@dataclass(slots=True)
class Funds:
    starting_capital: float = 0
    total: float = 0
    used: float = 0
    available: float = 0

    def __post_init__(self):
        self.available = self.total = self.starting_capital


@dataclass(slots=True)
class Greeks:
    """
    Dataclass to store greeks for a tick.
    """

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass(slots=True)
class LTPC:
    """
    Dataclass to store LTPC data of a tick.
    """

    ltp: float = np.nan
    ltt: str = "0"
    ltq: str = "0"
    cp: float = np.nan


@dataclass(slots=True)
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
    # CUSTOM FIELDS
    buy_signal: bool = 0
    sell_signal: bool = 0


@dataclass(slots=True)
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

    @classmethod
    def parse_tick(cls, key, feed: dict, timestamp: str):
        market_data = feed.get("marketFF", {})
        if not market_data:
            return None

        def load_ohlc(ohlc: dict):

            candle = Candle(
                timestamp=to_ist(ohlc.get("ts", "0")),
                open=ohlc.get("open", np.nan),
                high=ohlc.get("high", np.nan),
                low=ohlc.get("low", np.nan),
                close=ohlc.get("close", np.nan),
                volume=ohlc.get("vol", 0),
            )
            return candle

        # EXTRACTING MARKET DATA
        market_levels = market_data.get("marketLevel", {}).get("bidAskQuote", [])
        market_ohlc = market_data.get("marketOHLC", {}).get("ohlc", [])
        greeks = market_data.get("optionGreeks", {})
        ltpc = market_data.get("ltpc", {})
        oi = market_data.get("oi", np.nan)

        ohlc_1d_obj = None
        ohlc_1m_obj = None

        for item in market_ohlc:
            interval = item.get("interval")
            if interval == "1d":
                ohlc_1d_obj = load_ohlc(item)
            elif interval == "I1":
                ohlc_1m_obj = load_ohlc(item)

        # INSTANTIATING CLASSES
        greeks = Greeks(
            delta=greeks.get("delta", np.nan),
            gamma=greeks.get("gamma", np.nan),
            theta=greeks.get("theta", np.nan),
            vega=greeks.get("vega", np.nan),
            rho=greeks.get("rho", np.nan),
        )
        ltpc = LTPC(
            ltp=ltpc.get("ltp", np.nan),
            ltq=ltpc.get("ltq", "0"),
            ltt=ltpc.get("ltt", "0"),
            cp=ltpc.get("cp", np.nan),
        )

        return cls(
            key=key,
            timestamp=to_ist(int(timestamp)),
            greeks=greeks,
            ltpc=ltpc,
            depth=market_levels,
            oi=oi,
            ohlc_1d=ohlc_1d_obj,
            ohlc_1m=ohlc_1m_obj,
        )


@dataclass(slots=True)
class Order:
    """Class to hold order details recieved from upstox.

    Returns:
        Order
    """

    order_id: str = 0
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
    order_timestamp: str | None = None
    exchange_timestamp: str | None = None
    is_amo: bool | None = None
    order_request_id: str | None = None
    order_ref_id: str | None = None
    remark: str = ""

    @classmethod
    def parse_order(cls, order: dict):
        """
        Parses upstox order into Order class object.
        """
        return parse_object(cls,order)


@dataclass(slots=True)
class Position:
    """
    Dataclass to hold positions retrieved from upstox
    """

    exchange: str | None = None
    multiplier: float | None = None
    value: float | None = None
    pnl: float | None = None
    product: str | None = None
    instrument_token: str | None = None
    average_price: float | None = None
    buy_value: float | None = None
    overnight_quantity: int | None = None
    day_buy_value: float | None = None
    day_buy_price: float | None = None
    overnight_buy_amount: float | None = None
    overnight_buy_quantity: int | None = None
    day_buy_quantity: int | None = None
    day_sell_value: float | None = None
    day_sell_price: float | None = None
    overnight_sell_amount: float | None = None
    overnight_sell_quantity: int | None = None
    day_sell_quantity: int | None = None
    quantity: int | None = None
    last_price: float | None = None
    unrealised: float | None = None
    realised: float | None = None
    sell_value: float | None = None
    trading_symbol: str | None = None
    close_price: float | None = None
    buy_price: float | None = None
    sell_price: float | None = None

    @classmethod
    def parse_position(cls, position: dict):
        """Parses position update recieved from upstox.

        Args:
            position (dict): dict of position data.

        Returns:
            Position
        """
        return parse_object(cls,position)

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
    ):

        self.lot_size: int = 0
        self.key = instrument_key
        self.interval: str = "1"
        self.unit: str = "minute"
        self.freeze_qty: float = 0.0
        self.type: str | None = None
        self.strike_price: float = 0.0
        self.last_minute_ticks: list[Tick] = []
        self.historical_df: pd.DataFrame | None = None
        self.position: Position = Position(key=self.key)
        self.date = pd.to_datetime(date).date() if date else None
        self.historical_candles: deque[Candle] = deque(maxlen=800)
        self.expiry = pd.to_datetime(expiry).date() if expiry else None

    @classmethod
    def _worker(cls, source, lookback):
        if isinstance(source, Path):
            return cls.load_instrument(path=source, lookback=lookback)
        if isinstance(source, dict):
            return cls.load_instrument(**source, lookback=lookback)
        else:
            raise TypeError(f"Unrecognized source type: {type(source)}")

    @classmethod
    def load_multiple(cls, source: list[Path] | list[dict], lookback: list | int = 0):
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
            max_workers=os.cpu_count()
        ) as executor:
            if in_ram:
                futures = [
                    executor.submit(cls.load_instrument, **src, lookback=lb)
                    for src, lb in zip(source, lookbacks)
                ]
            else:
                futures = [
                    executor.submit(cls.load_instrument, path=src, lookback=lb)
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

    @classmethod
    def load_instrument(
        cls,
        path: Path | None = None,
        lookback: int = 0,
        data: pd.DataFrame | None = None,
        metadata: dict | None = None,
    ):
        """
        Loads instrument data and metadata into the instance. The data is expected to be a DataFrame containing the candles, and
        the metadata is expected to contain keys like 'instrument_key', 'date', 'expiry', 'lot_size', 'strike_price', and 'freeze_quantity'.
        """
        from core.upstox_methods import is_nse_holiday, DATA_DIR, get_historical
        from core.anatomy import load_parquet
        from scripts.download_historical import download_cache

        CACHE_DIR = DATA_DIR / "cache"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if path is None and data is None:
            raise ValueError(
                "Either a path to parquet file or a pd.DataFrame and a metadata must be provided. None was provided here"
            )

        if path is not None:
            data, metadata = load_parquet(path).values()

        def load_previous(ins: Instrument, prev_trading_day: datetime):

            target_dir: Path = (
                DATA_DIR
                / "historical"
                / prev_trading_day.strftime("%Y")
                / prev_trading_day.strftime("%m")
                / prev_trading_day.strftime("%d")
                / f"{ins.interval}_{ins.unit}"
                / f"{ins.key}.parquet"
            )

            CACHE_FILE = CACHE_DIR / f"{ins.key}.parquet"
            data = None

            if target_dir.exists():

                data = load_parquet(target_dir).get("data", pd.DataFrame())
                logger.info(
                    f"Data loaded for {ins.key} | Date {prev_trading_day} from file."
                )

            elif CACHE_FILE.exists():

                data = pd.read_parquet(CACHE_FILE)
                logger.info(
                    f"Data loaded for {ins.key} | Date {prev_trading_day} from cache."
                )
            else:
                if ins.expiry is None:
                    data = get_historical(
                        from_date=prev_trading_day, to_date=prev_trading_day
                    )
                else:
                    data = download_cache(
                        ins.key,
                        is_expired=True,
                        expiry=ins.expiry,
                        date=prev_trading_day,
                        out_path=CACHE_FILE,
                    )
            if data is None or data.empty:
                logger.error(
                    f"Could not load previous trading date data | Key : {ins.key} | Date = {prev_trading_day}"
                )
                return
            else:
                return data

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
        prev_trading_day = ins.date - timedelta(1)
        loaded_days = 0
        data_dfs = [data]
        while loaded_days < lookback:
            if is_nse_holiday(prev_trading_day):
                logger.info(f"Skipping holiday/weekend : {prev_trading_day}")
                prev_trading_day -= timedelta(days=1)
                continue
            data_dfs.append(load_previous(ins, prev_trading_day))
            prev_trading_day -= timedelta(1)
            loaded_days += 1

        data_dfs.reverse()
        data = pd.concat(data_dfs, ignore_index=True) if len(data_dfs) > 1 else data

        if data is not None and not data.empty:
            clean_data = data.copy()
            clean_data[["open", "high", "low", "close"]] = clean_data[
                ["open", "high", "low", "close"]
            ].ffill()
            clean_data["vol"] = (
                pd.to_numeric(clean_data["vol"], errors="coerce")
                .fillna(0)
                .astype(float)
            )
            candles = [
                Candle(timestamp=t, open=o, high=h, low=l, close=c, volume=v)
                for t, o, h, l, c, v in zip(
                    to_ist(clean_data["timestamp"]),
                    clean_data["open"],
                    clean_data["high"],
                    clean_data["low"],
                    clean_data["close"],
                    clean_data["vol"],
                )
            ]
            ins.historical_candles.extend(candles)
        return ins

    def to_dataframe(self, candles: deque[Candle] | None = None) -> pd.DataFrame:
        """
        Converts the historical candle data to a pandas DataFrame.
        """
        if candles is None:
            candles = self.historical_candles
        if candles is None:
            return pd.DataFrame()

        self.historical_df = pd.DataFrame(
            {
                "timestamp": [c.timestamp for c in candles],
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume": [c.volume for c in candles],
                "buy_signal": [c.buy_signal for c in candles],
                "sell_signal": [c.sell_signal for c in candles],
            }
        )
        self.historical_df.set_index("timestamp", inplace=True)
        self.historical_df.sort_index()
        return self.historical_df


@dataclass(slots=True)
class Portfolio:
    funds: Funds = field(default_factory=Funds)
    positions: dict[str, Position] = field(default_factory=dict)
    scheme: Callable | None = None

    def update_funds(self, amount: float, used: bool = True):
        if used:
            self.funds.used += amount
            self.funds.available -= amount
        else:
            self.funds.total += amount
            self.funds.available += amount

    def test_portfolio(self, target: pd.DataFrame, **kwargs):
        """Tests the portfolio by applying the scheme to calculate the returns."""
        if self.scheme is not None:
            return self.scheme(target, **kwargs)
        else:
            logger.warning("No scheme defined for portfolio.")
            return target


@dataclass
class Bucket:
    """A Bucket to collect instruments for being traded together."""

    date: datetime
    spot: Instrument | None = None
    legs: dict[str, Instrument] = field(default_factory=dict)
    open_position: Position | None = None

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
