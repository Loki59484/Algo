from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from functools import partial
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import numpy as np
import logging
import os

"""
This module contains custom dataclasses for smooth handling of trading data.
"""


# Set up logging
logger = logging.getLogger(__name__)


def to_ist(target: pd.Series | list | int | float, unit="ms"):
    """
    Converts UNIX timestamps (ms) to strict naive IST objects.
    Safely handles scalars, lists, and Pandas Series.
    """
    try:
        parsed = pd.to_datetime(target, unit="ms", utc=True)
    except ValueError:
        parsed = pd.to_datetime(target, utc=True)

    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    return parsed.tz_convert("Asia/Kolkata").tz_localize(None)


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
                ts=ohlc.get("ts", "0"),
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
            ohlc_1d_obj=ohlc_1d_obj,
            ohlc_1m_obj=ohlc_1m_obj,
        )


@dataclass
class Trade:
    trade_id: str = 0
    side : str | None = None
    timestamp: datetime = to_ist(0)
    buy_price: float = 0
    buy_qty: int = 0
    sell_price: float = 0
    sell_qty: int = 0
    movement: float = 0
    PnL: float = 0
    remark: str = ""
    stoploss: float | None = None
    target: float | None = None
    buy_adx: float | None = None
    buy_DMP: float | None = None
    buy_DMN: float | None = None
    buy_EMA: float | None = None
    buy_RSI: float | None = None
    buy_SUPT: float | None = None
    buy_VWAP: float | None = None

    @classmethod
    def from_candle(
        cls,
        candle,
        qty: int,
        id: str = "111",
        side: str | None = None
    ):
        """
        Creates a trade object using the current tick for backtesting purposes.
        Please use `Trade.from_order()` for live trading purposes.

        Args
        -------
        tick: Tick,
            Tick object for reading market state when the order was placed.
        id: str, optional
            Identifier for the trade. Defaults to 111 or order id of buy order, whichever is provided.

        Returns:
        --------
        Trade
        """
        return Trade(
            trade_id=id,
            side=side,
            timestamp=candle.timestamp,
            buy_price=candle.close,
            buy_qty=qty,
            buy_adx=candle.ADXR_14_2,
            buy_DMP=candle.DMP_14,
            buy_DMN=candle.DMN_14,
            buy_EMA=candle.EMA_200,
            buy_RSI=candle.RSI_14,
            buy_SUPT=candle.SUPERT_14_2,
            buy_VWAP=candle.VWAP_D,
        )

    def close_trade(self, price: float, qty: int, remark: str = ""):
        """Closes an open trade using a sell order

        Args:
            price (float): Selling price
            qty (int): Sold quantity
        """
        self.sell_price = price
        self.sell_qty = qty
        self.movement = self.sell_price - self.buy_price
        self.PnL = self.sell_qty * self.movement
        self.remark = remark


@dataclass(slots=True)
class Position:
    key: str | None = None
    open: bool = False
    trades: list[Trade] = field(default_factory=list)
    open_trade: Trade | None = None
    PnL: float = 0
    report: pd.DataFrame | None = None

    def open_position(self, trade: Trade):
        try:
            self.open = True
            self.trades.append(trade)
            self.open_trade = trade
            logger.info(f"Position opened for {self.key}")
        except Exception:
            logger.exception("Error while opening position")

    def close_position(self):
        self.open = False
        self.PnL = sum([trade.PnL for trade in self.trades])

    def trade_report(self):
        if self.trades:
            self.report = pd.DataFrame(
                {
                    "Timestamp": [trade.timestamp for trade in self.trades],
                    "Side": [trade.side for trade in self.trades],
                    "Buy_price": [trade.buy_price for trade in self.trades],
                    "Buy_qty": [trade.buy_qty for trade in self.trades],
                    "Sell_price": [trade.sell_price for trade in self.trades],
                    "Sell_qty": [trade.sell_qty for trade in self.trades],
                    "Movement": [trade.movement for trade in self.trades],
                    "PnL": [trade.PnL for trade in self.trades],
                    "Remark": [trade.remark for trade in self.trades],
                    "ADX": [trade.buy_adx for trade in self.trades],
                    "DMP": [trade.buy_DMP for trade in self.trades],
                    "DMN": [trade.buy_DMN for trade in self.trades],
                    "EMA": [trade.buy_EMA for trade in self.trades],
                    "RSI": [trade.buy_RSI for trade in self.trades],
                    "Supertrend": [trade.buy_SUPT for trade in self.trades],
                    "VWAP": [trade.buy_VWAP for trade in self.trades],
                }
            )
        return self.report


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
        from core.upstox_func import is_nse_holiday, DATA_DIR, get_historical
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
