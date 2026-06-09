from __future__ import annotations
from dataclasses import field, fields, asdict
from typing import Literal
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil import relativedelta
from collections import deque
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
    """"""

    starting_capital: float  # changes only after settlement
    pnl: float = 0.0
    total: float = 0.0
    used_margin: float = 0.0
    available_margin: float = 0.0 # always <= starting capital
    adhoc_margin: float= 0.0
    available_margin: float= 0.0
    exposure_margin: float= 0.0
    notional_cash: float= 0.0
    payin_amount: float= 0.0
    span_margin: float= 0.0

    @classmethod
    def parse_funds_json(cls:Funds,data:dict,new:bool=True) -> Funds|None:
        if new:
            cls = cls(starting_capital = data['equity']['available_margin'])

        cls.adhoc_margin: float= data['equity']['adhoc_margin']
        cls.available_margin: float= data['equity']['available_margin']
        cls.exposure_margin: float= data['equity']['exposure_margin']
        cls.notional_cash: float= data['equity']['notional_cash']
        cls.payin_amount: float= data['equity']['payin_amount']
        cls.span_margin: float= data['equity']['span_margin']
        cls.used_margin: float= data['equity']['used_margin']
        return cls if new else None

    def __post_init__(self):
        logger.info(f"Starting with capital:{self.starting_capital}")
        self.available_margin = self.total = self.starting_capital

    def credit(self, amount):
        self.total += amount
        self.available_margin = (
            self.total if self.total < self.starting_capital else self.starting_capital
        )
        self.pnl = self.total - self.starting_capital
        self.used_margin = max(0, self.used_margin - amount)

    def debit(self, amount):
        if amount > self.available_margin:
            logger.error("Insufficient funds to proceed.")
            return -1
        self.total -= amount
        self.available_margin = (
            self.total if self.total < self.starting_capital else self.starting_capital
        )
        self.pnl = self.total - self.starting_capital
        self.used_margin += amount

    def settle(self, simulate:bool=True, client=None):
        if not simulate and client is None:
            logger.error("A client instance of `UpstoxClient` class is required if not simulating [simulate=False].")
        logger.info(
            f"Day settled with starting :{self.starting_capital} | PnL: {self.pnl} | available: {self.available_margin}"
        )
        if simulate:
            self.available_margin = self.starting_capital = self.total
            self.pnl = 0
            return
        self.parse_funds_json(cls=self,data=client.get_funds(),new=False)


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
    buy_signal: bool = None
    sell_signal: bool = None

    @classmethod
    def load_ohlc(cls, ohlc: dict,ltpc: LTPC):
        return cls(
            timestamp=to_ist(ohlc.get("ts", "0")),
            open=ohlc.get("open", np.nan),
            high=ohlc.get("high", np.nan),
            low=ohlc.get("low", np.nan),
            close=ohlc.get("close", np.nan),
            volume=ohlc.get("vol", 0),
            ltp=getattr(ltpc,'ltp',None)
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
    market_open : bool | None = False

    @classmethod
    def parse_tick(cls, key, feed: dict, timestamp: str,market_status:bool):
        try:
            logger.debug("Parsing tick")
            market_data = feed.get("marketFF", {})
            if not market_data:
                logger.warning("Market data not available.")
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
                    ohlc_1d_obj = Candle.load_ohlc(ohlc=item,ltpc=ltpc)
                elif interval == "I1":
                    ohlc_1m_obj = Candle.load_ohlc(ohlc=item,ltpc=ltpc)


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
        self.date = pd.to_datetime(date).date() if date else None
        self.historical_candles: deque[Candle] = deque(maxlen=100000)
        self.expiry = pd.to_datetime(expiry).date() if expiry else None
        self.exchange:Literal["NSE","BSE"]="NSE"

    @classmethod
    def load_previous(cls, client, ins: Instrument, from_date: datetime,to_date:datetime,isexpired:bool=True):
        from core.upstox_methods import DATA_DIR
        from core.anatomy import load_parquet
        from tools.download_historical import download_cache

        CACHE_DIR = DATA_DIR / "cache"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        target_file: Path = (
            DATA_DIR
            / "historical"
            / ins.exchange
            / from_date.strftime("%Y")
            / from_date.strftime("%m")
            / from_date.strftime("%d")
            / f"{ins.interval}_{ins.unit}"
            / f"{ins.key}.parquet"
        )

        CACHE_FILE = CACHE_DIR / f"{ins.key}_{from_date}_{to_date}.parquet"
        data = None

        if target_file.exists():

            data = load_parquet(target_file).get("data", pd.DataFrame())
            logger.info(
                f"Data loaded for {ins.key} | Date {from_date} - {to_date} from file."
            )

        elif CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 0:

            data = pd.read_parquet(CACHE_FILE)
            logger.info(
                f"Data loaded for {ins.key} | Date {from_date} - {to_date} from cache."
            )
        else:
            if not isexpired:
                logger.info("Loading historical data from Upstox.")
                data = client.get_historical(instrument_key=ins.key,
                    from_date=from_date, to_date=to_date
                )
            else:
                logger.info(f"Saving cache for expired instrument from Upstox at {CACHE_FILE}.")

                data = download_cache(
                    ins.key,
                    is_expired=True,
                    expiry=ins.expiry,
                    from_date=from_date,
                    to_date=to_date,
                    out_path=CACHE_FILE,
                )
        if data is None or data.empty:
            logger.error(
                f"Could not load previous trading date data | Key : {ins.key} | Date = {from_date}"
            )
            return
        else:
            return data

    @classmethod
    def parse_options(
        cls, client, options: pd.DataFrame, lookback: int = 0
    ) -> list[Instrument]:
        """
        Parses a DataFrame containing options data into different Instrument class objects.

        Parameters
        ----------
        options : pd.DataFrame
            DataFrame containing options data where option contains fields provided directly by Upstox API.
        lookback : int, optional
            Number of previous days for which data is to be loaded, by default 0

        Returns
        -------
        list[Instrument]
        """

        parsed_options = []
        try:
            for option in tqdm(options.itertuples(),desc="Parsing options",leave=False,total=len(options)):
                data_dfs = []                
                ins = cls(
                    instrument_key=option.instrument_key,
                    expiry=option.expiry,
                )
                ins.lot_size = getattr(option,"lot_size")
                ins.freeze_qty = getattr(option,"freeze_quantity")
                ins.type = getattr(option,"instrument_type")
                ins.strike_price = getattr(option,"strike_price")
                ins.date = getattr(option, "date", datetime.today().date())
                ins.exchange = getattr(option,"exchange")
                #if ins.date == datetime.today().date():
                #    data = client.get_historical(dtype='intraday',instrument_key=ins.key, from_date=ins.date, to_date=ins.date)
                #    data_dfs.append(data)
                #current_day = ins.date - timedelta(days=1)
                end_date = current_day 

                while lookback > 0:
                    if client.is_exchange_holiday(current_day, ins.exchange):
                        #logger.info(f"Skipping holiday/weekend : {current_day}")
                        current_day -= timedelta(days=1)
                    else:
                        lookback -= 1
                        current_day -= timedelta(days=1)

                chunk_start = current_day 

                while chunk_start <= end_date:
                    chunk_end = min(chunk_start + relativedelta(months=1) - timedelta(days=1), end_date)
                    
                    logger.info(f"Fetching historical chunk: {chunk_start} to {chunk_end}")
                    
                    # Fetch the chunk
                    chunk_data = cls.load_previous(
                        client, 
                        ins, 
                        from_date=chunk_start, 
                        to_date=chunk_end, 
                        isexpired=True
                    )
                    
                    data_dfs.append(chunk_data)  
                    # Shift the start date for the next loop iteration
                    chunk_start = chunk_end + timedelta(days=1)

                data_dfs.reverse()
                if not data_dfs:
                    continue
                data_dfs = [df for df in data_dfs if not df.empty]
                data = pd.concat(data_dfs, ignore_index=True)
                if not data.empty:
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
                    parsed_options.append(ins)

        except Exception as e:
            logger.info(f"Exception while parsing options: \n{e}")
        return parsed_options

    @classmethod
    def load_multiple(cls,client,source: list[Path] | list[dict], lookback: list | int = 0):
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
            max_workers=1#os.cpu_count()
        ) as executor:
            if in_ram:
                futures = [
                    executor.submit(cls.load_instrument,client=client, **src, lookback=lb)
                    for src, lb in zip(source, lookbacks)
                ]
            else:
                futures = [
                    executor.submit(cls.load_instrument,client=client, path=src, lookback=lb)
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
        client,
        path: Path | str | None = None,
        lookback: int = 0,
        data: pd.DataFrame | None = None,
        metadata: dict | None = None,
    ):
        """
        Loads instrument data and metadata into the instance. The data is expected to be a DataFrame containing the candles, and
        the metadata is expected to contain keys like 'instrument_key', 'date', 'expiry', 'lot_size', 'strike_price', and 'freeze_quantity'.
        """
        from core.upstox_methods import DATA_DIR, UpstoxClient
        from core.anatomy import load_parquet
        from dateutil.relativedelta import relativedelta
        from datetime import timedelta

        if path is None and data is None:
            raise ValueError(
                "Either a path to parquet file or a pd.DataFrame and a metadata must be provided. None was provided."
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
        ins.exchange=str(metadata.get("exchange","NSE"))
        
# --- NEW BATCH CHUNKING LOGIC ---
        historical_dfs = []
        
        if lookback > 0:
            end_date = ins.date - timedelta(days=1)
            start_date = end_date
            days_found = 0
            
            # 1. Walk backwards to find the exact start_date
            while days_found < lookback:
                if client.is_exchange_holiday(start_date, exchange=ins.exchange):
                    #logger.info(f"Skipping holiday/weekend : {start_date}")
                    pass
                else:
                    days_found += 1
                
                if days_found < lookback:
                    start_date -= timedelta(days=1)

            # 2. Pre-calculate the explicit chunk boundaries to prevent infinite loops
            date_chunks = []
            curr_start = start_date
            while curr_start <= end_date:
                curr_end = min(curr_start + relativedelta(months=1) - timedelta(days=1), end_date)
                date_chunks.append((curr_start, curr_end))
                curr_start = curr_end + timedelta(days=1)

            # 3. Execute the strictly defined chunks
            for c_start, c_end in date_chunks:
                logger.info(f"Fetching historical chunk: {c_start} to {c_end}")
                
                chunk_data = cls.load_previous(
                    client, 
                    ins, 
                    from_date=c_start, 
                    to_date=c_end, 
                    isexpired=True
                )
                
                if chunk_data is not None and not chunk_data.empty:
                    historical_dfs.append(chunk_data)

        # --------------------------------

        data_dfs = historical_dfs + [data]
        data = pd.concat(data_dfs, ignore_index=True) if len(data_dfs) > 1 else data
        # --------------------------------

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
        df.sort_values(by="Buy_timestamp").reset_index(drop=True)
        if verbose:
            from num2words import num2words

        display_df = df.copy()
        if "Trade_id" in display_df.columns:
            display_df.drop(columns=["Trade_id"], inplace=True)

        # 1. Round the dataframe FIRST
        float_cols = display_df.select_dtypes(include=["float"]).columns
        display_df[float_cols] = display_df[float_cols].round(2)

        # 2. Print with floatfmt=".2f" to forcefully format all floats in the markdown table
        print(
            display_df[["Buy_timestamp", "PnL", "Remark", "Buy_qty", "Sell_qty", "Buy_price", "Sell_price"]]
            .to_markdown(tablefmt="pretty", floatfmt=".2f")
        )

        print(f"Total Movement: {display_df['Movement'].sum():.2f}")
        print(f"Final PnL: {display_df['PnL'].sum():.2f}")
        print(f"Final PnL (words): {num2words(display_df['PnL'].sum().round(), lang='en_IN')}")

        # 3. Use single quotes inside the method arguments to avoid breaking the f-string
        logger.info(
            f"Final Results:\n{display_df.to_markdown(tablefmt='pretty', floatfmt='.2f')}"
        )
        logger.info(f"Total Movement: {display_df['Movement'].sum():.2f}")
        logger.info(f"Final PnL: {display_df['PnL'].sum():.2f}")
        return df


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Bucket:
    """A Bucket to collect instruments for being traded together."""

    date: datetime
    tag: str | None = None
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


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Trade:
    Trade_id: str = None
    Instrument_key :str = None
    Side: str = None
    Buy_timestamp: datetime = None
    Buy_price: float = None
    Buy_qty: int = None
    Sell_timestamp: datetime = None
    Sell_price: float = None
    Sell_qty: int = None
    Movement: float = None
    PnL: float = None
    Remark: str = ""
    Buy_conditions: dict = None
    Sell_conditions: dict = None
    total: float = None
