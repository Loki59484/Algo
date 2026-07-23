"""
Module containing Upstox functions and methods.
NOTE : All functions are defined to be used with Upstox API. Refer to the official documentation for more details on the API and its usage.
"""

from google.protobuf.json_format import MessageToDict
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from collections import deque
from typing import Literal, Optional
from threading import Lock
from pathlib import Path
import pandas as pd
import websockets
import subprocess
import threading
import requests
import logging
import asyncio
import inspect
import joblib
import time
import gzip
import json
import sys
import ssl
import os
import io

subprocess.run("cls" if os.name == "nt" else "clear")

# SETTING UP DIRECTORIES

CORE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CORE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"
CONFIG_DIR = ROOT_DIR / "config"
SECRETS_PATH = ROOT_DIR / ".secrets"
TOKEN_FILE = SECRETS_PATH / "access_token.json"
SANDBOX_TOKEN_FILE = SECRETS_PATH / "sandbox_access_token.json"
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
ARCHIVE_PATH = (
    LOG_DIR / "archives" / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
LOG_FILE = LOG_DIR / "logs.log"
_HOLIDAY_CACHE: dict[int, dict[str, str] | None] = {}
EXPIRED_CACHE_DIR = (
    Path(__file__).resolve().parent / "data" / "cache" / "expired_instruments"
)
EXPIRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
# IMPORT CUSTOM MODULES
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from core.protobuffs import MarketDataFeedV3_pb2 as pb
from core.datatypes import Order, Tick
from core.methods import to_ist

# STATIC VARIABLES
if not ENV_PATH.exists():
    os.environ
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("UPSTOX_API_KEY")
API_SECRET = os.getenv("UPSTOX_API_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
MOBILE_NUM = os.getenv("MOBILE_NUMBER")
ALGO_NAME = os.getenv("ALGO_NAME")
AWS_TAILSCALE_IP = os.getenv("AWS_TAILSCALE_IP", None)

# LOGGING CONFIGURATION

logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger()

# SET LOGGING LEVEL
logging.basicConfig(level=logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s | %(name)s | | %(levelname)s | [%(module)s.%(funcName)s:%(lineno)d] | %(message)s"
)

# ARCHIVE HANDLER TO STORE AN ARCHIVE LOG OF EVERY RUN
archive_handler = logging.FileHandler(ARCHIVE_PATH)
archive_handler.setFormatter(formatter)
logger.addHandler(archive_handler)


# HANDLER TO STORE LOG OF ONLY THE LATEST RUN
latest_handler = logging.FileHandler(LOG_FILE, mode="w")
latest_handler.setFormatter(formatter)
latest_handler.setLevel(logging.DEBUG)
logger.addHandler(latest_handler)

# CONSOLE HANDLER FOR STREAMING LOG TO CONSOLE
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.ERROR)
logger.addHandler(console_handler)

api_lock = threading.Lock()
# ------------------------------------------------------------#

logger = logging.getLogger(__name__)


class UpstoxRateLimiter:
    """
    A thread-safe rate limiter tracking Upstox limits:
    - 50 per second
    - 500 per minute
    - 2000 per 30 minutes
    """

    def __init__(self):
        self.lock = Lock()
        self.sec_history = deque()
        self.min_history = deque()
        self.half_hr_history = deque()

    def wait_for_token(self):
        try:
            stack = inspect.stack()
            api_method = stack[2].function if len(stack) > 2 else "Unknown API"
            origin_func = stack[3].function if len(stack) > 3 else "Unknown Logic"
            origin_line = stack[3].lineno if len(stack) > 3 else 0

            leak_info = (
                f"[{api_method}() triggered by {origin_func}() at line {origin_line}]"
            )
        except Exception:
            leak_info = "[Unknown Caller]"

        logger.debug(f"Token consumed by: {leak_info}")

        with self.lock:
            now = time.time()

            while self.sec_history and now - self.sec_history[0] > 1.0:
                self.sec_history.popleft()
            while self.min_history and now - self.min_history[0] > 60.0:
                self.min_history.popleft()
            while self.half_hr_history and now - self.half_hr_history[0] > 1800.0:
                self.half_hr_history.popleft()

            # Check 1-Second Limit (Buffer at 45)
            if len(self.sec_history) >= 45:
                sleep_time = 1.0 - (now - self.sec_history[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            if len(self.min_history) >= 480:
                sleep_time = 60.0 - (now - self.min_history[0])
                logger.warning(
                    f"Minute Rate Limit Approaching. Thread pausing for {sleep_time:.2f}s. Culprit -> {leak_info}"
                )
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            if len(self.half_hr_history) >= 1950:
                sleep_time = 1800.0 - (now - self.half_hr_history[0])
                logger.warning(
                    f"30-Minute Rate Limit Approaching. Pausing for {sleep_time:.2f}s. Culprit -> {leak_info}"
                )
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            self.sec_history.append(now)
            self.min_history.append(now)
            self.half_hr_history.append(now)


class UpstoxClient:
    """
    API handler containing methods for communicating with API
    """

    def __init__(self):
        self.session = requests.Session()
        self.limiter = UpstoxRateLimiter()
        # Fetch tokens dynamically. If they are missing/expired, it throws an error immediately.
        self.access_token = self.get_access_token()

        try:
            self.sandbox_access_token = self.get_sandbox_access_token()
        except Exception as e:
            logger.warning(
                f"Sandbox Token not loaded: {e}. Sandbox mode will be unavailable."
            )
            self.sandbox_access_token = None

        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
                "X-Algo-Name": ALGO_NAME,
            }
        )

    def _make_request(
        self, method: str, url: str, max_retries=3, return_json=True, **kwargs
    ):
        """
        Internal method handling Rate Limiting, HTTP Sessions, and Retries.
        """
        retries = 0
        backoff_time = 2.0

        while retries <= max_retries:
            self.limiter.wait_for_token()

            try:
                response = self.session.request(method, url, **kwargs)
                if response.status_code == 401:
                    logger.error(
                        f"HTTP 401 Hit! Unauthorized. Retrying after {backoff_time}s..."
                    )
                    # This will raise a fatal ValueError if the token wasn't updated externally
                    self.access_token = self.get_access_token()
                    self.session.headers.update(
                        {"Authorization": f"Bearer {self.access_token}"}
                    )
                    time.sleep(backoff_time)
                    retries += 1
                    backoff_time *= 2
                    continue

                if response.status_code == 429:
                    logger.error(f"HTTP 429 Hit! Backing off for {backoff_time}s...")
                    time.sleep(backoff_time)
                    retries += 1
                    backoff_time *= 2
                    continue

                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    logger.exception(
                        f"HTTP error while making request\n {response.text}"
                    )

                return response.json() if return_json else response

            except requests.exceptions.RequestException as e:
                logger.exception(f"Network error on {url}: {e}")
                retries += 1
                time.sleep(backoff_time)

        logger.critical(f"Failed to fetch {url} after {max_retries} retries.")
        return None

    def get_access_token(self):
        """Strictly fetches the saved access token. Throws an error if invalid."""
        if TOKEN_FILE.exists():
            with open(TOKEN_FILE, "r+") as f:
                data = json.load(f)

            if datetime.now() > datetime.strptime(data["expiry"], "%Y-%m-%d %H:%M:%S"):
                logger.critical("Access token expired.")
                raise ValueError(
                    "Access token expired. Please run your standalone login script to generate a new token."
                )
            else:
                return data["access_token"]
        else:
            logger.critical("No access token found.")
            raise FileNotFoundError(
                f"Access token file not found at {TOKEN_FILE}. Please run your standalone login script."
            )

    def set_static_ip(self, prim_ip: str, sec_ip: str = ""):
        url = "https://api.upstox.com/v2/user/ip"
        data = {"primary_ip": f"{prim_ip}", "secondary_ip": f"{sec_ip}"}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        response = self._make_request("PUT", url=url, data=data, headers=headers)
        logger.info(response)

    def get_static_ip(self):
        url = "https://api.upstox.com/v2/user/ip"
        response = self._make_request("GET", url=url)
        return response["data"]

    def kill_switch(
        self,
        segments: list[
            Literal[
                "BSE_EQ",
                "NSE_EQ",
                "NCD_FO",
                "BCD_FO",
                "NSE_FO",
                "BSE_FO",
                "MCX_FO",
                "NSE_COM",
            ]
        ],
        action: Literal["ENABLE", "DISABLE"],
    ):
        url = "https://api.upstox.com/v2/user/kill-switch"
        payload = [
            {"segment": f"{segment}", "action": f"{action}"} for segment in segments
        ]
        response = self._make_request("POST", url=url, data=json.dumps(payload))
        return response

    def get_funds(self):
        funds_url = "https://api.upstox.com/v2/user/get-funds-and-margin"
        response = self._make_request("GET", funds_url)

        if response:
            funds = response["data"]
            return funds
        else:
            logger.error(f"Failed to get funds info!\n{response}")
            return 0

    def _fetch_and_cache_expired(
        self, underlying: str, expiry_date: str
    ) -> pd.DataFrame:
        safe_underlying = underlying.replace("|", "_").replace(" ", "_")
        cache_path = EXPIRED_CACHE_DIR / f"{safe_underlying}_{expiry_date}.joblib"

        if cache_path.exists():
            return joblib.load(cache_path)

        url = f"https://api.upstox.com/v2/expired-instruments/option/contract?instrument_key={underlying}&expiry_date={expiry_date}"
        response = self._make_request("GET", url)

        if not response or "data" not in response:
            logger.error(
                f"Failed to retrieve data for expired instruments! Response: {response}"
            )
            return pd.DataFrame()

        df = pd.DataFrame(response["data"])

        if not df.empty:
            joblib.dump(df, cache_path)

        return df

    def get_expired_instruments(
        self,
        instrument_key: str = "",
        expiry_date: str = "",
        underlying: Literal[
            "NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"
        ] = "NSE_INDEX|Nifty 50",
    ) -> Optional[pd.DataFrame]:
        data = self._fetch_and_cache_expired(underlying, expiry_date)
        if data.empty:
            return None if instrument_key != "" else data

        if instrument_key == "":
            return data

        matched_keys = data.loc[
            data["instrument_key"].str.contains(instrument_key, regex=False),
            "instrument_key",
        ]

        return matched_keys if not matched_keys.empty else None

    def _resolve_expired_key(
        self,
        instrument_key: str,
        expiry_date: str = None,
        underlying: Literal[
            "BSE_INDEX|SENSEX", "NSE_INDEX|Nifty 50"
        ] = "NSE_INDEX|Nifty 50",
    ) -> str | None:
        if expiry_date is not None:
            key = self.get_expired_instruments(
                instrument_key=instrument_key,
                expiry_date=expiry_date,
                underlying=underlying,
            )
            if isinstance(key, pd.Series):
                return key.iloc[0] if not key.empty else None
            return key

        expiries = self.get_options_with_expiry(options=instrument_key, is_expired=True)

        for exp_d in reversed(expiries):
            key = self.get_expired_instruments(
                instrument_key=instrument_key, expiry_date=exp_d, underlying=underlying
            )
            if isinstance(key, pd.Series):
                key = key.iloc[0] if not key.empty else None

            if key is not None:
                return key

        return None

    def get_historical(
        self,
        dtype: Literal["historical", "intraday"] = "historical",
        instrument_key: str = "NSE_INDEX|Nifty 50",
        interval: int = 1,
        unit: str = "minutes",
        to_date=None,
        from_date=None,
        is_expired: bool = False,
        expiry_date=None,
        expired_key=None,
    ):
        if dtype not in {"historical", "intraday"}:
            logger.warning(f"Invalid Value for 'dtype': {dtype}")
            raise TypeError("Possible values for 'dtype' : {'historical', 'intraday'}")

        if "NSE" in instrument_key[:3]:
            underlying = "NSE_INDEX|Nifty 50"
        elif "BSE" in instrument_key[:3]:
            underlying = "BSE_INDEX|SENSEX"

        to_date = to_date or date.today()
        from_date = from_date or (date.today() - timedelta(days=2))

        if dtype == "intraday":
            url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"

        elif expired_key:
            url = f"https://api.upstox.com/v2/expired-instruments/historical-candle/{expired_key}/1minute/{to_date}/{from_date}"

        elif is_expired and "INDEX" not in instrument_key:
            resolved_key = self._resolve_expired_key(
                instrument_key, expiry_date, underlying=underlying
            )
            if not resolved_key:
                logger.error(f"Could not resolve expired key for {instrument_key}")
                return None
            url = f"https://api.upstox.com/v2/expired-instruments/historical-candle/{resolved_key}/1minute/{to_date}/{from_date}"

        else:
            url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"

        logger.debug(
            f"Making request to get historical data from {from_date} to {to_date}."
        )

        with api_lock:
            response = self._make_request(method="GET", url=url)

        if not response:
            logger.warning("API Request Failed/Timeout.")
            logger.warning(
                f"Empty Dataframe returned for {instrument_key} | from_date: {from_date} | to_date: {to_date}",
                stack_info=True,
            )
            return None

        candles = response.get("data", {}).get("candles", [])

        if not candles:
            logger.warning(
                f"Empty Dataframe returned for {instrument_key} | from_date: {from_date} | to_date: {to_date} | response: \n{response}",
                stack_info=True,
            )
            return pd.DataFrame()

        candles.reverse()
        return pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "vol", "oi"]
        )

    def get_all_options(
        self,
        expiry="",
        dtype="contract",
        instrument_key: Literal[
            "NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"
        ] = "NSE_INDEX|Nifty 50",
    ):
        try:
            if dtype == "contract":
                url = "https://api.upstox.com/v2/option/contract"
            elif dtype == "chain":
                if expiry != "":
                    url = "https://api.upstox.com/v2/option/chain"
                else:
                    raise ValueError("Expiry required to pull an option chain")
            else:
                raise ValueError("dtype can only be 'contract' or 'chain'")
            params = {"instrument_key": f"{instrument_key}", "expiry_date": f"{expiry}"}
            payload = {}
            logger.debug(f"Making request to get all options for expiry {expiry}")
            response = self._make_request("GET", url, params=params, data=payload)

            if response:
                data = response["data"]
                data_df = pd.DataFrame(data)
                return pd.DataFrame(data) if not data_df.empty else None
            else:
                logger.error(f"{response}")
            if expiry == "":
                logger.info(
                    "Expiry not specified. All available instruments retrieved!"
                )
        except ValueError as v:
            logger.error(v)

    def get_marketquote(
        self,
        instrument_key: (
            Literal["NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"] | list[str]
        ) = "NSE_INDEX|Nifty 50",
    ):
        url = "https://api.upstox.com/v2/market-quote/quotes"
        payload = {}
        params = {"instrument_key": instrument_key}

        response = self._make_request("GET", url, data=payload, params=params)
        if response:
            data = response["data"]
            return data

    def place_order(
        self,
        instrument_token,
        transaction_type: Literal["BUY", "SELL"],
        price=0,
        quantity="75",
        product: Literal["I", "D", "MTF"] = "D",
        validity: Literal["DAY", "IOC"] = "DAY",
        tag="string",
        order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET",
        trigger_price=0,
        is_amo=False,
        sandbox=False,
        **kwargs,
    ):
        data = {
            "quantity": quantity,
            "product": product,
            "validity": validity,
            "price": price,
            "tag": tag,
            "instrument_token": instrument_token,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "disclosed_quantity": 0,
            "trigger_price": trigger_price,
            "is_amo": is_amo,
            "slice": True,
        }

        if sandbox:
            if not self.sandbox_access_token:
                logger.error("Sandbox token not configured!")
                return None
            url = "https://api-sandbox.upstox.com/v3/order/place"
            headers = {
                "Authorization": f"Bearer {self.sandbox_access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            response = self._make_request("POST", url, json=data, headers=headers)
        else:
            url = "https://api-hft.upstox.com/v3/order/place"
            response = self._make_request("POST", url, json=data)

        if response:
            return response

    def get_trades_for_day(self):
        url = "https://api.upstox.com/v2/order/trades/get-trades-for-day"
        response = self._make_request("GET", url=url)
        return response["data"] if response else None

    def get_order_history(self, order_id=""):
        url = "https://api.upstox.com/v2/order/history"
        params = {"order_id": f"{order_id}"}
        response = self._make_request("GET", url=url, params=params)
        return response["data"] if response else None

    def get_order_details(self, order_id=""):
        url = "https://api.upstox.com/v2/order/details"
        params = {"order_id": f"{order_id}"}
        response = self._make_request("GET", url, params=params)
        return Order.parse(response["data"])

    def modify_order(self, id: str, sandbox=False, **kwargs):
        data = {"order_id": id, **kwargs}

        if sandbox:
            if not self.sandbox_access_token:
                return None
            url = "https://api-sandbox.upstox.com/v3/order/place"
            headers = {
                "Authorization": f"Bearer {self.sandbox_access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            response = self._make_request("POST", url, json=data, headers=headers)
        else:
            url = "https://api-hft.upstox.com/v3/order/modify"
            response = self._make_request("PUT", url=url, json=data)

        return response

    def cancel_order(self, order_id, sandbox=False):
        payload = {"order_id": order_id}
        if sandbox:
            if not self.sandbox_access_token:
                return None
            url = "https://api-sandbox.upstox.com/v3/order/cancel"
            headers = {
                "Authorization": f"Bearer {self.sandbox_access_token}",
                "Accept": "application/json",
            }
            response = self._make_request(
                method="DELETE", url=url, data=payload, headers=headers
            )
        else:
            url = "https://api-hft.upstox.com/v3/order/cancel"
            response = self._make_request(method="DELETE", url=url, data=payload)

        if response:
            return response["data"]
        else:
            logger.error(response, stack_info=True)

    def cancel_multi_orders(
        self,
        segments: Literal[
            "BSE_EQ",
            "NSE_EQ",
            "NCD_FO",
            "BCD_FO",
            "NSE_FO",
            "BSE_FO",
            "MCX_FO",
            "NSE_COM",
        ] = "",
        tag="",
    ):

        if segments:
            filter_by = f"?segments={segments}"
        elif tag:
            filter_by = f"?tag={tag}"
        url = "https://api.upstox.com/v2/order/multi/cancel"+ filter_by
        

        response = self._make_request(method="DELETE",url=url)

        if response:
            return response["data"]
        else:
            logger.error(response, stack_info=True)

    def get_market_data_feed_authorize_v3(self):
        url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
        api_response = self._make_request("GET", url=url)
        return api_response

    def decode_protobuf(self, buffer):
        feed_response = pb.FeedResponse()
        feed_response.ParseFromString(buffer)
        return feed_response

    async def subscribe_ticks(
        self,
        buffer: asyncio.Queue,
        instrument_key: Literal[
            "NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"
        ] = "NSE_INDEX|Nifty 50",
        mode: Literal["ltpc", "option_greeks", "full", "full_d30"] = "full_d30",
    ) -> Tick:
        global data_ready
        if isinstance(instrument_key, str):
            instrument_key = [instrument_key]

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        response = self.get_market_data_feed_authorize_v3()
        async with websockets.connect(
            response["data"]["authorized_redirect_uri"], ssl=ssl_context
        ) as websocket:
            await asyncio.sleep(1)

            data = {
                "guid": "13syxu852ztodyqncwt0",
                "method": "sub",
                "data": {"mode": mode, "instrumentKeys": instrument_key},
            }

            binary_data = json.dumps(data).encode("utf-8")
            await websocket.send(binary_data)

            logger.info(f"Getting Data for {instrument_key}")
            while True:
                logger.info("New round")
                message = await websocket.recv()
                decoded_data = self.decode_protobuf(message)
                data_dict = MessageToDict(decoded_data)
                market_status = None
                if "type" in data_dict and data_dict["type"] == "market_info":
                    logger.info(data_dict)
                    market_status = (
                        True
                        if data_dict["marketInfo"]["segmentStatus"][
                            f"{instrument_key[0][:3]}_FO"
                        ]
                        == "NORMAL_OPEN"
                        else False
                    )
                else:
                    try:
                        if buffer is not None:
                            logger.info("Preparing data for buffer")
                            ts = data_dict.get("currentTs", "0")
                            new_ticks = {
                                (key, to_ist(ts).date()): Tick.parse_tick(
                                    key,
                                    feed_data["fullFeed"],
                                    timestamp=ts,
                                    market_status=market_status,
                                )
                                for key, feed_data in data_dict.get("feeds", {}).items()
                                if "fullFeed" in feed_data
                            }
                            logger.info("Data ready")

                            if buffer.full():
                                try:
                                    buffer.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                            buffer.put_nowait(new_ticks)
                            logger.info("Data put")
                        else:
                            logger.error("Failed to put data into the output queue.")
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        logger.exception(
                            f"Error while parsing or queing live data \n{e}."
                        )

    def get_portfolio_stream_url(self):
        url = "https://api.upstox.com/v2/feed/portfolio-stream-feed/authorize"
        api_response = self._make_request("GET", url=url)
        if api_response is None:
            logger.error(f"Failed to get portfolio stream URL: {api_response}")
            return None
        return api_response

    async def subscribe_portfolio(self, output: asyncio.Queue):
        while True:
            ws_url = self.get_portfolio_stream_url()["data"]["authorized_redirect_uri"]
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info("Connected to portfolio stream WebSocket.")

                    while True:
                        logger.info("Starting portfolio updater")
                        message = await websocket.recv()
                        data = json.loads(message)
                        logger.info(f"Received portfolio update: {data}")
                        await output.put(data)
            except Exception as e:
                logger.error(f"Error in portfolio stream WebSocket: {e}")
                logger.info("Attempting to reconnect to portfolio stream WebSocket...")
            logger.info("Reconnecting to portfolio stream WebSocket...")
            await asyncio.sleep(5)

    def get_options_with_expiry(
        self, options: pd.DataFrame | str, is_expired: bool = False, return_df=False
    ) -> list[datetime, pd.DataFrame]:
        if is_expired:
            if isinstance(options, str):
                url = f"https://api.upstox.com/v2/expired-instruments/expiries?instrument_key={options}"
                response = self._make_request("GET", url)
                if response:
                    return response["data"]
                else:
                    logger.error(f"Error while fetching expiries |\n{response}")
        else:
            dump = []
            for option in options.itertuples():
                expiry = datetime.strptime(option.expiry, "%Y-%m-%d")
                dump.append(expiry)
            dump = list(set(dump))
            dump.sort()

            if datetime.now() > dump[0]:
                expiry_date = date.strftime(dump[1], "%Y-%m-%d")
                return expiry_date, (
                    options.loc[options["expiry"] == expiry_date]
                    if return_df
                    else date.strftime(dump[1], "%Y-%m-%d")
                )
            else:
                expiry_date = date.strftime(dump[0], "%Y-%m-%d")
                return expiry_date, (
                    options.loc[options["expiry"] == expiry_date]
                    if return_df
                    else date.strftime(dump[0], "%Y-%m-%d")
                )

    def get_brokerage(
        self, price, instrument_token, quantity, transaction_type="BUY", product="D"
    ) -> None:
        url = "https://api.upstox.com/v2/charges/brokerage"
        params = {
            "instrument_token": instrument_token,
            "quantity": quantity,
            "product": product,
            "transaction_type": transaction_type,
            "price": price,
        }
        response = self._make_request("GET", url, params=params)
        if response:
            data = response["data"]["charges"]["total"]
            return data
        else:
            logger.error(
                f"Error while fetching brokerage |\n{response}", stack_info=True
            )
            return None

    def get_total_charges(
        self,
        from_date: datetime = datetime.today().date(),
        to_date: datetime = datetime.today().date(),
        segment="FO",
        financial_year="2526",
    ):
        url = "https://api.upstox.com/v2/trade/profit-loss/charges"
        params = {
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
            "segment": segment,
            "financial_year": financial_year,
        }
        response = self._make_request("GET", url, params=params)
        if response:
            data = response["data"]["charges_breakdown"]["total"]
            return data
        else:
            logger.error(
                f"Error while fetching total charges |\n{response}", stack_info=True
            )
            return None

    def get_pnl_report(
        self,
        from_date: datetime = datetime.today().date(),
        to_date: datetime = datetime.today().date(),
        segment="FO",
        financial_year="2627",
        pagenumber=1,
        pagesize=20,
    ):
        url = "https://api.upstox.com/v2/trade/profit-loss/data"
        params = {
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
            "segment": segment,
            "financial_year": financial_year,
            "page_number": pagenumber,
            "page_size": pagesize,
        }
        response = self._make_request("GET", url, params=params)
        if response:
            data = response["data"]
            return data
        else:
            logger.error(
                f"Error while fetching Profit and Loss report |\n{response}",
                stack_info=True,
            )
            return None

    def get_report_metadata(
        self,
        from_date: datetime = datetime.today().date(),
        to_date: datetime = datetime.today().date(),
        segment="FO",
        financial_year="2526",
    ):
        url = "https://api.upstox.com/v2/trade/profit-loss/metadata"
        params = {
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
            "segment": segment,
            "financial_year": financial_year,
        }
        response = self._make_request("GET", url, params=params)
        if response:
            data = response["data"]
            return data
        else:
            logger.error(
                f"Error while fetching report metadata |\n{response}", stack_info=True
            )
            return None

    def get_charges_report(
        self,
        from_date: datetime = datetime.today().date(),
        to_date: datetime = datetime.today().date(),
        segment="FO",
        financial_year="2526",
    ):
        url = "https://api.upstox.com/v2/trade/profit-loss/charges"
        params = {
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
            "segment": segment,
            "financial_year": financial_year,
        }
        response = self._make_request("GET", url, params=params)
        if response:
            data = response["data"]["charges_breakdown"]["total"]
            return data
        else:
            print("Failed to get trade charges!")
            print(response)
            return None

    def get_holidays(self, date=""):
        url = f"https://api.upstox.com/v2/market/holidays/{date}"
        response = self._make_request("GET", url)
        if response:
            data = response["data"]
            return data
        else:
            print("Failed to retrieve data for holidays.")
            return None

    def get_order_book(self):
        url = "https://api.upstox.com/v2/order/retrieve-all"
        response = self._make_request("GET", url)
        if response:
            data = response["data"]
            return data
        else:
            print("Failed to retrieve data for holidays.")
            return None

    def exchanges_status(self, exchange: Literal["BSE", "NSE"] = "NSE"):
        url = f"https://api.upstox.com/v2/market/status/{exchange}"
        response = self._make_request("GET", url)
        if response:
            data = response["data"]["status"]
            return data
        else:
            logger.error(
                f"Error while fetching exchange status |\n{response}", stack_info=True
            )
            return response

    def get_positions(self):
        url = "https://api.upstox.com/v2/portfolio/short-term-positions"
        response = self._make_request("GET", url)
        if response:
            return response["data"]
        else:
            logger.error(
                f"Error while fetching positions |\n{response}", stack_info=True
            )

    def update_database(self):
        indices = ["NSE", "BSE"]
        for idx in indices:
            url = f"https://assets.upstox.com/market-quote/instruments/exchange/{idx}.json.gz"
            response = self._make_request("GET", url, stream=True, return_json=False)
            if not response:
                logger.error(f"Failed to download {idx} database from Upstox.")
                return None
            try:
                print("Updating local database", end="\r")
                with gzip.open(io.BytesIO(response.content), "rb") as gzfile:
                    decompressed_file = gzfile.read()
                    jsonstr = decompressed_file.decode(encoding="utf-8")
                    json_data = json.loads(jsonstr)
                    df = pd.DataFrame(json_data)
                    df.to_parquet(
                        DATA_DIR / f"{idx}_DATABASE.parquet", engine="pyarrow"
                    )
                    logger.info("Local database updated successfully.")
            except Exception as e:
                logger.exception(f"Exception while updating local database. \n {e}")
                return None

    def exitall(self):
        url = "https://api.upstox.com/v2/order/positions/exit"
        data = {}
        try:
            response = self._make_request("POST", url, json=data)
            return response
        except Exception as e:
            logger.error(
                f"Error while exiting positions |\n{e}\n{response}", stack_info=True
            )

    def is_exchange_holiday(
        self, date: datetime, exchange: Literal["NSE", "BSE"] = "NSE"
    ) -> bool:
        if date.weekday() >= 5:
            return True

        year = date.year
        date_str = date.strftime("%Y-%m-%d")

        if year not in _HOLIDAY_CACHE:
            path: Path = CONFIG_DIR / "holidays" / f"{year}.json"

            if path.exists():
                with open(path, "r") as file:
                    raw_data = json.load(file)

                _HOLIDAY_CACHE[year] = {
                    item["date"]: str(item.get("closed_exchanges", ""))
                    for item in raw_data
                }
            else:
                logger.info(
                    f"Holiday file for {year} not found. Fetching full list from Upstox..."
                )
                try:
                    all_holidays = self.get_holidays()
                    year_data = [
                        h
                        for h in all_holidays
                        if h.get("date", "").startswith(str(year))
                    ]

                    if year_data:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "w") as file:
                            json.dump(year_data, file, indent=4)

                        _HOLIDAY_CACHE[year] = {
                            item["date"]: str(item.get("closed_exchanges", ""))
                            for item in year_data
                        }
                        logger.info(
                            f"Successfully cached {len(year_data)} holidays for {year}."
                        )
                    else:
                        logger.warning(f"Upstox returned no holidays for {year}.")
                        _HOLIDAY_CACHE[year] = {}

                except Exception as e:
                    logger.exception(
                        f"Failed to auto-download holiday list for {year}: {e}"
                    )
                    _HOLIDAY_CACHE[year] = {}

        year_holidays = _HOLIDAY_CACHE[year]
        if date_str in year_holidays and exchange in year_holidays[date_str]:
            return True

        return False

    def generate_holidays(self):
        holidays_data = pd.read_json(CONFIG_DIR / "holidays.json")
        holidays_data["Gdate"] = to_ist(holidays_data["date"]).dt.year
        holidays_data["date"] = holidays_data["date"].dt.strftime("%Y-%m-%d")
        grouped = holidays_data.groupby("Gdate")
        for year, df in grouped:
            path: Path = CONFIG_DIR / "holidays" / f"{year}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            df = df.reset_index(drop=True)
            df.to_json(path, orient="records")
