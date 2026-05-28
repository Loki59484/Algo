"""
Module containing Upstox functions and methods.
NOTE : All functions are defined to be used with Upstox API. Refer to the official documentation for more details on the API and its usage.
"""

from google.protobuf.json_format import MessageToDict
from playwright.sync_api import sync_playwright, Error
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from collections import deque
from typing import Literal
from threading import Lock
from pathlib import Path
import urllib.parse
import pandas as pd
import socketserver
import http.server
import websockets
import threading
import requests
import logging
import getpass
import asyncio
import time
import gzip
import json
import sys
import ssl
import os
import io

os.system("cls" if os.name == "nt" else "clear")

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
    LOG_DIR / "archives" / f"log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log"
)
LOG_FILE = LOG_DIR / "logs.log"
_HOLIDAY_CACHE: dict[int, dict[str, str] | None] = {}

ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
# IMPORT CUSTOM MODULES
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from core.protobuffs import MarketDataFeedV3_pb2 as pb
from core.datatypes import *
from core.methods import *

# STATIC VARIABLES
if not ENV_PATH.exists():
    os.environ
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("UPSTOX_API_KEY")
API_SECRET = os.getenv("UPSTOX_API_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
MOBILE_NUM = os.getenv("MOBILE_NUMBER")
ALGO_NAME = os.getenv("ALGO_NAME")

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


# ------------------------------------------------------------#


class UpstoxRateLimiter:
    """
    A thread-safe rate limiter tracking Upstox limits:
    - 50 per second
    - 500 per minute
    - 2000 per 30 minutes
    """

    def __init__(self):
        self.lock = Lock()
        # Deques to store the exact timestamp of every request
        self.sec_history = deque()
        self.min_history = deque()
        self.half_hr_history = deque()

    def wait_for_token(self):
        with self.lock:
            now = time.time()

            # 1. Clean up old timestamps that have expired
            while self.sec_history and now - self.sec_history[0] > 1.0:
                self.sec_history.popleft()
            while self.min_history and now - self.min_history[0] > 60.0:
                self.min_history.popleft()
            while self.half_hr_history and now - self.half_hr_history[0] > 1800.0:
                self.half_hr_history.popleft()

            # 2. Check 1-Second Limit (Buffer at 45 instead of 50 for safety)
            if len(self.sec_history) >= 45:
                sleep_time = 1.0 - (now - self.sec_history[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()  # Update 'now' after sleeping!

            # 3. Check 1-Minute Limit (Buffer at 480 instead of 500)
            if len(self.min_history) >= 480:
                sleep_time = 60.0 - (now - self.min_history[0])
                logger.warning(
                    f"Minute Rate Limit Approaching. Thread pausing for {sleep_time:.2f}s"
                )
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            # 4. Check 30-Minute Limit (Buffer at 1950)
            if len(self.half_hr_history) >= 1950:
                sleep_time = 1800.0 - (now - self.half_hr_history[0])
                logger.warning(
                    f"30-Minute Rate Limit Approaching. Pausing for {sleep_time:.2f}s"
                )
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            # 5. Log the new request timestamp into all buckets
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
        self.access_token = self.get_access_token()
        self.sandbox_access_token = self.get_sandbox_access_token()
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
                    logger.error(f"HTTP 401 Hit! Retrying after {backoff_time}s...")
                    self.access_token = self.get_access_token()
                    time.sleep(backoff_time)
                    retries += 1
                    backoff_time *= 2
                    continue

                # 3. Handle Rate Limit Error (429)
                if response.status_code == 429:
                    logger.error(f"HTTP 429 Hit! Backing off for {backoff_time}s...")
                    time.sleep(backoff_time)
                    retries += 1
                    backoff_time *= 2
                    continue

                # Raise an error for 500s or 400s
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    logger.exception(
                        f"HTTP error while making request\n {response.text} "
                    )

                # If successful, return the parsed JSON immediately
                return response.json() if return_json else response

            except requests.exceptions.RequestException as e:
                logger.exception(f"Network error on {url}: {e}")
                retries += 1
                time.sleep(backoff_time)

        logger.critical(f"Failed to fetch {url} after {max_retries} retries.")
        return None

    def authorization(self):  # Log in to Upstox API
        """
        Generates access token for a session through a browser and saves in a 'access_token.json' file for usage.
        """
        server_ready_event = threading.Event()
        server_stop_event = threading.Event()  # Event to signal the server to stop
        PORT = 5000
        CALLBACK_PATH = "/callback"

        class OAuthRedirectHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                global auth_code
                parsed_url = urllib.parse.urlparse(self.path)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                if parsed_url.path == CALLBACK_PATH and "code" in query_params:
                    auth_code = query_params.get("code", [None])[0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    server_stop_event.set()  # Signal the main program that we've received the code
                else:
                    self.send_response(404)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()

        def run_simple_server():
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("", PORT), OAuthRedirectHandler) as httpd:
                logger.info(
                    f"Local HTTP server listening on http://localhost:{PORT}{CALLBACK_PATH}..."
                )
                server_ready_event.set()
                while not server_stop_event.is_set():
                    httpd.handle_request()  # Handle one request
                    # Add a small sleep to prevent busy-waiting if no requests are coming
                    time.sleep(0.1)
                logger.info("Local HTTP server shutting down.")
                httpd.shutdown()  # Cleanly shut down the server

        logger.info("Authorization initiated")
        redirect_uri = "http://localhost:5000/callback"
        login_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={API_KEY}&redirect_uri={redirect_uri}"
        server_thread = threading.Thread(target=run_simple_server, daemon=True)
        server_thread.start()
        server_ready_event.wait(
            timeout=10
        )  # Wait up to 10 seconds for server to be ready
        if not server_ready_event.is_set():
            logger.critical("Error: Local server did not start in time. Exiting.")
            exit(1)

        def _run_playwright():
            try:
                with sync_playwright() as p:
                    try:
                        browser = p.chromium.launch(headless=True)
                    except Error:
                        browser = p.chromium.launch(
                            executable_path="/usr/bin/chromium-browser", headless=True
                        )

                    context = browser.new_context(ignore_https_errors=True)
                    page = context.new_page()
                    page.goto(login_url)
                    print("#" * 50, "LOGIN PAGE LOADED", "#" * 50)
                    print("Mobile Number : ", MOBILE_NUM)
                    page.fill("#mobileNum", MOBILE_NUM)
                    page.click("#getOtp")
                    page.fill("#otpNum", input("Enter OTP : "))
                    page.click("#continueBtn")
                    page.fill("#pinCode", getpass.getpass("Enter 6 digit PIN : "))
                    page.click("#pinContinueBtn")
                    page.wait_for_url(f"{redirect_uri}*", timeout=15000)
                    browser.close()

                token_url = "https://api.upstox.com/v2/login/authorization/token"  # Generating access token

                token_headers = {
                    "accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                }

                token_data = {
                    "code": auth_code,
                    "client_id": API_KEY,
                    "client_secret": API_SECRET,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                }

                data = self._make_request(
                    method="POST", url=token_url, headers=token_headers, data=token_data
                )
                if data:
                    logger.info("Login successful!")
                    print("#" * 50, "LOGIN SUCCESSFUL", "#" * 50)
                    now = datetime.now()
                    if 0 <= now.hour < 3 or (now.hour == 3 and 0 <= now.minute <= 30):
                        data["expiry"] = (
                            f"{(now).replace(hour = 3,minute=30, second=0, microsecond = 0)}"
                        )
                    else:
                        data["expiry"] = (
                            f"{(now+timedelta(days=1)).replace(hour = 3,minute=30, second=0, microsecond = 0)}"
                        )
                    SECRETS_PATH.mkdir(parents=True, exist_ok=True)
                    with open(TOKEN_FILE, "w") as outputfile:
                        outputfile.write(json.dumps(data))
                        return data["access_token"]

            except Exception as e:
                logger.exception("Error while getting access token. \n{e}")

        worker_thread = threading.Thread(target=_run_playwright)
        worker_thread.start()
        worker_thread.join()

    def get_access_token(self):
        if TOKEN_FILE.exists():
            with open(TOKEN_FILE, "r+") as f:
                data = json.load(f)
            if datetime.now() > datetime.strptime(data["expiry"], "%Y-%m-%d %H:%M:%S"):
                print("Access token expired. Generating new token...")
                access_token = self.authorization()
                self.update_database()
            else:
                access_token = data["access_token"]
        else:
            logger.info("No access token found. Generating new token...")
            print("No access token found. Generating new token...")
            access_token = self.authorization()
            self.update_database()
        return access_token

    def get_funds(self):  # Getting funds available
        funds_url = "https://api.upstox.com/v2/user/get-funds-and-margin"
        response = self._make_request("GET", funds_url)

        if response:
            funds = response["data"]
            return funds
        else:
            logger.error(f"Failed to get funds info!\n{response}")

            return 0

    def get_expired_instruments(
        self,
        instrument_key="",
        expiry_date="",
        underlying: Literal[
            "NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"
        ] = "NSE_INDEX|Nifty 50",
        keys_only=True,
    ):  # Getting expired instruments for a stock/index
        """
        Returns expired instruments for specified stock/index instrument for the given expiry date.
        If instrument key provided, it returns data containing that instrument key only in the form of a dataframe.
        If not provided, it returns data for all expired instruments for the given expiry date in the form of a dataframe.
        """
        url = f"https://api.upstox.com/v2/expired-instruments/option/contract?instrument_key={underlying}&expiry_date={expiry_date}"

        response = self._make_request("GET", url)

        if response:
            data = pd.DataFrame(response["data"])
            if instrument_key == "":
                return data
            keys = data["instrument_key"].loc[
                data["instrument_key"].str.contains(instrument_key, regex=False)
            ]
            keys = keys if keys.size > 0 else None
            return keys
        else:
            logger.error(
                f"Failed to retrieve data for expired instruments!\n{response}"
            )

    def get_historical(
        self,
        dtype="historical",
        instrument_key: Literal[
            "NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"
        ] = "NSE_INDEX|Nifty 50",
        interval: int = 1,
        unit: str = "minutes",
        to_date=date.today(),
        from_date=date.today() - timedelta(days=2),
        is_expired: bool = False,
        expiry_date=None,
        expired_key=None,
    ):
        """
        Returns historical candle data for specified intrument for given time interval. Structure of response = {"instrument_key" : "...", "candles" : {[...]}}
        """
        valid = {"historical", "intraday"}
        if dtype in valid:
            if dtype == "historical":
                if is_expired:
                    if expiry_date is None:
                        expiry_date = set(
                            self.get_options_with_expiry(
                                options=instrument_key, is_expired=True
                            )[::-1]
                        )
                        for date in expiry_date:
                            expired_key = self.get_expired_instruments(
                                instrument_key=instrument_key, expiry_date=date
                            )
                            if expired_key is not None:
                                break
                    else:
                        if expired_key is None:
                            expired_key = self.get_expired_instruments(
                                instrument_key=instrument_key, expiry_date=expiry_date
                            )

                    url = f"https://api.upstox.com/v2/expired-instruments/historical-candle/{expired_key}/1minute/{to_date}/{from_date}"
                else:
                    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
            elif dtype == "intraday":
                url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
            logger.debug(
                f"Making request to get historical data for {instrument_key} from {from_date} to {to_date}."
            )
            response = self._make_request(method="GET", url=url)

            if response:
                data = response["data"]
                data["candles"].reverse()
                df = pd.DataFrame(
                    data["candles"],
                    columns=["timestamp", "open", "high", "low", "close", "vol", "oi"],
                )
                df = df if df is not None else None
                if df.empty:
                    logger.warning(
                        f"Empty Dataframe recieved for key : {instrument_key} | from_date : {from_date} | to_date : {to_date} ",
                        stack_info=True,
                    )
                return df
            else:
                error_msg = (
                    response if response is not None else "API Request Failed/Timeout"
                )

                logger.warning(error_msg)
                logger.warning(
                    f"Empty Dataframe returned for key : {instrument_key} | from_date : {from_date} | to_date : {to_date} ",
                    stack_info=True,
                )
                return None
        else:
            logger.warning(f"Invalid Value for 'dtype' {dtype}")
            raise TypeError(f"Possible values for 'dtype' : {valid}")

    def get_all_options(
        self,
        expiry="",
        dtype="contract",
        instrument_key: Literal[
            "NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"
        ] = "NSE_INDEX|Nifty 50",
    ):
        """
        Returns available options for specified stock/index instrument.
        """
        try:
            if dtype == "contract":
                url = f"https://api.upstox.com/v2/option/contract"
            elif dtype == "chain":
                if expiry != "":
                    url = f"https://api.upstox.com/v2/option/chain"
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

    # Getting market quote
    def get_marketquote(
        self,
        instrument_key: (
            Literal["NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"] | list[str]
        ) = "NSE_INDEX|Nifty 50",
    ):
        """
        Returns market quote for given instrument(s) [upto 500 at a time]
        """
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
        """
        Place Buy or Sell order at market or sandbox
        """
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
            url = "https://api-sandbox.upstox.com/v3/order/place"
            headers = {
                "Authorization": f"Bearer {self.sandbox_access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            response = self._make_request("POST", url, json=data, headers=headers)
        else:
            url = "https://api-hft.upstox.com/v3/order/place"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            response = self._make_request("POST", url, json=data)

        if response:
            return response

    def get_order_details(self, order_id=""):
        """
        Get details of the specified order.
        """

        url = "https://api.upstox.com/v2/order/details"

        params = {"order_id": f"{order_id}"}

        response = self._make_request("GET", url, params=params)
        return Order.parse(response["data"])

    def modify_order(self, id: str, sandbox=False, **kwargs):
        """
        Modifies the provided order according to the given information.
        """
        data = {"order_id": id, **kwargs}

        if sandbox:
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
        """
        Cancel the specified order.
        """
        payload = {"order_id": order_id}
        if sandbox:
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

    def get_market_data_feed_authorize_v3(self):
        """Get authorization for market data feed."""
        url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
        api_response = self._make_request("GET", url=url)
        return api_response

    def decode_protobuf(self, buffer):
        """Decode protobuf message."""
        feed_response = pb.FeedResponse()
        feed_response.ParseFromString(buffer)
        return feed_response

    async def subscribe_ticks(
        self,
        buffer: asyncio.Queue,
        instrument_key:Literal[
            "NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"
        ] = "NSE_INDEX|Nifty 50",
        mode:Literal["ltpc","option_greeks","full","full_d30"]="full_d30",
    ) -> Tick:
        """
        Get data steam of live market data for given instrument keys.
        """
        global data_ready
        if isinstance(instrument_key, str):
            instrument_key = [instrument_key]
        # Create default SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        # Get market data feed authorization
        response = self.get_market_data_feed_authorize_v3()
        # Connect to the WebSocket with SSL context
        async with websockets.connect(
            response["data"]["authorized_redirect_uri"], ssl=ssl_context
        ) as websocket:
            await asyncio.sleep(1)  # Wait for 1 second

            # Data to be sent over the WebSocket
            data = {
                "guid": "13syxu852ztodyqncwt0",
                "method": "sub",
                "data": {"mode": mode, "instrumentKeys": instrument_key},
            }

            # Convert data to binary and send over WebSocket
            binary_data = json.dumps(data).encode("utf-8")
            await websocket.send(binary_data)

            # Continuously receive and decode data from WebSocket
            logger.info(f"Getting Data for {instrument_key}")
            while True:
                message = await websocket.recv()
                decoded_data = self.decode_protobuf(message)
                # Convert the decoded data to a dictionary
                data_dict = MessageToDict(decoded_data)
                market_status = None
                if "type" in data_dict.keys() and data_dict["type"] == "market_info":
                    logger.info(data_dict)
                    market_status = (
                        True
                        if data_dict["marketInfo"]["segmentStatus"][f"{instrument_key[:3]}_FO"]
                        == "NORMAL_OPEN"
                        else False
                    )
                else:
                    try:
                        if not buffer is None:
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
                            if buffer.full():
                                try:
                                    buffer.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                            buffer.put_nowait(new_ticks)
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
        """
        Get data steam of live portfolio updates.
        """
        # Similar implementation to subscribe_ticks, but with different WebSocket endpoint and data parsing logic
        while True:
            ws_url = self.get_portfolio_stream_url()["data"]["authorized_redirect_uri"]
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info("Connected to portfolio stream WebSocket.")

                    while True:
                        logger.info(f"Starting portfolio updater")
                        message = await websocket.recv()
                        data = json.loads(message)
                        logger.info(f"Received portfolio update: {data}")
                        await output.put(data)
            except Exception as e:
                logger.error(f"Error in portfolio stream WebSocket: {e}")
                logger.info("Attempting to reconnect to portfolio stream WebSocket...")
            logger.info("Reconnecting to portfolio stream WebSocket...")
            await asyncio.sleep(5)  # Wait before trying to reconnect

    def get_options_with_expiry(
        self, options: pd.DataFrame | str, is_expired: bool = False, return_df=False
    ) -> list[datetime, pd.DataFrame]:
        """
        Returns nearest weekly or monthly expiration date for given set of `options` and the filtered options with that expiry date.
        """
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
        self, price, instrument_key, quantity, transaction_type="BUY", product="D"
    ) -> float:
        """
        Returns brokerage for a transaction
        """
        url = "https://api.upstox.com/v2/charges/brokerage"

        params = {
            "instrument_token": instrument_key,
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
                f"Error while fetching brokerage |\n{response}",
                stack_info=True,
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
                f"Error while fetching total charges |\n{response}",
                stack_info=True,
            )
            return None

    def get_pnl_report(
        self,
        from_date: datetime = datetime.today().date(),
        to_date: datetime = datetime.today().date(),
        segment="FO",
        financial_year="2526",
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
                f"Error while fetching report metadata |\n{response}",
                stack_info=True,
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
            print(
                "Failed to retrieve data for holidays.",
            )
            return None

    def exchanges_status(self, exchange:Literal["BSE","NSE"]="NSE"):
        url = f"https://api.upstox.com/v2/market/status/{exchange}"

        response = self._make_request("GET", url)

        if response:
            data = response["data"]["status"]
            return data
        else:
            logger.error(
                f"Error while fetching exchange status |\n{response}",
                stack_info=True,
            )
            return response

    def get_positions(self):
        url = "https://api.upstox.com/v2/portfolio/short-term-positions"

        response = self._make_request("GET", url)
        if response:
            return Position.parse(response["data"])
        else:
            logger.error(
                f"Error while fetching positions |\n{response}",
                stack_info=True,
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
                logger.exception("Exception while updating local database")
                return None

    def exitall(self):
        url = "https://api.upstox.com/v2/order/positions/exit"
        data = {}

        try:
            # Send the POST request
            response = self._make_request("POST", url, json=data)
            return response

        except Exception as e:
            # Handle exceptions
            logger.error(
                f"Error while exiting positions |\n{response}",
                stack_info=True,
            )

    def is_exchange_holiday(self, date: datetime,exchange:Literal["NSE","BSE"]="NSE") -> bool:

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
                _HOLIDAY_CACHE[year] = None

        year_holidays = _HOLIDAY_CACHE[year]

        if year_holidays is not None:
            if date_str in year_holidays and exchange in year_holidays[date_str]:
                return True
            return False

        else:
            holiday = self.get_holidays(date_str)
            try:
                cond = any(item["date"] == date_str for item in holiday)
                return cond
            except Exception:
                logger.exception(
                    f"Failed to check whether the date `{date_str}` is a holiday."
                )
                return None

    def get_sandbox_access_token(self):

        def sandbox_authorization():

            print(
                """ Login into https://account.upstox.com/developer/apps#sandbox and create a sandbox app to get the access token.
                                            \n Paste the new sandbox access token here : """
            )
            lines = []
            while True:
                token = input()
                if not token:
                    break
                lines.append(token)
            access_token = "\n".join(lines)
            sandbox_expiry = (datetime.now() + timedelta(days=30)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            payload = {"sandbox_access_token": access_token, "expiry": sandbox_expiry}
            with open(SANDBOX_TOKEN_FILE, "w") as f:
                json.dump(payload, f)
            return access_token

        if SANDBOX_TOKEN_FILE.exists():
            with open(SANDBOX_TOKEN_FILE, "r+") as f:
                data = json.load(f)
                sandbox_access_token = data["sandbox_access_token"]
            if datetime.now() > datetime.strptime(data["expiry"], "%Y-%m-%d %H:%M:%S"):
                print("Sandbox access token expired. Generating new token...")
                sandbox_access_token = sandbox_authorization()
        else:
            logger.info("No sandbox access token found. Generating new token...")
            sandbox_access_token = sandbox_authorization()
        return sandbox_access_token

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
