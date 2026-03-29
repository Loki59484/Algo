# Module for Upstox
from google.protobuf.json_format import MessageToDict
from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from collections import deque
from functools import wraps
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

""" NOTE : All functions are defined to be used with Upstox API. Refer to the official documentation for more details on the API and its usage."""

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

# STATIC VARIABLES
if not ENV_PATH.exists():
    os.env
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("UPSTOX_API_KEY")
API_SECRET = os.getenv("UPSTOX_API_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
MOBILE_NUM = os.getenv("MOBILE_NUMBER")

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

def with_retry_and_limits(max_retries=3):
    """
    Decorator that enforces the rate limit BEFORE the call, 
    and handles HTTP 429 Too Many Requests AFTER the call.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff_time = 2.0  # Start with a 2-second timeout
            
            while retries <= max_retries:
                # 1. Ask the Limiter for permission to fire
                GLOBAL_UPSTOX_LIMITER.wait_for_token()
                
                try:
                    # 2. Execute the actual API call
                    response = func(*args, **kwargs)
                    
                    # 3. Check for the specific Rate Limit error code
                    if response.status_code == 429:
                        logger.error(f"HTTP 429 Hit! Backing off for {backoff_time}s...")
                        time.sleep(backoff_time)
                        retries += 1
                        backoff_time *= 2  # Exponential backoff (2s, 4s, 8s)
                        continue
                        
                    response.raise_for_status() # Raise other standard errors (404, 500)
                    return response
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error: {e}")
                    retries += 1
                    time.sleep(backoff_time)
            
            logger.critical(f"Failed to fetch data after {max_retries} retries.")
            return None # Or raise an exception depending on your architecture
            
        return wrapper
    return decorator

def authorization():  # Log in to Upstox API
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
    server_ready_event.wait(timeout=10)  # Wait up to 10 seconds for server to be ready
    if not server_ready_event.is_set():
        logger.critical("Error: Local server did not start in time. Exiting.")
        exit(1)
    result_container = []
    error_container = []

    def _run_playwright():
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
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

            token_response = requests.post(
                token_url, headers=token_headers, data=token_data
            )

            if token_response.status_code == 200:
                logger.info("Login successful!")
                print("#" * 50, "LOGIN SUCCESSFUL", "#" * 50)
                data = token_response.json()
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

            else:
                logger.critical(
                    f"Login Failed! Status : {token_response.status_code} \n {token_response.json()}"
                )
                raise (f"Login failed.\n {token_response.json()}")
        except Exception as e:
            logger.exception("Error while getting access token. \n{e}")

    worker_thread = threading.Thread(target=_run_playwright)
    worker_thread.start()
    worker_thread.join()


def get_access_token():
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, "r+") as f:
            data = json.load(f)
        if datetime.now() > datetime.strptime(data["expiry"], "%Y-%m-%d %H:%M:%S"):
            print("Access token expired. Generating new token...")
            access_token = authorization()
            update_database()
        else:
            access_token = data["access_token"]
    else:
        logger.info("No access token found. Generating new token...")
        print("No access token found. Generating new token...")
        access_token = authorization()
        update_database()
    return access_token


def get_funds():  # Getting funds available
    funds_url = "https://api.upstox.com/v2/user/get-funds-and-margin"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    response = requests.get(funds_url, headers=headers)

    if response.status_code == 200:
        funds = response.json()["data"]
        return funds
    else:
        logger.error(
            f"Failed to get funds info! Status : {response.status_code} | {response.json()}"
        )

        return 0


def get_expired_instruments(
    instrument_key="", expiry_date="", underlying="NSE_INDEX|Nifty 50", keys_only=True
):  # Getting expired instruments for a stock/index
    """
    Returns expired instruments for specified stock/index instrument for the given expiry date.
    If instrument key provided, it returns data containing that instrument key only in the form of a dataframe.
    If not provided, it returns data for all expired instruments for the given expiry date in the form of a dataframe.
    """
    url = f"https://api.upstox.com/v2/expired-instruments/option/contract?instrument_key={underlying}&expiry_date={expiry_date}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
        "Accept": "application/json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = pd.DataFrame(response.json()["data"])
        if instrument_key == "":
            return data
        keys = data["instrument_key"].loc[
            data["instrument_key"].str.contains(instrument_key, regex=False)
        ]
        keys = keys if keys.size > 0 else None
        return keys
    else:
        logger.error(
            f"Failed to retrieve data for expired instruments : {response.status_code} \n{response.json()}"
        )

@with_retry_and_limits(max_retries=3)
def _safe_api_request(url: str, headers: dict, payload: dict = None):
    """
    An isolated HTTP fetcher. The decorator wraps THIS function, 
    so it correctly receives and evaluates the requests.Response object.
    """
    return requests.get(url, headers=headers, data=payload)

def get_historical(
    dtype="historical",
    instrument_key="NSE_INDEX|Nifty 50",
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
                        get_expiry(options="NSE_INDEX|Nifty 50", is_expired=True)[::-1]
                    )
                    for date in expiry_date:
                        expired_key = get_expired_instruments(
                            instrument_key=instrument_key, expiry_date=date
                        )
                        if expired_key is not None:
                            break
                else:
                    if expired_key is None:
                        expired_key = get_expired_instruments(
                            instrument_key=instrument_key, expiry_date=expiry_date
                        )

                url = f"https://api.upstox.com/v2/expired-instruments/historical-candle/{expired_key}/1minute/{to_date}/{from_date}"
            else:
                url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
        elif dtype == "intraday":
            url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
        payload = {}
        headers = {
            "Authorization": f"Bearer {get_access_token()}",
            "Accept": "application/json",
        }

        response = _safe_api_request(url=url, headers=headers, payload=payload)
        if response is not None and response.status_code == 200:
            data = response.json()["data"]
            data["candles"].reverse()
            df = pd.DataFrame(
                data["candles"],
                columns=["timestamp", "open", "high", "low", "close", "vol", "oi"],
            )
            df = df if df is not None else None
            if df.empty:
                logger.error(
                    f"Empty Dataframe recieved for key : {instrument_key} | from_date : {from_date} | to_date : {to_date} ",
                    stack_info=True,
                )
            return df
        else:
            error_msg = response.json() if response is not None else "API Request Failed/Timeout"
            status = response.status_code if response is not None else "N/A"
            
            logger.error(error_msg)
            logger.error(
                f"Status code : {status} | Empty Dataframe returned for key : {instrument_key} | from_date : {from_date} | to_date : {to_date} ",
                stack_info=True,
            )
            return None
    else:
        logger.warning(f"Invalid Value for 'dtype' {dtype}")
        raise TypeError(f"Possible values for 'dtype' : {valid}")


def get_options(expiry="", dtype="contract", instrument_key="NSE_INDEX|Nifty 50"):
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
        print(f"Getting Options for {instrument_key}")
        params = {"instrument_key": f"{instrument_key}", "expiry_date": f"{expiry}"}
        payload = {}
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {get_access_token()}",
        }
        response = requests.get(url, params=params, headers=headers, data=payload)
        if response.status_code == 200:
            data = response.json()["data"]
            return data
        else:
            logger.error(f"{response.json()}")
        if expiry == "":
            logger.info("Expiry not specified. All available instruments retrieved!")
    except ValueError as v:
        logger.error(v)


def get_suitable(
    expiry, funds=10000, instrument_key="NSE_INDEX|Nifty 50", ui=None
):  # Getting suitable puts/calls
    """
    Returns calls and puts suitable for trade based on defined conditions
    """
    suitable_calls = []
    suitable_puts = []
    options = get_options(dtype="chain", instrument_key=instrument_key, expiry=expiry)
    """
    DELTA : Change in premium per change in spot price : PUT < -0.5   CALL > 0.5
    GAMMA : Change in Delta per change in spot price : 0.0003 < G < 0.0005
    -----IV and VEGA are used to trade volatility : I don't know if usefull for intraday-----
    IV : Predicts fluctuation in the market, high iv high premium and vice versa : 20% - 30%
    VEGA : Change in premium for 1% change in IV
    POP : Probability of option to be ITM till expiry : I don't know how to relate it to intraday trading
    theta : time decay, amount by which premium decays per day if nothing increases it : Not sure how to use it for intraday

    Other factors like spread etc still need to be included
    """
    spot = options[0]["underlying_spot_price"]

    for item in options:
        try:
            ask = item["call_options"]["market_data"]["ask_price"]
            bid = item["call_options"]["market_data"]["bid_price"]
            spread = ask - bid
            delta = item["call_options"]["option_greeks"]["delta"]
            gamma = item["call_options"]["option_greeks"]["gamma"]
            if all([spread < 2, abs(item["strike_price"] - spot) < 100]):
                item["call_options"]["pcr"] = item["pcr"]
                item["call_options"]["strike_price"] = item["strike_price"]
                item["call_options"]["type"] = "Call"
                suitable_calls.append(item["call_options"])

            ask = item["put_options"]["market_data"]["ask_price"]
            bid = item["put_options"]["market_data"]["bid_price"]
            spread = ask - bid
            delta = item["put_options"]["option_greeks"]["delta"]
            gamma = item["put_options"]["option_greeks"]["gamma"]
            if all([spread < 2, abs(item["strike_price"] - spot) < 100]):
                item["put_options"]["pcr"] = item["pcr"]
                item["put_options"]["strike_price"] = item["strike_price"]
                item["put_options"]["type"] = "Put"
                suitable_puts.append(item["put_options"])
        except KeyError:
            pass
    return suitable_calls, suitable_puts


# Getting market quote
def get_marketquote(instrument_key="NSE_INDEX|Nifty 50"):
    """
    Returns market quote for given instrument(s) [upto 500 at a time]
    """
    url = "https://api.upstox.com/v2/market-quote/quotes"
    payload = {}
    params = {"instrument_key": instrument_key}
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    respone = requests.request("GET", url, headers=headers, data=payload, params=params)
    if respone.status_code == 200:
        data = respone.json()["data"]
        return


def place_order(
    instrument_token,
    transaction_type,
    price=0,
    quantity="75",
    product="D",
    validity="DAY",
    tag="string",
    order_type="MARKET",
    trigger_price=0,
    is_amo=False,
    sandbox=False,
):
    """
    Place Buy or Sell order at market or sandbox
    """

    if sandbox == True:
        url = "https://api-sandbox.upstox.com/v3/order/place"
        headers = {
            "Authorization": f"Bearer {get_sandbox_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    else:
        url = "https://api-hft.upstox.com/v3/order/place"
        headers = {
            "Authorization": f"Bearer {get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    payload = json.dumps(
        {
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
    )

    response = requests.post(url, headers=headers, data=payload)
    if not response.status_code == 200:
        logger.error(response.json())
    return response


def get_order_details(order_id=""):
    """
    Get details of the specified order.
    """

    url = "https://api.upstox.com/v2/order/details"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }

    params = {"order_id": f"{order_id}"}

    response = requests.get(url, headers=headers, params=params)
    return response.json()["data"] if response.status_code == 200 else None


def cancel(order_id, sandbox=False):
    """
    Cancel the specified order.
    """
    if sandbox == True:
        url = "https://api-sandbox.upstox.com/v3/order/cancel"
        headers = {
            "Authorization": f"Bearer {get_sandbox_access_token()}",
            "Accept": "application/json",
        }
    else:
        url = "https://api-hft.upstox.com/v3/order/cancel"
        headers = {
            "Authorization": f"Bearer {get_access_token()}",
            "Accept": "application/json",
        }
    payload = {"order_id": order_id}
    response = requests.request("DELETE", url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json()["data"]
    else:
        logger.error(response.json(), stack_info=True)


def get_market_data_feed_authorize_v3(access_token):
    """Get authorization for market data feed."""
    access_token = access_token
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    api_response = requests.get(url=url, headers=headers)
    return api_response.json()


def decode_protobuf(buffer):
    """Decode protobuf message."""
    feed_response = pb.FeedResponse()
    feed_response.ParseFromString(buffer)
    return feed_response


async def get_live(
    output: asyncio.Queue,
    instrument_key="NSE_INDEX|Nifty 50",
    mode="full_d30",
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
    response = get_market_data_feed_authorize_v3(get_access_token())
    # Connect to the WebSocket with SSL context
    async with websockets.connect(
        response["data"]["authorized_redirect_uri"], ssl=ssl_context
    ) as websocket:
        print("Connection established")

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
            decoded_data = decode_protobuf(message)
            # Convert the decoded data to a dictionary
            data_dict = MessageToDict(decoded_data)
            if "type" in data_dict.keys() and data_dict["type"] == "market_info":
                logger.info(data_dict)
            else:
                try:
                    if not output is None:
                        ts = data_dict.get("currentTs", "0")
                        new_ticks = {
                            key: Tick.parse_tick(
                                key, feed_data["fullFeed"], timestamp=ts
                            )
                            for key, feed_data in data_dict.get("feeds", {}).items()
                            if "fullFeed" in feed_data
                        }
                        if output.full():
                            try:
                                output.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        output.put_nowait(new_ticks)
                    else:
                        logger.error("Failed to put data into the output queue.")
                except KeyboardInterrupt:
                    break
                except Exception:
                    logger.exception(f"Error while parsing or queing live data.")


def get_expiry(options=None, is_expired=False):
    """
    Returns nearest weekly or monthly expiration date for given set of Options.
    """
    if options is None and not is_expired:
        raise ValueError("'options' cannot be None if 'is_expired' is False")
    if is_expired:
        if not isinstance(options, str):
            raise TypeError("String expected for 'options'")
        url = f"https://api.upstox.com/v2/expired-instruments/expiries?instrument_key=NSE_INDEX|Nifty 50"
        headers = {
            "Authorization": f"Bearer {get_access_token()}",
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()["data"]
        else:
            logger.error(
                f"Error while fetching expiries | Status code : {response.status_code} |\n{response.text}"
            )
    else:
        print("Expiries of unexpired instruments.")
        dump = []
        for option in options:
            expiry = datetime.strptime(option["expiry"], "%Y-%m-%d")
            dump.append(expiry)
        dump = list(set(dump))
        dump.sort()
        if datetime.now() > dump[0]:
            return date.strftime(dump[1], "%Y-%m-%d")
        else:
            return date.strftime(dump[0], "%Y-%m-%d")


def get_brokerage(
    price, instrument_key, quantity, transaction_type="BUY", product="D"
) -> float:
    """
    Returns brokerage for a transaction
    """
    url = "https://api.upstox.com/v2/charges/brokerage"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }

    params = {
        "instrument_token": instrument_key,
        "quantity": quantity,
        "product": product,
        "transaction_type": transaction_type,
        "price": price,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()["data"]["charges"]["total"]
        return data
    else:
        logger.error(
            f"Error while fetching brokerage | Status code : {response.status_code} |\n{response.text}",
            stack_info=True,
        )
        return None


def get_total_charges(
    from_date: datetime = datetime.today().date(),
    to_date: datetime = datetime.today().date(),
    segment="FO",
    financial_year="2526",
):
    url = "https://api.upstox.com/v2/trade/profit-loss/charges"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    params = {
        "from_date": from_date.strftime("%d-%m-%Y"),
        "to_date": to_date.strftime("%d-%m-%Y"),
        "segment": segment,
        "financial_year": financial_year,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()["data"]["charges_breakdown"]["total"]
        return data
    else:
        logger.error(
            f"Error while fetching total charges | Status code : {response.status_code} |\n{response.text}",
            stack_info=True,
        )
        return None


def get_pnl_report(
    from_date: datetime = datetime.today().date(),
    to_date: datetime = datetime.today().date(),
    segment="FO",
    financial_year="2526",
    pagenumber=1,
    pagesize=20,
):
    url = "https://api.upstox.com/v2/trade/profit-loss/data"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    params = {
        "from_date": from_date.strftime("%d-%m-%Y"),
        "to_date": to_date.strftime("%d-%m-%Y"),
        "segment": segment,
        "financial_year": financial_year,
        "page_number": pagenumber,
        "page_size": pagesize,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()["data"]
        return data
    else:
        logger.error(
            f"Error while fetching Profit and Loss report | Status code : {response.status_code} |\n{response.text}",
            stack_info=True,
        )
        return None


def get_report_metadata(
    from_date: datetime = datetime.today().date(),
    to_date: datetime = datetime.today().date(),
    segment="FO",
    financial_year="2526",
):
    url = "https://api.upstox.com/v2/trade/profit-loss/metadata"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    params = {
        "from_date": from_date.strftime("%d-%m-%Y"),
        "to_date": to_date.strftime("%d-%m-%Y"),
        "segment": segment,
        "financial_year": financial_year,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()["data"]
        return data
    else:
        logger.error(
            f"Error while fetching report metadata | Status code : {response.status_code} |\n{response.text}",
            stack_info=True,
        )
        return None


def get_charges_report(
    from_date: datetime = datetime.today().date(),
    to_date: datetime = datetime.today().date(),
    segment="FO",
    financial_year="2526",
):

    url = "https://api.upstox.com/v2/trade/profit-loss/charges"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }

    params = {
        "from_date": from_date.strftime("%d-%m-%Y"),
        "to_date": to_date.strftime("%d-%m-%Y"),
        "segment": segment,
        "financial_year": financial_year,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()["data"]["charges_breakdown"]["total"]
        return data
    else:
        print("Failed to get trade charges!")
        print(response.json())
        return None


def get_holidays(date=""):
    url = f"https://api.upstox.com/v2/market/holidays/{date}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()["data"]
        return data
    else:
        print(
            "Failed to retrieve data for holidays. Status code:", response.status_code
        )
        return None


def exchanges_status(exchange="NSE"):
    url = f"https://api.upstox.com/v2/market/status/{exchange}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()["data"]["status"]
        return data
    else:
        logger.error(
            f"Error while fetching exchange status | Status code : {response.status_code} |\n{response.text}",
            stack_info=True,
        )
        return response.json()


def get_positions():
    url = "https://api.upstox.com/v2/portfolio/short-term-positions"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["data"]
    else:
        logger.error(
            f"Error while fetching positions | Status code : {response.status_code} |\n{response.text}",
            stack_info=True,
        )


def update_database():
    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with gzip.open(io.BytesIO(response.content), "rb") as gzfile:
            decompressed_file = gzfile.read()
            jsonstr = decompressed_file.decode(encoding="utf-8")
            json_data = json.loads(jsonstr)
            df = pd.DataFrame(json_data)
            df.to_parquet(DATA_DIR / "NSE_DATABASE.parquet", engine="pyarrow")
            logger.info("Local database updated successfully.")
            return
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Status code : {response.status_code} |\n{response.text}", stack_info=True
        )
        return None


def exitall():
    url = "https://api.upstox.com/v2/order/positions/exit"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }

    data = {}

    try:
        # Send the POST request
        response = requests.post(url, json=data, headers=headers)

        # Print the response status code and body
        print("Response Code:", response.status_code)
        print("Response Body:", response.json())

    except Exception as e:
        # Handle exceptions
        logger.error(
            f"Error while exiting positions | Status code : {response.status_code} |\n{response.text}",
            stack_info=True,
        )
        return 1


def is_nse_holiday(date: datetime) -> bool:

    if date.weekday() >= 5:
        return True
        
    year = date.year
    date_str = date.strftime("%Y-%m-%d")
    
    if year not in _HOLIDAY_CACHE:
        path: Path = CONFIG_DIR / "holidays" / f"{year}.json"
        
        if path.exists():
            with open(path, 'r') as file:
                raw_data = json.load(file)
                
            _HOLIDAY_CACHE[year] = {
                item["date"]: str(item.get("closed_exchanges", "")) 
                for item in raw_data
            }
        else:
            _HOLIDAY_CACHE[year] = None 
            
    year_holidays = _HOLIDAY_CACHE[year]
    
    if year_holidays is not None:
        if date_str in year_holidays and "NSE" in year_holidays[date_str]:
            return True
        return False
        
    else:
        holiday = get_holidays(date_str)
        try:
            cond = any(item['date']==date_str for item in holiday)
            return cond
        except Exception:
            logger.exception(f"Failed to check whether the date `{date_str}` is a holiday.")
            return None    


def get_sandbox_access_token():
    def sandbox_authorization():
        access_token = input(
            """ Login into https://account.upstox.com/developer/apps#sandbox and create a sandbox app to get the access token.
                                        \n Paste the new sandbox access token here : """
        )
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

def generate_holidays():
    holidays_data = pd.read_json(CONFIG_DIR / "holidays.json")
    holidays_data['Gdate'] = to_ist(holidays_data["date"]).dt.year
    holidays_data['date']=holidays_data['date'].dt.strftime("%Y-%m-%d")
    grouped = holidays_data.groupby('Gdate')
    for year,df in grouped:
        path:Path=CONFIG_DIR/'holidays'/f"{year}.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        df = df.reset_index(drop=True)
        df.to_json(path,orient='records')





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
                    now = time.time() # Update 'now' after sleeping!

            # 3. Check 1-Minute Limit (Buffer at 480 instead of 500)
            if len(self.min_history) >= 480:
                sleep_time = 60.0 - (now - self.min_history[0])
                logger.warning(f"Minute Rate Limit Approaching. Thread pausing for {sleep_time:.2f}s")
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            # 4. Check 30-Minute Limit (Buffer at 1950)
            if len(self.half_hr_history) >= 1950:
                sleep_time = 1800.0 - (now - self.half_hr_history[0])
                logger.warning(f"30-Minute Rate Limit Approaching. Pausing for {sleep_time:.2f}s")
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            # 5. Log the new request timestamp into all buckets
            self.sec_history.append(now)
            self.min_history.append(now)
            self.half_hr_history.append(now)

# Instantiate a SINGLE global instance of this lock for the whole project
GLOBAL_UPSTOX_LIMITER = UpstoxRateLimiter()
get_access_token()