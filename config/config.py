# configuration file for anatomy.py and gui.py

import asyncio
import threading
from core import upstox_func as ustox
from concurrent.futures import ThreadPoolExecutor
import datetime as dt

sandbox_orders = True
stopevent = threading.Event()
market_close = asyncio.Event()
tick_ready = threading.Event()
looprun = asyncio.Event()
log_queue = None
latency: float = 0.5
current_index = 0
previous_target = 1157611
target_profit = 77
prev_days = 50
prev_starting = 452
pbar = 0
