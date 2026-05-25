"""
Configuration file for the GUI. 
"""
import asyncio
import threading

sandbox_orders : bool = True # Boolean to check whether to use sandox orders
stopevent = threading.Event() # Event to stop the program
market_close = asyncio.Event() # Event set once the market closes
tick_ready = threading.Event() # Event to signal readiness of ticks
looprun = asyncio.Event() 
log_queue = None
latency: float = 0.5
current_index = 0
previous_target = 1157611
target_profit = 77
prev_days = 50
prev_starting = 452
pbar = 0
