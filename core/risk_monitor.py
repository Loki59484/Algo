import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime as dt
# Setup paths based on your existing structure
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from upstox_methods import UpstoxClient  # Replace with your actual import
from planner import Planner

# Setup Logging
logger = logging.getLogger(__name__)
planner = Planner()
args = planner.setup_cli()
plan = planner.create(args)
today =  dt.today().strftime('%Y-%m-%d')
DAY_TARGET = plan.loc[plan['Date'] == today]['Profit'].item()  # Your profit target in INR
SURPLUS = 5000        # Additional amount to cover charges
TARGET_THRESHOLD = DAY_TARGET + SURPLUS

async def monitor_orders():
    ustox = UpstoxClient()
    logger.info("Initializing 24/7 Upstox Risk Management Stream...")
    data_queue = asyncio.Queue(maxsize=100)
    
    # 1. Start the websocket listener in the background so it can fill the queue
    ws_task = asyncio.create_task(ustox.subscribe_portfolio(output=data_queue))

    try:
        while True:
            # 2. Wait for the next update to arrive in the queue
            update = await data_queue.get()
            
            # Check if the event is a completed order (position closed/opened)
            if True:# getattr(update, 'status', '').lower() == 'complete':
                logger.info(f"Order filled for {update.trading_symbol}. Checking Daily PnL...")
                
                # Fetch your current total realized MTM (Profit/Loss)
                # Ensure get_funds() is actually an async method returning your PnL
                current_pnl = await ustox.get_funds()
                # breakpoint() # Keep in mind breakpoint() will pause the background server
                
                logger.info(f"Current Realized PnL: ₹{current_pnl}")

                # Check if we hit the limit
                if current_pnl >= TARGET_THRESHOLD:
                    logger.warning(f"🚨 TARGET REACHED (₹{current_pnl}). ACTIVATING KILL SWITCH! 🚨")
                    
                    # Fire the Kill Switch
                    ustox.kill_switch(["NSE_FO","BSE_FO"], action='DISABLE') 
                    
                    logger.info("Kill switch activated successfully. Sleeping until tomorrow...")
                    
                    # Sleep for 12 hours to prevent rapid re-triggering today
                    await asyncio.sleep(60 * 60 * 12) 

    except asyncio.CancelledError:
        logger.info("Monitor shutting down.")
        ws_task.cancel() # Clean up the background task
    except Exception as e:
        logger.error(f"Encountered error: {e}. Reconnecting in 5s...")
        ws_task.cancel() # Clean up before attempting reconnect (if you add a retry loop later)
        await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(monitor_orders())
    except KeyboardInterrupt:
        logger.info("Risk monitor stopped by user.")