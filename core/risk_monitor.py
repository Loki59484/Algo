import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime as dt
# Setup paths based on your existing structure
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from upstox_methods import UpstoxClient, logger  # Replace with your actual import
from planner import Planner

# Setup Logging
planner = Planner()
args = planner.setup_cli()
plan = planner.create(args)
today =  dt.today().strftime('%Y-%m-%d')
DAY_TARGET = plan.loc[plan['Date'] == today]['Profit'].item()  # Your profit target in INR
SURPLUS = 5000        # Additional amount to cover charges
TARGET_THRESHOLD = DAY_TARGET + SURPLUS
logger.info(f"Trading will stop at ₹{TARGET_THRESHOLD}")

async def monitor_orders():
    ustox = UpstoxClient()
    logger.info("Initializing 24/7 Upstox Risk Management Stream...")
    data_queue = asyncio.Queue(maxsize=100)
    
    ws_task = asyncio.create_task(ustox.subscribe_portfolio(output=data_queue))

    # Define how much profit you want *per trade* (or use your DAY_TARGET)
    TRADE_PROFIT_TARGET = 500  

    try:
        while True:
            logger.info("Awaiting update")
            update = await data_queue.get()
            
            # 1. Check if the event is a completed BUY order
            if update.get('status') == 'complete' and update.get("transaction_type") == "BUY":
                
                # Extract execution details from the websocket payload
                filled_qty = int(update.get('filled_quantity', 0))
                avg_price = float(update.get('average_price', 0.0))
                instrument = update.get('instrument_token')
                product_type = update.get('product', 'D') # E.g., 'I' for Intraday, 'D' for Delivery
                
                if filled_qty > 0 and avg_price > 0:
                    # 2. Calculate the exact Limit Price
                    required_points = TRADE_PROFIT_TARGET / filled_qty
                    raw_target_price = avg_price + required_points
                    
                    # 3. Round to the nearest 0.05 (Tick Size enforcement)
                    target_price = round(raw_target_price * 20) / 20.0
                    logger.info(f"BUY filled for {update['trading_symbol']} at ₹{avg_price}. "
                                f"Sending SELL limit order at ₹{target_price} to hit ₹{TRADE_PROFIT_TARGET} target.")
                    

                    breakpoint()                    
                    ustox.place_order(
                        instrument_token=instrument,
                        transaction_type="SELL",
                        quantity=filled_qty,
                        order_type="LIMIT",
                        price=target_price,
                        product=product_type 
                    )
            
            # --- YOUR EXISTING PNL / KILL SWITCH LOGIC ---
            elif update.get('status') == 'complete' and update.get("transaction_type") == "SELL":
                # Only check daily PnL after a SELL closes a position
                current_pnl = sum([float(item.get('realised', 0)) for item in ustox.get_positions()])
                logger.info(f"Current Realized PnL: ₹{current_pnl}")

                if current_pnl >= TARGET_THRESHOLD:
                    logger.warning(f"🚨 TARGET REACHED (₹{current_pnl}). ACTIVATING KILL SWITCH! 🚨")
                    resp = ustox.kill_switch(["NSE_FO","BSE_FO"], action='DISABLE') 
                    logger.info(f"{resp}")
                    logger.info("Kill switch activated successfully. Sleeping until tomorrow...")
                    await asyncio.sleep(60 * 60 * 12) 

            else: 
                logger.info(f"Update received for {update.get('trading_symbol', 'Unknown')} | status : {update.get('status')} ")

    except asyncio.CancelledError:
        logger.info("Monitor shutting down.")
        ws_task.cancel() 
        raise
    except Exception as e:
        logger.exception(f"Encountered error: {e}. Reconnecting in 5s...")
        ws_task.cancel() 
        await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(monitor_orders())
    except KeyboardInterrupt:
        logger.info("Risk monitor stopped by user.")