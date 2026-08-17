import asyncio
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Setup paths based on existing structure
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.methods import ZMQErrorLogger, start_heartbeat, TelegramHandler, TELEGRAM_TOKEN, CHAT_ID
from core.upstox_methods import UpstoxClient, logger
from planner import Planner
from tools.charges_calculator import charges_calculator

# Configure ZMQ Error Logging
zmq_handler = ZMQErrorLogger(component_name="Risk Monitor", port=5567)
zmq_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(zmq_handler)

telegram_handler = TelegramHandler(bot_token=TELEGRAM_TOKEN, chat_id=CHAT_ID)
telegram_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(telegram_handler)

class RiskManager:
    """Monitors live portfolio PnL and enforces account-level kill switches."""

    # Configurable Constants
    COST_BUFFER = 5000.0
    DEFAULT_MAX_LOSS = -5000.0

    def __init__(self):
        self.ustox = UpstoxClient()
        self.planner = Planner()
        
        # Risk Thresholds
        self.day_target: float = 0.0
        self.target_threshold: float = 0.0
        self.max_loss_threshold: float = 0.0

    def initialize_thresholds(self) -> None:
        """Syncs with the Planner to establish today's dynamic profit and loss limits."""
        args = self.planner.setup_cli()
        
        # UPDATE: Extract and pass explicit kwargs based on the new Planner.create() signature
        plan_df = self.planner.create(
            target=args.target, 
            starting=args.starting,
            multiplier=args.multiplier, 
            force=args.force,
            save_state=False  # The monitor only reads the state; it shouldn't overwrite the JSON
        )

        today_str = dt.datetime.today().date().strftime("%Y-%m-%d")

        # 1. Check if today is a valid trading day
        if plan_df is not None and not plan_df.empty and today_str not in plan_df["Date"].values:
            logger.warning(f"No trading scheduled for {today_str}. Market likely closed.")
            logger.info("Exiting Risk Monitor.")
            sys.exit(0)

        # 2. Establish Profit Target
        self.day_target = self.planner.chronological_target
        charges = charges_calculator()
        self.target_threshold = self.day_target + charges

        # 3. Establish Max Loss (Based on previous day's profit)
        if plan_df is not None and not plan_df.empty:
            today_idx_list = plan_df.index[plan_df["Date"] == today_str].tolist()
            if today_idx_list and today_idx_list[0] > 0:
                prev_day_profit = float(plan_df.iloc[today_idx_list[0] - 1]["Profit"])
                self.max_loss_threshold = -abs(prev_day_profit) 
            else:
                self.max_loss_threshold = self.DEFAULT_MAX_LOSS
        else:
            self.max_loss_threshold = self.DEFAULT_MAX_LOSS 

        logger.info(f"Target acquired from Planner: ₹{self.day_target:,.2f}")
        logger.info(f"Current charges : ₹{charges:,.2f}")
        logger.info(f"Trading will STOP for profit at: ₹{self.target_threshold:,.2f} (Includes ₹{self.COST_BUFFER:,.2f} buffer)")
        logger.info(f"Trading will STOP for loss at: ₹{self.max_loss_threshold:,.2f} (Previous day's target)")

    async def start_monitoring(self) -> None:
        """Main entry point for the async WebSocket monitor with robust reconnection."""
        logger.info("Initializing 24/7 Upstox Risk Management Stream...")
        
        while True:
            data_queue = asyncio.Queue(maxsize=100)
            ws_task = asyncio.create_task(self.ustox.subscribe_portfolio(output=data_queue))

            try:
                await self._process_stream(data_queue)
            except asyncio.CancelledError:
                logger.info("Monitor shutting down.")
                ws_task.cancel()
                raise
            except Exception as e:
                logger.exception(f"Stream encountered error: {e}. Reconnecting in 5s...")
                ws_task.cancel()
                await asyncio.sleep(5) 

    async def _process_stream(self, queue: asyncio.Queue) -> None:
        """Consumes updates from the WebSocket queue."""
        while True:
            update: Dict[str, Any] = await queue.get()
            
            status = update.get("status")
            txn_type = update.get("transaction_type")
            
            # ONLY evaluate PnL when a SELL order completes (position closed by livetrader)
            if status == "complete" and txn_type == "SELL":
                await self._evaluate_pnl_and_kill()
            elif status == "complete":
                logger.info(f"Trade filled for {update.get('trading_symbol', 'Unknown')} | side: {txn_type}")

    async def _evaluate_pnl_and_kill(self) -> None:
        """Evaluates daily realized PnL and triggers kill switches if thresholds are breached."""
        # Calculate current net realized PnL
        positions = self.ustox.get_positions()
        
        # Guard against None if API fails
        if not positions:
            return
            
        current_pnl = sum(float(item.get("realised", 0.0)) for item in positions)
        charges = charges_calculator()
        self.target_threshold = self.day_target + charges
        self.max_loss_threshold = self.DEFAULT_MAX_LOSS - charges
        
        logger.info(f"Current Realized PnL: ₹{current_pnl:,.2f} | Target: ₹{self.target_threshold:,.2f} | Max Loss: ₹{self.max_loss_threshold:,.2f}")

        # Check thresholds
        if current_pnl >= self.target_threshold:
            await self._trigger_kill_switch(current_pnl, "PROFIT TARGET REACHED")
            
        elif current_pnl <= self.max_loss_threshold:
            await self._trigger_kill_switch(current_pnl, "MAX LOSS REACHED - EMERGENCY")

    async def _trigger_kill_switch(self, pnl: float, reason: str) -> None:
        """Cancels orders, exits positions, disables trading, and sleeps."""
        logger.warning(f"🚨 {reason} (₹{pnl:,.2f})")
        
        # 1. Cancel all open / pending orders
        logger.info("Canceling all open orders...")
        try:
            self.ustox.cancel_multi_orders()
        except Exception as e:
            logger.error(f"Failed to cancel open orders: {e}")

        # 2. Exit all open positions
        logger.info("Exiting all open positions...")
        try:
            self.ustox.exitall()
        except Exception as e:
            logger.error(f"Failed to exit positions: {e}")

        # 3. Wait for the broker's backend database to clear the ledger
        logger.info("Waiting 4 seconds for Upstox ledger to settle...")
        await asyncio.sleep(4)

        # 4. Activate the Kill Switch
        logger.warning("ACTIVATING SEGMENT KILL SWITCH!")
        try:
            resp = self.ustox.kill_switch(["NSE_FO", "BSE_FO", "NSE_EQ", "BSE_EQ"], action="DISABLE")
            
            # Verify success
            if isinstance(resp, dict) and resp.get("status") == "success":
                logger.info("Kill switch confirmed successful by broker.")
                # tg_bot.send_alert(f"✅ Segment Kill Switch Activated! Final PnL: ₹{pnl:,.2f}")
            else:
                logger.error(f"Kill switch may have failed! Broker response: {resp}")
                
        except Exception as e:
            logger.error(f"Fatal error activating kill switch: {e}")

        # 5. Exit the daemon
        logger.info("Risk Engine halted for the day. Exiting process.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        start_heartbeat(component_name="Risk Monitor", port=5557)
        
        manager = RiskManager()
        manager.initialize_thresholds()
        
        asyncio.run(manager.start_monitoring())
        
    except KeyboardInterrupt:
        logger.info("Risk monitor stopped by user.")
    except Exception as e:
        logger.exception(f"Fatal error in Risk Monitor execution: {e}")