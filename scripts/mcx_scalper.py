import asyncio
import logging
from datetime import datetime
from core.upstox_methods import UpstoxClient
from core.datatypes import Tick, Order

logger = logging.getLogger("MCXScalper")

class MCXLimitScalper:
    def __init__(self, instrument_key: str, quantity: int, target_pts: float, stop_pts: float):
        self.ustox = UpstoxClient()
        self.instrument_key = instrument_key
        self.quantity = quantity
        self.target_pts = target_pts
        self.stop_pts = stop_pts
        self.in_position = False
        self.active_order_id = None
        self.tick_buffer = []

    def check_entry_signal(self, tick: Tick) -> bool:
        # Example Micro-Scalp Condition: Upward tick momentum over 5 ticks
        if not tick.ltpc or not tick.depth:
            return False
        self.tick_buffer.append(tick.ltpc.ltp)
        if len(self.tick_buffer) > 5:
            self.tick_buffer.pop(0)
            return all(x < y for x, y in zip(self.tick_buffer, self.tick_buffer[1:]))
        return False

    async def execute_limit_order(self, side: str, price: float) -> str:
        """Submits a LIMIT order via Upstox API."""
        order_payload = {
            "quantity": self.quantity,
            "product": "I",  # Intraday
            "validity": "DAY",
            "price": round(price, 2),
            "instrument_token": self.instrument_key,
            "order_type": "LIMIT",
            "transaction_type": side.upper(),
            "disclosed_quantity": 0,
            "trigger_price": 0.0,
            "is_amo": False,
        }
        response = await asyncio.to_thread(self.ustox.place_order, **order_payload)
        return response.get("data", {}).get("order_id")

    async def run_scalp_cycle(self, best_bid: float, best_ask: float):
        self.in_position = True
        entry_price = best_bid
        logger.info(f"Triggering BUY limit entry at {entry_price}")
        entry_id = await self.execute_limit_order("BUY", entry_price)

        # 3-second timeout for limit fill, otherwise cancel
        await asyncio.sleep(3)
        order_status = await asyncio.to_thread(self.ustox.get_order_details, entry_id)
        if order_status.get("status") != "complete":
            await asyncio.to_thread(self.ustox.cancel_order, entry_id)
            self.in_position = False
            return

        # Immediate target limit placement
        target_price = entry_price + self.target_pts
        logger.info(f"Filled. Placing SELL limit scalp at {target_price}")
        exit_id = await self.execute_limit_order("SELL", target_price)

    async def process_ticks(self, data_queue: asyncio.Queue):
        while True:
            tick_data = await data_queue.get()
            tick = Tick.parse_tick(
                key=self.instrument_key,
                feed=tick_data,
                timestamp=str(int(datetime.now().timestamp() * 1000)),
                market_status=True,
            )
            if not self.in_position and self.check_entry_signal(tick):
                best_bid = tick.depth[0].get("bidPrice", tick.ltpc.ltp)
                best_ask = tick.depth[0].get("askPrice", tick.ltpc.ltp)
                asyncio.create_task(self.run_scalp_cycle(best_bid, best_ask))
            data_queue.task_done()
