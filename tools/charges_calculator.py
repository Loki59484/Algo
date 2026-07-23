from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.upstox_methods import UpstoxClient# Replace with your actual import
from core.methods import calculate_trade_charges

def charges_calculator():
    ustox = UpstoxClient()
    order_book = ustox.get_order_book()
    completed = [item for item in order_book if item['status']=='complete']
    buys = [item for item in completed if item['transaction_type']=='BUY']
    sells = [item for item in completed if item['transaction_type']=='SELL']
    sorted_sells = sorted(sells, key=lambda x: x["order_timestamp"])
    sorted_buys = sorted(buys, key=lambda x: x["order_timestamp"])
    charges = 0
    for buy,sell in zip(sorted_buys,sorted_sells):
        charges+=calculate_trade_charges(buy['price'], sell['price'],buy['filled_quantity'])
    return int(charges)


if __name__ == "__main__":
    total_charges = charges_calculator()
    print(f"Total charges for all completed orders: {total_charges}")