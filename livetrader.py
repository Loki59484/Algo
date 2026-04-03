"""Main file to run the live trading algorithm. This will be used to run the algo in production.
    """
import asyncio
from core import upstox_methods as ustox
from core.datatypes import *
from core.anatomy import Trader
from ui.tui import TradingTUI


def filter_options(options: pd.DataFrame):
    """Filter options based on strike price and option type."""
    spot = options.iloc[0]["underlying_spot_price"]
    calls = [(option['instrument_key'], option['market_data']['oi'],option['market_data']['ltp']) for option in options.call_options if  option['market_data']['oi']> 0]
    sorted_calls = sorted(calls, key=lambda x: x[1], reverse=True)
    puts = [(option['instrument_key'], option['market_data']['oi'],option['market_data']['ltp']) for option in options.put_options if  option['market_data']['oi']> 0]
    sorted_puts = sorted(puts, key=lambda x: x[1], reverse=True)
    return sorted_calls[5], sorted_puts[5]

opt = ustox.get_options()
expiry = ustox.get_expiry(opt)
options = ustox.get_options(dtype="chain", instrument_key="NSE_INDEX|Nifty 50", expiry=expiry)
call,put = filter_options(options)
bucket = Bucket(date=datetime.today().date(),legs={"CE": Instrument(instrument_key=call[0]), "PE": Instrument(instrument_key=put[0])})
trader : Trader = Trader()
trader.buckets.append(bucket)

if __name__ == "__main__":
    app = TradingTUI(trader=trader, simulate=True)
    app.run()
    