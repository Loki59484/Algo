"""Main file to run the live trading algorithm. This will be used to run the algo in production.
    """
from core.upstox_methods import UpstoxClient 
from core.datatypes import *
from core.anatomy import Trader
from ui.tui import TradingTUI
from core.methods import filter_options
ustox = UpstoxClient()
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
    