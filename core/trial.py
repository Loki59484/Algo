from pathlib import Path
import sys
import asyncio
# Setting up paths to manage imports
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from core import upstox_func as ustox
tick_buffer = asyncio.Queue(maxsize=10)
expiry_date = ustox.get_expiry(ustox.get_options())
print("Trading options with expiry on: ", expiry_date)
suitable_puts, suitable_calls = ustox.get_suitable(
    expiry_date,
    funds=30000,
)

asyncio.run(ustox.get_live(
    instrument_key=suitable_calls[0]['instrument_key'],
    output=tick_buffer,
))