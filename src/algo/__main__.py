"""
Main file to fire the trading engine.

Simulation or Live modes can be chosen by passing respective arguments.

The connection to live market, the trading and the UI are run on different async loops
"""

# Import modules
from pathlib import Path
import argparse
import logging
import asyncio
import sys

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core import anatomy as ana
from ui import tui  # ,gui


def setup_cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="Launch Trading or Simulation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    liveparser = subparsers.add_parser("live", help="Start live trading engine")
    simparser = subparsers.add_parser(
        "sim", help="Start Trading engine in simulation mode"
    )
    mode_group = simparser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "-b",
        "--bulk",
        nargs="+",
        metavar="PATH",
        help="(Default Mode) Initiates engine in bulk simulation mode for given FILES or files inside the given DIRECTORY.",
    )
    mode_group.add_argument(
        "-t",
        "--tickwise",
        type=str,
        metavar="FILE",
        help="Initiates engine in a chronological tickwise mode for the given FILE.",
    )
    ui_group = parser.add_mutually_exclusive_group(required=False)
    ui_group.add_argument(
        "--gui",
        action="store_true",
        help="Lauch engine with Graphical User Interface"
    )
    ui_group.add_argument(
        "--tui",
        action="store_true",
        help="Lauch engine with Terminal User Interface"        
    )
    ui_group.add_argument(
        "--headless",
        action="store_true",
        help="Lauch engine in headless mode (DEFAULT)"        
    )
    return parser.parse_args()

def main():
    args = setup_cli()
    engine = None

    if args is None:
        logger.critical(f"{__name__} requires at least one argument") 
        return

    if args.command == "live":
        logger.info("BOOTING ENGINE IN LIVE MODE.")

    elif args.command == "sim":
        logger.info("BOOTING ENGINE IN LIVE MODE.")
        asyncio.run(ana.SimfeedStreamer())  
        

if __name__ == "__main__":
    main()
