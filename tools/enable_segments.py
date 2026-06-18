import logging
import sys
from pathlib import Path

# Setup paths based on your existing structure
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from upstox_methods import UpstoxClient

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("MorningReset")

def reset_kill_switch():
    try:
        ustox = UpstoxClient()
        logger.info("Good morning! Re-enabling trading segments for the new session...")
        
        # Assuming 'ENABLE' is the string your method uses to turn segments back on
        ustox.kill_switch(["NSE_FO", "BSE_FO"], action='ENABLE') 
        
        logger.info("✅ Trading segments enabled successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to reset kill switch: {e}")

if __name__ == "__main__":
    reset_kill_switch()