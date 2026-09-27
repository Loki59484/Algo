import os
from pathlib import Path
import platformdirs

# ---------------------------------------------------------
# 1. STATIC PACKAGE PATHS (Immutable data shipped with code)
# ---------------------------------------------------------
# config.py sits at src/algo/core/config.py. 
# Resolving parent.parent gets us exactly to src/algo/
PACKAGE_DIR = Path(__file__).resolve().parent.parent
STATIC_CONFIG_DIR = PACKAGE_DIR / "config"
HOLIDAYS_FILE = STATIC_CONFIG_DIR / "holidays.json"

# ---------------------------------------------------------
# 2. DETERMINISTIC WORKSPACE PATHS (Mutable user data & secrets)
# ---------------------------------------------------------
# If running locally, the project root is two levels up from src/algo (i.e., ./)
DEV_ROOT = PACKAGE_DIR.parent.parent

# Detect execution mode by checking for pyproject.toml
if (DEV_ROOT / "pyproject.toml").exists():
    WORKSPACE_DIR = DEV_ROOT
else:
    # If pip-installed in production, route data to standard OS directories
    default_app_dir = platformdirs.user_data_path("Algo", "AlgoTrading")
    # Allow overriding via environment variable for Docker/AWS flexibility
    WORKSPACE_DIR = Path(os.getenv("ALGO_WORKSPACE", default_app_dir))

# Mutable Directories
USER_CONFIG_DIR = WORKSPACE_DIR / "config"
DATA_DIR = WORKSPACE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = WORKSPACE_DIR / "logs"
SECRETS_DIR = WORKSPACE_DIR / ".secrets"

# Mutable Files
ENV_FILE = WORKSPACE_DIR / ".env"
USER_SETTINGS_FILE = USER_CONFIG_DIR / "user_settings.json"

# Bootstrap: Ensure necessary mutable directories exist before anything runs
for directory in [USER_CONFIG_DIR, DATA_DIR, CACHE_DIR, LOGS_DIR, SECRETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
