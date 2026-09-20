import pandas as pd
import requests
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class InstrumentMaster:
    """Singleton registry for all Upstox exchange instruments."""
    
    _instance = None
    _cache_file = Path("data/instrument_master.parquet")
    _url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InstrumentMaster, cls).__new__(cls)
            cls._instance._registry = {}
            cls._instance._load_master()
        return cls._instance

    def _load_master(self):
        """Loads the registry from cache or downloads it if missing/stale."""
        if self._cache_file.exists():
            # Check if cache is from today to ensure accurate expirations
            mtime = datetime.fromtimestamp(self._cache_file.stat().st_mtime).date()
            if mtime == datetime.today().date():
                self._build_registry(pd.read_parquet(self._cache_file))
                return
                
        logger.info("Downloading fresh Instrument Master from Upstox...")
        try:
            df = pd.read_csv(self._url)
            # Standardize columns to snake_case
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            
            # Save locally for fast subsequent loads
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._cache_file, index=False)
            self._build_registry(df)
        except Exception as e:
            logger.error(f"Failed to fetch Instrument Master: {e}")

    def _build_registry(self, df: pd.DataFrame):
        """Converts DataFrame to a fast dictionary for O(1) lookups."""
        # Index by the exact format Upstox API expects: exchange_token (e.g. NSE_EQ|INE123...)
        self._registry = df.set_index("instrument_key").to_dict(orient="index")
        logger.info(f"Loaded {len(self._registry)} instruments into master registry.")

    def get(self, instrument_key: str) -> dict:
        """Fetch metadata for any instrument key seamlessly."""
        if instrument_key not in self._registry:
            raise ValueError(f"Unknown instrument key: {instrument_key}")
        return self._registry[instrument_key]
