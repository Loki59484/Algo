import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch
from algo.core.master import InstrumentMaster
from algo.core.datatypes import Candle
from algo.core.instrument import Instrument


@pytest.fixture
def mock_registry():
    """Mock registry dictionary isolating tests from external network downloads."""
    sample_registry = {
        "NSE_INDEX|Nifty 50": {
            "exchange": "NSE",
            "segment": "NSE_INDEX",
            "instrument_type": "INDEX",
            "lot_size": 1,
            "tick_size": 0.05,
            "strike_price": 0.0,
            "freeze_quantity": 0.0,
            "expiry": None,
        },
        "NSE_FO|58529": {
            "exchange": "NSE",
            "segment": "NSE_FO",
            "instrument_type": "CE",
            "lot_size": 25,
            "tick_size": 0.05,
            "strike_price": 24500.0,
            "freeze_quantity": 1800.0,
            "expiry": "2026-10-29",
        },
        "MCX_FO|428123": {
            "exchange": "MCX",
            "segment": "MCX_FO",
            "instrument_type": "FUT",
            "lot_size": 100,
            "tick_size": 1.0,
            "strike_price": 0.0,
            "freeze_quantity": 500.0,
            "expiry": "2026-11-20",
        },
    }
    with patch.object(InstrumentMaster, "_load_master"):
        master = InstrumentMaster()
        master._registry = sample_registry
        yield master


@pytest.fixture
def mock_client():
    """Mock Upstox client to prevent network or rate-limiter calls."""
    client = MagicMock()
    # Define weekend/holiday behavior (e.g., Saturday 2026-06-06 and Sunday 2026-06-07 are holidays)
    client.is_exchange_holiday.side_effect = lambda d, exchange="NSE": (
        d.weekday() in (5, 6) or d == date(2026, 6, 10)
    )
    return client


# =====================================================================
# 1. Metadata & Initialization Tests
# =====================================================================
class TestInstrumentInitialization:
    def test_equity_index_parsing(self, mock_registry):
        ins = Instrument("NSE_INDEX|Nifty 50")
        assert ins.exchange == "NSE"
        assert ins.is_index is True
        assert ins.strike_price == 0.0
        assert ins.expiry is None
        assert ins.lot_size == 1

    def test_derivative_parsing(self, mock_registry):
        ins = Instrument("NSE_FO|58529")
        assert ins.exchange == "NSE"
        assert ins.is_index is False
        assert ins.type == "CE"
        assert ins.lot_size == 25
        assert ins.strike_price == 24500.0
        assert ins.expiry == date(2026, 10, 29)

    def test_mcx_commodity_parsing(self, mock_registry):
        ins = Instrument("MCX_FO|428123")
        assert ins.exchange == "MCX"
        assert ins.lot_size == 100
        assert ins.tick_size == 1.0

    def test_unknown_key_raises_value_error(self, mock_registry):
        with pytest.raises(ValueError, match="Unknown instrument key"):
            Instrument("UNKNOWN|00000")


# =====================================================================
# 2. Data Cleaning & Candle Conversion
# =====================================================================
class TestCandleProcessing:
    def test_df_to_candles_cleaning_and_tz(self, mock_registry):
        ins = Instrument("NSE_INDEX|Nifty 50")

        raw_df = pd.DataFrame(
            {
                "timestamp": ["2026-06-09 09:15:00", "2026-06-09 09:16:00"],
                "open": [24000.0, np.nan],  # Requires ffill
                "high": [24050.0, 24060.0],
                "low": [23990.0, 24010.0],
                "close": [24020.0, 24050.0],
                "vol": ["1500", "invalid_vol"],  # String & non-numeric coercion
            }
        )

        candles = ins.df_to_candles(raw_df)

        assert len(candles) == 2
        assert candles[1].open == 24000.0  # Successfully forward-filled
        assert candles[0].volume == 1500.0
        assert candles[1].volume == 0.0  # Coerced invalid to 0.0
        assert str(candles[0].timestamp.tzinfo) == "Asia/Kolkata"

    def test_df_to_candles_empty_df(self, mock_registry):
        ins = Instrument("NSE_INDEX|Nifty 50")
        assert ins.df_to_candles(pd.DataFrame()) == []


# =====================================================================
# 3. Lookback & Chunking Strategy
# =====================================================================
class TestLookbackAndChunking:
    def test_calculate_lookback_dates_skips_holidays(self, mock_registry, mock_client):
        ins = Instrument("NSE_FO|58529", date="2026-06-12")  # Friday

        start, end = ins._calculate_lookback_dates(mock_client, lookback=3)
        assert end == date(2026, 6, 12)
        # Expected: June 12 (Day 1), June 11 (Day 2), June 10 is mock holiday, June 9 (Day 3)
        assert start == date(2026, 6, 9)  # FIXED: Changed from 8 to 9

    def test_build_date_chunks_split_across_months(self, mock_registry):
        ins = Instrument("NSE_FO|58529")
        start = date(2026, 1, 15)
        end = date(2026, 3, 10)

        chunks = ins._build_date_chunks(start, end)
        assert len(chunks) == 2  # FIXED: Changed from 3 to 2
        assert chunks[0] == (date(2026, 1, 15), date(2026, 2, 14))
        assert chunks[1] == (date(2026, 2, 15), date(2026, 3, 10))

    def test_zero_lookback_returns_empty_dataframe(self, mock_registry, mock_client):
        ins = Instrument("NSE_FO|58529")
        result = ins.load_historical_df(mock_client, lookback=0, is_expired=False)
        assert result.empty


# =====================================================================
# 4. Cache & Storage Isolation
# =====================================================================
class TestCacheAndStorage:
    def test_load_previous_reads_existing_cache(self, mock_registry, mock_client, tmp_path):
        ins = Instrument("NSE_FO|58529")
        from_d = date(2026, 5, 1)
        to_d = date(2026, 5, 15)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / f"{ins.key}_{from_d}_{to_d}.parquet"

        dummy_data = pd.DataFrame({"close": [100.0, 102.0], "vol": [50, 60]})
        dummy_data.to_parquet(cache_file)

        with patch("algo.core.upstox_methods.DATA_DIR", tmp_path):
            result = ins.load_previous(mock_client, from_date=from_d, to_date=to_d)

        assert not result.empty
        assert len(result) == 2
        mock_client.get_historical.assert_not_called()  # Proves API hit was avoided


# =====================================================================
# 5. Factory Methods
# =====================================================================
class TestFactoryMethods:
    def test_load_instrument_from_metadata(self, mock_registry, mock_client):
        meta = {
            "instrument_key": "NSE_FO|58529",
            "date": "2026-06-09",
            "interval": "5",
            "unit": "minute",
        }
        ins = Instrument.load_instrument(mock_client, metadata=meta)

        assert ins.key == "NSE_FO|58529"
        assert ins.interval == "5"
        assert ins.lot_size == 25  # Master enforces static attribute integrity
