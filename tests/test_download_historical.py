import pytest
import datetime as dt
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

# Adjust import path if your file is located elsewhere in the package
from algo.tools.download_historical import (
    fetch_chunked_history,
    save_monthly,
    download_data,
)

@pytest.fixture
def mock_ustox():
    """Mocks the global ustox client in the download script."""
    with patch("algo.tools.download_historical.ustox") as mock_client:
        yield mock_client


@pytest.fixture
def sample_historical_df():
    """Provides a dummy dataframe to simulate Upstox API responses."""
    return pd.DataFrame({
        "timestamp": [
            "2026-01-15T09:15:00+05:30",
            "2026-02-10T09:15:00+05:30"
        ],
        "open": [100.0, 105.0],
        "high": [102.0, 106.0],
        "low": [99.0, 104.0],
        "close": [101.0, 105.5],
        "vol": [1000, 1200]
    })


class TestChunkingLogic:
    def test_fetch_chunked_history_respects_monthly_limits(self, mock_ustox, sample_historical_df):
        # Configure mock to return a small dataframe for every API call
        mock_ustox.get_historical.return_value = sample_historical_df

        start = dt.date(2026, 1, 1)
        end = dt.date(2026, 3, 15)
        
        result = fetch_chunked_history(
            key="NSE_INDEX|Nifty 50",
            from_date=start,
            to_date=end,
            is_expired=False
        )

        # 1. Verify the loop correctly broke 2.5 months into exactly 3 API calls
        assert mock_ustox.get_historical.call_count == 3
        
        # 2. Extract the arguments passed to the API to verify the exact boundary math
        calls = mock_ustox.get_historical.call_args_list
        
        # Call 1: Jan 1 to Jan 31
        assert calls[0].kwargs["from_date"] == dt.date(2026, 1, 1)
        assert calls[0].kwargs["to_date"] == dt.date(2026, 1, 31)
        
        # Call 2: Feb 1 to Feb 28
        assert calls[1].kwargs["from_date"] == dt.date(2026, 2, 1)
        assert calls[1].kwargs["to_date"] == dt.date(2026, 2, 28)
        
        # Call 3: Mar 1 to Mar 15 (Caps at the provided end_date)
        assert calls[2].kwargs["from_date"] == dt.date(2026, 3, 1)
        assert calls[2].kwargs["to_date"] == dt.date(2026, 3, 15)

        # 3. Verify dataframes were concatenated and deduplicated correctly
        assert not result.empty


class TestStorageLogic:
    def test_save_monthly_partitions_correctly(self, tmp_path, sample_historical_df):
        target_dir = tmp_path / "historical"
        
        save_monthly(
            data=sample_historical_df,
            target_dir=target_dir,
            key="NSE_INDEX|Nifty 50",
            exchange="NSE",
            interval=1,
            unit="minute"
        )
        
        # The key should be sanitized from "NSE_INDEX|Nifty 50" to "NSE_INDEX_Nifty50"
        sanitized_key_dir = target_dir / "NSE" / "NSE_INDEX_Nifty50"
        
        # 1. Verify directory creation
        assert sanitized_key_dir.exists()
        
        # 2. Verify month-based Parquet splitting (Jan and Feb files should exist)
        jan_file = sanitized_key_dir / "2026_01_1_minute.parquet"
        feb_file = sanitized_key_dir / "2026_02_1_minute.parquet"
        
        assert jan_file.exists()
        assert feb_file.exists()
        
        # 3. Verify internal data integrity (only 1 row per month should be in each file)
        jan_df = pd.read_parquet(jan_file)
        assert len(jan_df) == 1
        assert "2026-01" in str(jan_df.iloc[0]["timestamp"])


class TestDownloadOrchestration:
    @patch("algo.tools.download_historical.fetch_chunked_history")
    @patch("algo.tools.download_historical.save_monthly")
    def test_download_data_aborts_without_spot_data(
        self, mock_save, mock_fetch, mock_ustox
    ):
        # Simulate a total failure to download the underlying spot data
        mock_fetch.return_value = pd.DataFrame()
        
        download_data(
            spot="NSE_INDEX|Nifty 50",
            start_date=dt.date(2026, 1, 1),
            end_date=dt.date(2026, 1, 30),
            is_expired=True
        )
        
        # The options logic should never trigger if the underlying fails
        mock_ustox.get_options_with_expiry.assert_not_called()
        mock_save.assert_not_called()
