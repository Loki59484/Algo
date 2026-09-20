from __future__ import annotations

import concurrent.futures
import logging
from collections import deque
from dataclasses import field
from datetime import date as dt_date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from algo.core.datatypes import Candle, Position
from algo.core.master import InstrumentMaster

logger = logging.getLogger(__name__)


class Instrument:
    """A segment-agnostic financial instrument."""

    def __init__(
        self,
        instrument_key: str,
        date: str | datetime | None = None,
        interval: str = "1",
        unit: str = "minute",
    ):
        self.key = instrument_key
        self.date = pd.to_datetime(date).date() if date else datetime.today().date()
        self.interval = interval
        self.unit = unit

        master = InstrumentMaster()
        meta = master.get(self.key)

        self.exchange: str = meta.get("exchange", "UNKNOWN")
        self.type: str = meta.get("instrument_type", "UNKNOWN")

        lot_size_raw = meta.get("lot_size", 1)
        self.lot_size: int = int(lot_size_raw) if pd.notna(lot_size_raw) else 1

        tick_size_raw = meta.get("tick_size", 0.05)
        self.tick_size: float = float(tick_size_raw) if pd.notna(tick_size_raw) else 0.05

        strike_raw = meta.get("strike_price", 0.0)
        self.strike_price: float = float(strike_raw) if pd.notna(strike_raw) else 0.0

        freeze_raw = meta.get("freeze_quantity", 0.0)
        self.freeze_qty: float = float(freeze_raw) if pd.notna(freeze_raw) else 0.0

        expiry_raw = meta.get("expiry")
        self.expiry = pd.to_datetime(expiry_raw).date() if pd.notna(expiry_raw) else None

        self.historical_candles: deque[Candle] = deque(maxlen=100000)

    @property
    def is_index(self) -> bool:
        return self.type == "INDEX"

    def df_to_candles(self, df: pd.DataFrame) -> list[Candle]:
        candles = []
        if not df.empty:
            clean_data = df.copy()
            clean_data[["open", "high", "low", "close"]] = clean_data[
                ["open", "high", "low", "close"]
            ].ffill()
            clean_data["vol"] = (
                pd.to_numeric(clean_data["vol"], errors="coerce").fillna(0).astype(float)
            )

            def _normalize_tz(ts):
                if pd.isna(ts):
                    return ts
                parsed = pd.to_datetime(ts)
                return (
                    parsed.tz_localize("Asia/Kolkata")
                    if parsed.tzinfo is None
                    else parsed.tz_convert("Asia/Kolkata")
                )

            clean_data["timestamp"] = clean_data["timestamp"].apply(_normalize_tz)
            candles = [
                Candle(timestamp=t, open=o, high=h, low=l, close=c, volume=v)
                for t, o, h, l, c, v in zip(
                    clean_data["timestamp"],
                    clean_data["open"],
                    clean_data["high"],
                    clean_data["low"],
                    clean_data["close"],
                    clean_data["vol"],
                )
            ]
        return candles

    def _calculate_lookback_dates(self, client, lookback: int):
        end_date = self.date
        start_date = end_date
        days_found = 0
        while days_found < lookback:
            if not client.is_exchange_holiday(start_date, exchange=self.exchange):
                days_found += 1
            if days_found < lookback:
                start_date -= timedelta(days=1)
        return start_date, end_date

    def _build_date_chunks(self, start_date, end_date):
        date_chunks = []
        curr_start = start_date
        while curr_start <= end_date:
            curr_end = min(
                curr_start + relativedelta(months=1) - timedelta(days=1), end_date
            )
            date_chunks.append((curr_start, curr_end))
            curr_start = curr_end + timedelta(days=1)
        return date_chunks

    def load_previous(
        self, client, from_date: datetime, to_date: datetime, is_expired: bool = True
    ):
        from algo.core.upstox_methods import DATA_DIR
        from algo.tools.download_historical import download_cache

        cache_dir = DATA_DIR / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.key}_{from_date}_{to_date}.parquet"

        if cache_file.exists() and cache_file.stat().st_size > 0:
            return pd.read_parquet(cache_file)

        if not is_expired:
            return client.get_historical(
                instrument_key=self.key, from_date=from_date, to_date=to_date
            )

        return download_cache(
            instrument_key=self.key,
            is_expired=True,
            expiry=self.expiry,
            from_date=from_date,
            to_date=to_date,
            out_path=cache_file,
        )

    def load_historical_df(self, client, lookback: int, is_expired: bool = False) -> pd.DataFrame:
        if lookback <= 0:
            return pd.DataFrame()
        start_date, end_date = self._calculate_lookback_dates(client, lookback)
        date_chunks = self._build_date_chunks(start_date, end_date)
        dfs = []
        for c_start, c_end in date_chunks:
            chunk = self.load_previous(client, from_date=c_start, to_date=c_end, is_expired=is_expired)
            if chunk is not None and not chunk.empty:
                dfs.append(chunk)
        if not dfs:
            return pd.DataFrame()
        clean = pd.concat(dfs, ignore_index=True)
        clean[["open", "high", "low", "close"]] = clean[["open", "high", "low", "close"]].ffill()
        clean["vol"] = pd.to_numeric(clean["vol"], errors="coerce").fillna(0).astype(float)
        return clean

    @classmethod
    def load_instrument(
        cls,
        client,
        path: Path | str | None = None,
        lookback: int = 0,
        data: pd.DataFrame | None = None,
        metadata: dict | None = None,
        load_history: bool = False,
        is_expired: bool = True,
    ):
        from algo.core.methods import load_parquet

        if path is not None:
            data, metadata = load_parquet(path).values()

        if metadata is None:
            metadata = {}

        ins = cls(
            instrument_key=metadata.get("instrument_key", "UNKNOWN"),
            date=metadata.get("date"),
            interval=str(metadata.get("interval", "1")),
            unit=str(metadata.get("unit", "minute")),
        )

        data_dfs = []
        if load_history and lookback > 0:
            hist_data = ins.load_historical_df(client, lookback, is_expired=is_expired)
            if not hist_data.empty:
                data_dfs.append(hist_data)

        if data is not None and not data.empty:
            data_dfs.append(data)

        if data_dfs:
            final_data = pd.concat(data_dfs, ignore_index=True)
            ins.historical_candles.extend(ins.df_to_candles(final_data))

        return ins


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Bucket:
    """A collection of instruments managed and traded together."""

    date: datetime
    tag: str | None = None
    spot: Instrument | None = None
    legs: dict[str, Instrument] = field(default_factory=dict)
    open_position: Position | None = None
    probability_matrix: Any = None
    pending_entry: Any = None
    pending_exit: Any = None

    def add_leg(self, item: Instrument | dict[str, Instrument], leg_type: str | None = None):
        """Adds Instrument instances as legs to the bucket."""
        if isinstance(item, dict):
            self.legs = {**self.legs, **item}
        elif isinstance(item, Instrument):
            key = leg_type if leg_type is not None else item.type
            self.legs[key] = item
