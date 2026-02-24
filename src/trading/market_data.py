"""
Market data collection from Binance.
"""

import logging
from binance.client import Client
import polars as pl
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

_log = logging.getLogger(__name__)


def fetch_candles(
    client: Client, symbol: str, interval: str = "5m", limit: int = 300
) -> pl.DataFrame:
    """
    Fetch historical candles for a symbol.

    Args:
        client: Binance client
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval (default '5m')
        limit: Number of candles to fetch (default 300)

    Returns:
        Polars DataFrame with OHLCV data
    """
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)

    # Convert to Polars DataFrame
    df = pl.DataFrame(
        {
            "timestamp": [k[0] for k in klines],
            "open": [float(k[1]) for k in klines],
            "high": [float(k[2]) for k in klines],
            "low": [float(k[3]) for k in klines],
            "close": [float(k[4]) for k in klines],
            "volume": [float(k[5]) for k in klines],
        }
    )

    # Convert timestamp to datetime
    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("timestamp")
    )

    return df


def fetch_all_coins(
    client: Client, symbols: List[str], interval: str = "5m", limit: int = 300
) -> Dict[str, pl.DataFrame]:
    """
    Fetch candles for multiple symbols concurrently.

    Args:
        client: Binance client
        symbols: List of trading pairs
        interval: Candle interval
        limit: Number of candles per symbol

    Returns:
        Dict mapping symbol to Polars DataFrame
    """
    data = {}

    def _fetch_one(symbol: str):
        try:
            return symbol, fetch_candles(client, symbol, interval, limit)
        except Exception as e:
            _log.warning(f"Failed to fetch {symbol}: {e}")
            return symbol, None

    # Fetch all symbols in parallel; overall wall-clock time ≈ one request timeout
    # instead of N × timeout when sequential.
    with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures, timeout=60):
            symbol, df = future.result()
            if df is not None:
                data[symbol] = df
                _log.debug(f"{symbol}: {len(df)} candles")

    return data
