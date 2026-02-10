"""
Market data collection from Binance.
"""

from binance.client import Client
import polars as pl
from typing import List, Dict
import time


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
    Fetch candles for multiple symbols.

    Args:
        client: Binance client
        symbols: List of trading pairs
        interval: Candle interval
        limit: Number of candles per symbol

    Returns:
        Dict mapping symbol to Polars DataFrame
    """
    data = {}

    for i, symbol in enumerate(symbols, 1):
        try:
            df = fetch_candles(client, symbol, interval, limit)
            data[symbol] = df
            print(f"[{i}/{len(symbols)}] {symbol}: {len(df)} candles")
            time.sleep(0.1)  # Rate limting
        except Exception as e:
            print(f"⚠️ Failed {symbol}: {e}")

    return data
