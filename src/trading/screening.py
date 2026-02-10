"""
Morning screening to identify top gainers and losers.
"""

from binance.client import Client
from typing import Dict, List
import time


def get_top_movers(
    client: Client, num_coins: int = 10, min_volume_usdt: float = 10_000_000
) -> Dict:
    """
    Screen for top gainers and losers in the last 24 hours.

    Args:
        client: Binance client
        num_coins: Number of gainers/losers to return (default 10)
        min_volume_usdt: Minimum 24h volume in USDT (default 10M)

    Returns:
        Dict with gainers, losers, and their 24h price changes
    """
    # Get all 24h tickers
    tickers = client.futures_ticker()

    # Filter for USDT perpetuals with sufficient volume
    usdt_tickers = []
    for ticker in tickers:
        if ticker["symbol"].endswith("USDT"):
            volume = float(ticker["quoteVolume"])
            if volume >= min_volume_usdt:
                usdt_tickers.append(
                    {
                        "symbol": ticker["symbol"],
                        "priceChangePercent": float(ticker["priceChangePercent"]),
                        "volume": volume,
                        "lastPrice": float(ticker["lastPrice"]),
                    }
                )

    # Sort by price change
    sorted_tickers = sorted(
        usdt_tickers, key=lambda x: x["priceChangePercent"], reverse=True
    )

    # Get top gainers
    gainers = sorted_tickers[:num_coins]

    # Get top losers (from the end)
    losers = sorted_tickers[-num_coins:][::-1]  # Reverse ti get worst first

    return {
        "gainers": [coin["symbol"] for coin in gainers],
        "losers": [coin["symbol"] for coin in losers],
        "gainers_pct": [coin["priceChangePercent"] for coin in gainers],
        "losers_pct": [coin["priceChangePercent"] for coin in losers],
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    # Add project root to path
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Test the screening
    from src.config.secrets import get_secret

    api_key = get_secret("BINANCE_API_KEY")
    api_secret = get_secret("BINANCE_SECRET")
    trading_mode = get_secret("TRADING_MODE")

    client = Client(api_key, api_secret, testnet=trading_mode == "testnet")

    # Handle time offset
    server_time = client.get_server_time()
    local_time = int(time.time() * 1000)
    time_offset = server_time["serverTime"] - local_time
    client.timestamp_offset = time_offset

    print("Running morning screening...")
    result = get_top_movers(client)

    print("\n📈 Top 10 Gainers:")
    for i, (symbol, pct) in enumerate(zip(result["gainers"], result["gainers_pct"]), 1):
        print(f"  {i:2}. {symbol:12} +{pct:6.2f}%")

    print("📉 Top 10 losers:")
    for i, (symbol, pct) in enumerate(zip(result["losers"], result["losers_pct"]), 1):
        print(f"  {i:2}. {symbol:12} {pct:7.2f}%")
