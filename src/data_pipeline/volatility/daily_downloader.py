"""
Daily Data Downloader (1D)

Downloads daily candle data for ALL perpetual futures available on Binance.
This data is used to calculate top gainers and top losers for each day.

Timeline visualization:
    5 years ago                                            Today
        │                                                    │
        ▼                                                    ▼
    ┌────────────────────────────────────────────────────────────┐
    │  Download 1D candles for ~200-400 USDT perpetual futures   │
    │  Save per symbol: data/volatility/daily/{SYMBOL}.parquet   │
    └────────────────────────────────────────────────────────────┘
"""

import requests
import zipfile
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyDownloader:
    """
    Downloads daily candle data (1d) for all USDT perpetual futures.

    Daily data is relatively small (~500 rows per symbol for 5 years)
    so it's safe to download all symbols.
    """

    BASE_URL = "https://data.binance.vision/data/futures/um"
    BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    def __init__(self, output_directory: str = "data/volatility/daily"):
        """
        Initialize the daily downloader.

        Args:
            output_directory: Directory to save daily data
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # File to store the list of available symbols
        self.symbols_list_file = self.output_directory / "_symbols_list.txt"

        # Cache for symbols list
        self._symbols_cache: Optional[List[str]] = None

    def fetch_perpetual_symbols(self, force_refresh: bool = False) -> List[str]:
        """
        Fetch list of all USDT perpetual futures from Binance API.

        Results are cached to file to avoid repeated requests.

        Args:
            force_refresh: If True, fetch from API even if cache exists

        Returns:
            List of USDT perpetual symbols (e.g., ["BTCUSDT","ETHUSDT", ...])
        """
        # Check memory cache
        if self._symbols_cache and not force_refresh:
            return self._symbols_cache

        # Check file cache (if not older than 1 day)
        if self.symbols_list_file.exists() and not force_refresh:
            modification_time = datetime.fromtimestamp(
                self.symbols_list_file.stat().st_mtime
            )
            if datetime.now() - modification_time < timedelta(days=1):
                with open(self.symbols_list_file, "r") as f:
                    symbols = [line.strip() for line in f if line.strip()]
                self._symbols_cache = symbols
                logger.info(f"Loaded {len(symbols)} symbols from cache")
                return symbols

        # Fetch from API
        logger.info("Fetching symbols list from Binance API...")
        try:
            response = requests.get(self.BINANCE_API_URL, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Filter only USDT perpetuals
            symbols = []
            for item in data.get("symbols", []):
                # Perpetual = contractType PERPETUAL, quoteAsset USDT
                if (
                    item.get("contractType") == "PERPETUAL"
                    and item.get("quoteAsset") == "USDT"
                    and item.get("status") == "TRADING"
                ):
                    symbols.append(item["symbol"])

            symbols.sort()

            # Save to file cache
            with open(self.symbols_list_file, "w") as f:
                f.write("\n".join(symbols))

            self._symbols_cache = symbols
            logger.info(f"Found {len(symbols)} USDT perpetual futures")
            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch symbols list: {e}")

            # Fallback to cache if exists
            if self.symbols_list_file.exists():
                with open(self.symbols_list_file, "r") as f:
                    symbols = [line.strip() for line in f if line.strip()]
                logger.warning(f"Using old cache: {len(symbols)} symbols")
                return symbols

            raise

    def _generate_monthly_range(self, start_date: str, end_date: str) -> List[str]:
        """Generate list of month strings (YYYY-MM) for downloading."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        months = []
        current = start.replace(day=1)

        while current <= end:
            months.append(current.strftime("%Y-%m"))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return months

    def _download_monthly_chunk(
        self, symbol: str, month_string: str
    ) -> Optional[pl.DataFrame]:
        """
        Download one monthly chunk of 1d data.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            month_string: Month in YYYYY-MM format

        Returns:
            DataFrame with candle data, or None if not available
        """
        url = (
            f"{self.BASE_URL}/monthly/klines/{symbol}/1d/{symbol}-1d-{month_string}.zip"
        )

        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 404:
                return None

            response.raise_for_status()

            # Save temp zip
            temp_zip = self.output_directory / f"temp_{symbol}_{month_string}.zip"
            with open(temp_zip, "wb") as f:
                f.write(response.content)

            # Extract CSV from zip
            with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                csv_name = zip_ref.namelist()[0]
                zip_ref.extract(csv_name, self.output_directory)

            # Read CSV
            csv_path = self.output_directory / csv_name
            df = pl.read_csv(
                csv_path,
                has_header=False,
                new_columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trade_count",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            # Cleanup temp files
            temp_zip.unlink()
            csv_path.unlink()

            # Check if first row is header
            first_val = df.select(pl.col("open_time").first()).item()
            if first_val == "open_time" or not str(first_val).isdigit():
                df = df.slice(1)

            df = df.with_columns(
                [
                    pl.col("open_time")
                    .cast(pl.Int64, strict=False)
                    .cast(pl.Datetime("ms")),
                    pl.col("close_time")
                    .cast(pl.Int64, strict=False)
                    .cast(pl.Datetime("ms")),
                ]
            )

            # Convert timestamp to datetime
            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_base",
                "taker_buy_quote",
            ]
            df = df.with_columns(
                [pl.col(col).cast(pl.Float64, strict=False) for col in numeric_columns]
            )

            df = df.with_columns(pl.col("trade_count").cast(pl.Int64, strict=False))

            # Select needed columns
            df = df.select(
                [
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "trade_count",
                ]
            )

            return df

        except Exception as e:
            logger.debug(f"Failed to download {symbol} for {month_string}: {e}")
            return None

    def _check_existing_data(self, symbol: str) -> Optional[datetime]:
        """
        Check if data already exists for this symbol and when it ends.

        Args:
            symbol: Trading symbol

        Returns:
            Last data date, or None if no data exists
        """
        symbol_file = self.output_directory / f"{symbol}.parquet"

        if not symbol_file.exists():
            return None

        try:
            df = pl.read_parquet(symbol_file)
            last_date = df.select(pl.col("open_time").max()).item()
            return last_date
        except Exception:
            return None

    def download_symbol(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        incremental_mode: bool = True,
    ) -> bool:
        """
        Download daily data for one symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), default yesterday
            incremental_mode: if True, only download data that doesn't exist yet

        Returns:
            True if successful, False if failed
        """
        if end_date is None:
            # Use yesterday since today's data might be incomplete
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Check existing data for incremental download
        if incremental_mode:
            last_date = self._check_existing_data(symbol)
            if last_date:
                # Start from month after last data
                new_start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                if new_start_date > end_date:
                    logger.debug(f"{symbol}: Data already complete up to {last_date}")
                    return True
                start_date = new_start_date
                logger.info(f"{symbol}: Incremental mode, starting from {start_date}")

        # Generate month range
        months = self._generate_monthly_range(start_date, end_date)

        if not months:
            return True

        # Download each month
        all_data = []
        for month_string in months:
            df = self._download_monthly_chunk(symbol, month_string)
            if df is not None and len(df) > 0:
                all_data.append(df)
            time.sleep(0.1)  # Rate limiting

        if not all_data:
            logger.warning(f"No new data for {symbol}")
            return False

        # Combine new data
        df_new = pl.concat(all_data)
        df_new = df_new.unique(subset=["open_time"], maintain_order=True)
        df_new = df_new.sort("open_time")

        # If incremental, combine with old data
        symbol_file = self.output_directory / f"{symbol}.parquet"
        if incremental_mode and symbol_file.exists():
            df_old = pl.read_parquet(symbol_file)
            df_combined = pl.concat([df_old, df_new])
            df_combined = df_combined.unique(subset=["open_time"], maintain_order=True)
            df_combined = df_combined.sort("open_time")
        else:
            df_combined = df_new

        # Save
        df_combined.write_parquet(symbol_file)
        logger.info(
            f"{symbol}: Saved {len(df_combined)} daily candles "
            f"({df_combined['open_time'].min()} - {df_combined['open_time'].max()})"
        )

        return True

    def download_all_symbols(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        incremental_mode: bool = True,
        symbol_filter: Optional[List[str]] = None,
    ) -> dict:
        """
        Download daily data for all USDT perpetual futures.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            incremental_mode: If True, only download data that doesn't exist
            symbol_filter: If provided, only download these symbols
        """
        # Get symbols list
        if symbol_filter:
            symbols = symbol_filter
        else:
            symbols = self.fetch_perpetual_symbols()

        logger.info(f"Will download daily data for {len(symbols)} symbols")

        statistics = {
            "total": len(symbols),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "failed_symbols": [],
        }

        for symbol in tqdm(symbols, desc="Downloading daily data"):
            try:
                success = self.download_symbol(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    incremental_mode=incremental_mode,
                )
                if success:
                    statistics["successful"] += 1
                else:
                    statistics["failed"] += 1
                    statistics["failed_symbols"].append(symbol)
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                statistics["failed"] += 1
                statistics["failed_symbols"].append(symbol)

        logger.info(
            f"Done: {statistics['successful']} successful, "
            f"{statistics['failed']} failed out of {statistics['total']} symbols"
        )

        return statistics

    def load_all_daily_data(self) -> pl.DataFrame:
        """
        Load all downloaded daily data into one DataFrame.

        Returns:
            DataFrame with additional 'symbol' column for identification
        """
        all_files = list(self.output_directory.glob("*.parquet"))

        if not all_files:
            raise FileNotFoundError("No daily data found")

        all_data = []
        for file in all_files:
            symbol = file.stem  # Filename without extension
            if symbol.startswith("_"):  # Skip internal files like _symbols_list
                continue

            df = pl.read_parquet(file)
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            all_data.append(df)

        df_combined = pl.concat(all_data)
        logger.info(
            f"Loaded {len(all_data)} symbols with total {len(df_combined)} rows"
        )

        return df_combined


if __name__ == "__main__":
    # Test download
    downloader = DailyDownloader()

    # Fetch symbols list
    symbols = downloader.fetch_perpetual_symbols()
    print(f"Found {len(symbols)} symbols")
    print(f"Sample: {symbols[:10]}")
