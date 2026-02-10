"""
Five Minute Downloader

Downloads 5-minute candle data ONLY for symbols and dates that are in top movers.
This saves bandwidth and storage since we don't need to download all symbols.

Timeline visualization:
    top_movers.parquet
        │
        ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  For each (date, symbol) in top movers:                      │
    │                                                              │
    │  Check: Does data/volatility/5m/{symbol}/{date}.parquet      │
    │         already exist?                                       │
    │                                                              │
    │  If NO  → Download 5m data for that date                     │
    │  If YES → Skip (don't re-download)                           │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
        │
        ▼
    5m data saved per symbol per date:
    data/volatility/5m/
    ├── BTCUSDT/
    │   ├── 2020-01-01.parquet  (288 candles)
    │   ├── 2020-01-05.parquet
    │   └── ...
    ├── ETHUSDT/
    │   ├── 2020-01-03.parquet
    │   └── ...
    └── ...
"""

import requests
import zipfile
import polars as pl
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional, Tuple
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FiveMinuteDownloader:
    """
    Downloads 5-minute candle data for specific symbols and dates.

    Supports incremental mode: only downloads if file doesn't exist yet.
    """

    BASE_URL = "https://data.binance.vision/data/futures/um"

    def __init__(
        self,
        output_directory: str = "data/volatility/5m",
        top_movers_file: str = "data/volatility/top_movers.parquet",
    ):
        """
        Initialize the 5-minute downloader.

        Args:
            output_directory: Directory to save 5m data
            top_movers_file: Path to top_movers.parquet file
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.top_movers_file = Path(top_movers_file)

        # Manifest file for tracking what's been downloaded
        self.manifest_file = self.output_directory / "_manifest.parquet"

    def _get_file_path(self, symbol: str, target_date: date) -> Path:
        """
        Generate file path for 5m data.

        Args:
            symbol: Trading symbol
            target_date: Date of data

        Returns:
            Path to parquet file
        """
        symbol_directory = self.output_directory / symbol
        symbol_directory.mkdir(exist_ok=True)
        return symbol_directory / f"{target_date}.parquet"

    def _check_already_exists(self, symbol: str, target_date: date) -> bool:
        """
        Check if data for symbol and date already exists.

        Args:
            symbol: Trading symbol
            target_date: Date of data

        Returns:
            True if exists, False otherwise
        """
        return self._get_file_path(symbol, target_date).exists()

    def _download_daily_5m_data(
        self, symbol: str, target_date: date
    ) -> Optional[pl.DataFrame]:
        """
        Download 5m data for one symbol on one date.

        Binance stores daily data in format:
        {SYMBOL}-5m-{YYYY-MM-DD}.zip

        Args:
            symbol: Trading symbol
            target_date: Date of data

        Returns:
            DataFrame with 5m data, or None if failed
        """
        date_string = target_date.strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/daily/klines/{symbol}/5m/{symbol}-5m-{date_string}.zip"

        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 404:
                logger.debug(f"Data not available: {symbol} {date_string}")
                return None

            response.raise_for_status()

            # Save temp zip
            temp_zip = self.output_directory / f"temp_{symbol}_{date_string}.zip"
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

            # Convert timestamp to datetime
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

            # Convert to numeric types
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
            logger.warning(f"Failed to download {symbol} {date_string}: {e}")
            # Cleanup temp files if they exist
            temp_zip = self.output_directory / f"temp_{symbol}_{date_string}.zip"
            if temp_zip.exists():
                temp_zip.unlink()
            return None

    def download_for_date_symbol(
        self, symbol: str, target_date: date, force: bool = False
    ) -> bool:
        """
        Download 5m data for one symbol on one date.

        Args:
            symbol: Trading symbol
            target_date: Date of data
            force: If True, download even if already exists

        Returns:
            True if successful (or already exists), False if failed
        """
        # Convert to date if needed
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        elif isinstance(target_date, str):
            target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

        # Check if already exists
        if not force and self._check_already_exists(symbol, target_date):
            logger.debug(f"Already exists: {symbol} {target_date}")
            return True

        # Download
        df = self._download_daily_5m_data(symbol, target_date)

        if df is None or len(df) == 0:
            return False

        # Save
        output_path = self._get_file_path(symbol, target_date)
        df.write_parquet(output_path)
        logger.debug(f"Saved: {symbol} {target_date} ({len(df)} candles)")

        return True

    def _load_schedule_from_top_movers(self) -> List[Tuple[str, date]]:
        """
        Load download schedule (symbol, date) from top_movers.parquet.

        Returns:
            List of (symbol, date) tuples
        """
        if not self.top_movers_file.exists():
            raise FileNotFoundError(
                f"File {self.top_movers_file} not found. "
                "Run TopMoversCalculator.calculate() first."
            )

        df = pl.read_parquet(self.top_movers_file)

        schedule = []
        for row in df.iter_rows(named=True):
            target_date = row["date"]
            # Convert to date if needed
            if isinstance(target_date, datetime):
                target_date = target_date.date()

            # Combine gainers and losers, remove duplicates
            unique_symbols = set(row["gainers"] + row["losers"])
            for symbol in unique_symbols:
                schedule.append((symbol, target_date))

        return schedule

    def download_all_from_top_movers(
        self, limit: Optional[int] = None, delay_between_requests: float = 0.1
    ) -> dict:
        """
        Download all 5m data based on top_movers.parquet.

        Args:
            limit: Limit number of downloads (for testing)
            delay_between_requests: Delay between requests in seconds

        Returns:
            Dictionary with download statistics
        """
        schedule = self._load_schedule_from_top_movers()
        logger.info(f"Total schedule: {len(schedule)} (symbol, date) combinations")

        # Filter out already existing
        schedule_pending = [
            (symbol, target_date)
            for symbol, target_date in schedule
            if not self._check_already_exists(symbol, target_date)
        ]

        logger.info(
            f"Need to download: {len(schedule_pending)} "
            f"(already exists: {len(schedule) - len(schedule_pending)})"
        )

        if not schedule_pending:
            logger.info("All data already downloaded!")
            return {
                "total_schedule": len(schedule),
                "already_exists": len(schedule),
                "downloaded": 0,
                "failed": 0,
            }

        # Apply limit if provided
        if limit:
            schedule_pending = schedule_pending[:limit]
            logger.info(f"Limited to {limit} downloads")

        # Statistics
        statistics = {
            "total_schedule": len(schedule),
            "already_exists": len(schedule) - len(schedule_pending),
            "downloaded": 0,
            "failed": 0,
            "failed_list": [],
        }

        # Download with progress bar
        for symbol, target_date in tqdm(schedule_pending, desc="Downloading 5m data"):
            success = self.download_for_date_symbol(symbol, target_date)

            if success:
                statistics["downloaded"] += 1
            else:
                statistics["failed"] += 1
                statistics["failed_list"].append((symbol, str(target_date)))

            time.sleep(delay_between_requests)

        # Update manifest
        self._update_manifest()

        logger.info(
            f"Done: {statistics['downloaded']} downloaded, "
            f"{statistics['failed']} failed"
        )

        return statistics

    def _update_manifest(self):
        """
        Update manifest file with list of downloaded files.
        """
        all_files = []

        for symbol_directory in self.output_directory.iterdir():
            if not symbol_directory.is_dir():
                continue
            if symbol_directory.name.startswith("_"):
                continue

            symbol = symbol_directory.name
            for file in symbol_directory.glob("*.parquet"):
                target_date = file.stem
                all_files.append(
                    {"symbol": symbol, "date": target_date, "path": str(file)}
                )

        if all_files:
            df = pl.DataFrame(all_files)
            df.write_parquet(self.manifest_file)
            logger.info(f"Manifest updated: {len(all_files)} files")

    def load_data_for_date(
        self, target_date: date, symbol_filter: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Load all 5m data for one trading date.

        Args:
            target_date: Trading date
            symbol_filter: If provided, only load these symbols

        Returns:
            DataFrame with additional 'symbol' column
        """
        # Convert to date if needed
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        elif isinstance(target_date, str):
            target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

        all_data = []

        for symbol_directory in self.output_directory.iterdir():
            if not symbol_directory.is_dir():
                continue
            if symbol_directory.name.startswith("_"):
                continue

            symbol = symbol_directory.name

            # Filter symbols if provided
            if symbol_filter and symbol not in symbol_filter:
                continue

            date_file = symbol_directory / f"{target_date}.parquet"
            if not date_file.exists():
                continue

            df = pl.read_parquet(date_file)
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            all_data.append(df)

        if not all_data:
            logger.warning(f"No data found for date {target_date}")
            return pl.DataFrame()

        return pl.concat(all_data)

    def statistics(self) -> dict:
        """
        Generate statistics of downloaded data.

        Returns:
            Dictionary with statistics
        """
        total_files = 0
        total_symbols = 0
        symbol_set = set()

        for symbol_directory in self.output_directory.iterdir():
            if not symbol_directory.is_dir():
                continue
            if symbol_directory.name.startswith("_"):
                continue

            symbol = symbol_directory.name
            symbol_set.add(symbol)
            total_symbols += 1

            file_count = len(list(symbol_directory.glob("*.parquet")))
            total_files += file_count

        return {
            "total_symbols": total_symbols,
            "total_files": total_files,
            "symbols": sorted(list(symbol_set)),
        }


if __name__ == "__main__":
    # Test download
    downloader = FiveMinuteDownloader()

    # Test download one
    # success = downloader.download_for_date_symbol("BTCUSDT", date(2024, 1, 15))
    # print(f"Success: {success}")

    # Test download all (with limit for testing)
    # stats = downloader.download_all_from_top_movers(limit=10)
    # print(stats)

    # Statistics
    stats = downloader.statistics()
    print(f"Statistics: {stats}")
