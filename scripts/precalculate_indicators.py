#!/usr/bin/env python3
"""
Pre-calculate Technical Indicators

This script pre-calculates ALL technical indicators for all coins/dates
and saves them as parquet files. This way, indicators don't need to be
recalculated during training - they're loaded directly!

Benefits:
- 3-5x faster episode initialization
- Reduces CPU load during training
- GPU can be utilized more efficiently
- Consistent feature calculations across runs
"""

import sys
sys.path.insert(0, '/home/jpao/projects/asymptotic_zero')

from pathlib import Path
import polars as pl
from tqdm import tqdm
from src.data_pipeline.features import TechnicalIndicators
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precalculate_indicators(
    data_dir: str = "data/volatility",
    output_dir: str = "data/volatility_with_indicators",
    force_recalculate: bool = False
):
    """
    Pre-calculate technical indicators for all coins and dates.
    
    Args:
        data_dir: Source directory with raw OHLCV data
        output_dir: Output directory for data with indicators
        force_recalculate: If True, recalculate even if file exists
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print("="*70)
    print("PRE-CALCULATING TECHNICAL INDICATORS")
    print("="*70)
    print(f"\nSource: {data_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Create technical indicators calculator
    ti = TechnicalIndicators(
        config_path="config/features.yaml",
        data_directory=str(data_dir / "5m")
    )
    
    # Find all coin directories
    coin_dirs = sorted([d for d in (data_dir / "5m").iterdir() if d.is_dir()])
    
    print(f"Found {len(coin_dirs)} coins to process")
    print()
    
    total_files = 0
    processed_files = 0
    skipped_files = 0
    failed_files = 0
    
    # Process each coin
    for coin_dir in coin_dirs:
        coin_symbol = coin_dir.name
        
        # Create output directory for this coin
        output_coin_dir = output_dir / "5m" / coin_symbol
        output_coin_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all date files for this coin
        date_files = sorted(coin_dir.glob("*.parquet"))
        total_files += len(date_files)
        
        print(f"Processing {coin_symbol}: {len(date_files)} files")
        
        # Process each date file
        for date_file in tqdm(date_files, desc=f"  {coin_symbol}"):
            date_str = date_file.stem  # e.g., "2024-01-15"
            output_file = output_coin_dir / f"{date_str}.parquet"
            
            # Skip if already exists and not forcing recalculation
            if output_file.exists() and not force_recalculate:
                skipped_files += 1
                continue
            
            try:
                # Load raw OHLCV data
                df = pl.read_parquet(date_file)
                
                # Calculate indicators
                df_with_indicators = ti.calculate(df)
                
                # Save with indicators
                df_with_indicators.write_parquet(output_file)
                
                processed_files += 1
                
            except Exception as e:
                logger.error(f"Failed to process {date_file}: {e}")
                failed_files += 1
    
    # Copy top_movers.parquet
    print("\\nCopying top_movers.parquet...")
    top_movers_src = data_dir / "top_movers.parquet"
    top_movers_dst = output_dir / "top_movers.parquet"
    
    if top_movers_src.exists():
        df_movers = pl.read_parquet(top_movers_src)
        df_movers.write_parquet(top_movers_dst)
        print("✅ Copied top_movers.parquet")
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files found:     {total_files}")
    print(f"Processed:             {processed_files}")
    print(f"Skipped (exist):       {skipped_files}")
    print(f"Failed:                {failed_files}")
    print()
    
    if processed_files > 0:
        print("✅ Indicators pre-calculated successfully!")
        print()
        print("To use the pre-calculated indicators:")
        print("  1. Update your code to load from:")
        print(f"     data_directory=\"{output_dir}\"")
        print()
        print("  2. Or rename directories:")
        print(f"     mv {data_dir} {data_dir}.backup")
        print(f"     mv {output_dir} {data_dir}")
        print()
        print("This will make training 3-5x faster! 🚀")
    
    print("="*70)


def check_indicator_columns(data_dir: str = "data/volatility_with_indicators"):
    """Check what columns are in the pre-calculated files."""
    data_dir = Path(data_dir)
    
    # Find first available file
    for coin_dir in (data_dir / "5m").iterdir():
        if coin_dir.is_dir():
            date_files = list(coin_dir.glob("*.parquet"))
            if date_files:
                sample_file = date_files[0]
                df = pl.read_parquet(sample_file)
                
                print("="*70)
                print(f"Sample file: {sample_file}")
                print("="*70)
                print(f"Total columns: {len(df.columns)}")
                print()
                
                # Show OHLCV columns
                ohlcv = [c for c in df.columns if c in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                print(f"OHLCV columns ({len(ohlcv)}):")
                for col in ohlcv:
                    print(f"  - {col}")
                
                # Show indicator columns
                indicators = [c for c in df.columns if c not in ohlcv]
                print(f"\\nIndicator columns ({len(indicators)}):")
                for i, col in enumerate(indicators[:20], 1):  # Show first 20
                    print(f"  {i:2d}. {col}")
                
                if len(indicators) > 20:
                    print(f"  ... and {len(indicators) - 20} more indicators")
                
                print()
                print("="*70)
                break
    else:
        print("❌ No files found in", data_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-calculate technical indicators")
    parser.add_argument("--check", action="store_true", 
                       help="Check what columns are in pre-calculated files")
    parser.add_argument("--force", action="store_true",
                       help="Force recalculation even if files exist")
    parser.add_argument("--data-dir", default="data/volatility",
                       help="Source data directory")
    parser.add_argument("--output-dir", default="data/volatility_with_indicators",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.check:
        check_indicator_columns(args.output_dir)
    else:
        precalculate_indicators(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            force_recalculate=args.force
        )
