#!/usr/bin/env python3
"""
Token transfer extraction.

Extract ERC-20 token transfers from Ethereum via the Etherscan API.
Supports checkpointing for large batch runs.
"""

import pandas as pd
import time
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import gc
import psutil

from transaction_tracker_complete import CompleteEthereumTransactionTracker


# ========================================
# Configuration Constants
# ========================================

# Input CSV file containing token contract addresses
INPUT_CSV_FILE = "tokensout_scam.csv"

# Column name in CSV that holds the contract address
CONTRACT_ADDRESS_COLUMN = "contract_address"

# Column name for token names (optional, will auto-generate if missing)
TOKEN_NAME_COLUMN = "tokenname"


@dataclass
class Config:
    """Centralized configuration for the extraction pipeline."""
    max_workers: int = 4
    delay_between_tokens: float = 0.1
    retry_attempts: int = 3
    retry_delay: float = 1.0
    memory_threshold: float = 0.8
    log_level: str = "INFO"
    max_transfers: int = 500  # Cap transfers per token (None = unlimited)
    enable_checkpoint: bool = True  # Enable checkpoint/resume functionality
    
    # These paths are auto-generated based on input CSV name
    output_dir: str = None
    log_dir: str = "logs"
    checkpoint_file: str = None


@dataclass
class TokenResult:
    """Stores extraction results for a single token."""
    token_name: str
    contract_address: str
    total_transfers: int = 0
    erc20_transfers: int = 0
    eth_transfers: int = 0
    internal_transfers: int = 0
    output_file: Optional[str] = None
    status: str = "PENDING"
    error: Optional[str] = None
    processed_at: Optional[str] = None
    processing_time: float = 0.0
    file_size_mb: float = 0.0


class TokenTransferExtractor:
    """
    Handles bulk extraction of token transfers from Ethereum.
    
    Features:
    - Multi-threaded processing with API key rotation
    - Automatic checkpointing for crash recovery
    - Memory-aware processing with garbage collection
    """
    
    def __init__(self, api_keys: List[str], csv_file: str, config: Optional[Config] = None):
        # Handle single API key passed as string
        if isinstance(api_keys, str):
            self.api_keys = [api_keys]
        else:
            self.api_keys = api_keys
        
        self.csv_file = csv_file
        self.config = config or Config()
        
        # Generate output paths based on CSV filename
        self._setup_paths_from_csv()
        
        self.tracker = CompleteEthereumTransactionTracker(self.api_keys)
        
        self._setup_logging()
        self._setup_directories()
        
        # Track which contracts we've already processed
        self.processed_contracts: Set[str] = set()
        self._load_checkpoint()
        
        # Runtime statistics
        self._stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'memory_peak': 0.0
        }
        
        self.logger.info(f"Input CSV: {self.csv_file}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"API Keys available: {len(self.api_keys)}")
        if self.config.max_transfers:
            self.logger.info(f"Transfer limit: {self.config.max_transfers} per token")
        if self.config.enable_checkpoint:
            self.logger.info(f"Checkpoint loaded: {len(self.processed_contracts)} tokens already done")
    
    def _setup_paths_from_csv(self):
        """Generate output paths based on the input CSV filename."""
        csv_basename = Path(self.csv_file).stem
        
        if self.config.output_dir is None:
            self.config.output_dir = f"{csv_basename}_results"
        
        if self.config.checkpoint_file is None:
            self.config.checkpoint_file = f"{self.config.log_dir}/{csv_basename}_checkpoint.json"
    
    def _setup_logging(self):
        """Configure logging to both file and console."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_file = f"{self.config.log_dir}/extractor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        self.logger = logging.getLogger('TokenTransferExtractor')
        self.logger.setLevel(getattr(logging, self.config.log_level))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Silence noisy HTTP libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
    
    def _setup_directories(self):
        """Create required output directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
    
    def _load_checkpoint(self):
        """Load checkpoint file to resume from previous run."""
        if not self.config.enable_checkpoint:
            return
        
        checkpoint_path = Path(self.config.checkpoint_file)
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    self.processed_contracts = set(data.get('processed_contracts', []))
                self.logger.info(f"Checkpoint loaded: {len(self.processed_contracts)} tokens")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
                self.processed_contracts = set()
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        if not self.config.enable_checkpoint:
            return
        
        try:
            data = {
                'csv_file': self.csv_file,
                'processed_contracts': list(self.processed_contracts),
                'last_update': datetime.now().isoformat(),
                'total_processed': len(self.processed_contracts)
            }
            
            # Write to temp file first, then rename (atomic operation)
            temp_file = f"{self.config.checkpoint_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            os.rename(temp_file, self.config.checkpoint_file)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _check_memory_usage(self) -> float:
        """Monitor memory usage and trigger GC if needed."""
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        
        if memory_percent > self._stats['memory_peak']:
            self._stats['memory_peak'] = memory_percent
        
        if memory_percent > self.config.memory_threshold * 100:
            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
            gc.collect()
        
        return memory_percent
    
    def read_token_csv(self) -> Optional[pd.DataFrame]:
        """Read and validate the input CSV file."""
        try:
            self.logger.info(f"Reading CSV: {self.csv_file}")
            
            df = pd.read_csv(self.csv_file, dtype={CONTRACT_ADDRESS_COLUMN: str})
            
            if CONTRACT_ADDRESS_COLUMN not in df.columns:
                self.logger.error(f"Missing required column: '{CONTRACT_ADDRESS_COLUMN}'")
                return None
            
            # Generate token names if column is missing or empty
            if TOKEN_NAME_COLUMN not in df.columns or df[TOKEN_NAME_COLUMN].isna().all():
                self.logger.warning(f"Column '{TOKEN_NAME_COLUMN}' missing or empty, generating names")
                df[TOKEN_NAME_COLUMN] = df[CONTRACT_ADDRESS_COLUMN].apply(lambda x: f"Token_{x[:8]}")
            
            # Clean and validate addresses
            df = df.dropna(subset=[CONTRACT_ADDRESS_COLUMN])
            df[CONTRACT_ADDRESS_COLUMN] = df[CONTRACT_ADDRESS_COLUMN].str.lower().str.strip()
            df = df[df[CONTRACT_ADDRESS_COLUMN].str.match(r'^0x[a-fA-F0-9]{40}$')]
            df = df.drop_duplicates(subset=[CONTRACT_ADDRESS_COLUMN])
            
            self.logger.info(f"CSV loaded: {len(df)} valid tokens")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading CSV: {e}")
            return None
    
    def _check_existing_output(self, contract_address: str) -> Optional[str]:
        """Check if we already have output for this contract."""
        output_dir = Path(self.config.output_dir)
        
        for file_path in output_dir.glob(f"*_{contract_address}.csv"):
            if file_path.stat().st_size > 100:  # Ignore empty files
                return str(file_path)
        
        return None
    
    def check_existing_files(self, tokens_df: pd.DataFrame) -> Dict[str, Any]:
        """Scan output directory to find already-processed tokens."""
        self.logger.info("Scanning for existing output files...")
        
        output_dir = Path(self.config.output_dir)
        existing_contracts = {}
        
        # Single directory scan is much faster than per-token lookups
        if output_dir.exists():
            self.logger.info("Scanning output directory...")
            for file_path in output_dir.glob("*.csv"):
                try:
                    # Extract contract address from filename: TokenName_0xAddress.csv
                    filename = file_path.stem
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        possible_address = parts[-1]
                        if possible_address.startswith('0x') and len(possible_address) == 42:
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            if file_path.stat().st_size > 100:
                                existing_contracts[possible_address] = {
                                    'path': str(file_path),
                                    'size_mb': file_size_mb
                                }
                except Exception:
                    continue
        
        self.logger.info(f"Found {len(existing_contracts)} existing output files")
        
        existing_files = []
        missing_files = []
        
        for _, row in tokens_df.iterrows():
            contract_address = row[CONTRACT_ADDRESS_COLUMN]
            token_name = row[TOKEN_NAME_COLUMN]
            
            if contract_address in existing_contracts:
                info = existing_contracts[contract_address]
                existing_files.append({
                    'token_name': token_name,
                    'contract_address': contract_address,
                    'file_path': info['path'],
                    'file_size_mb': info['size_mb']
                })
            else:
                missing_files.append({
                    'token_name': token_name,
                    'contract_address': contract_address
                })
        
        stats = {
            'total_tokens': len(tokens_df),
            'existing_files': len(existing_files),
            'missing_files': len(missing_files),
            'completion_percentage': (len(existing_files) / len(tokens_df) * 100) if len(tokens_df) > 0 else 0,
            'existing_files_list': existing_files,
            'missing_files_list': missing_files
        }
        
        self.logger.info(f"Already processed: {stats['existing_files']}/{stats['total_tokens']}")
        self.logger.info(f"Remaining: {stats['missing_files']}")
        
        return stats
    
    def ask_user_continue(self, stats: Dict[str, Any]) -> Tuple[bool, bool]:
        """Prompt user for how to proceed with extraction."""
        print(f"\n{'='*60}")
        print("EXTRACTION OPTIONS")
        print(f"{'='*60}")
        print(f"Progress: {stats['completion_percentage']:.1f}% complete")
        print(f"Already done: {stats['existing_files']}")
        print(f"Remaining: {stats['missing_files']}")
        print("\n1. Process ALL tokens (including already done)")
        print("2. Process ONLY missing tokens")
        print("3. Cancel")
        
        while True:
            try:
                choice = input("\nSelect option (1-3): ").strip()
                
                if choice == "1":
                    return True, False
                elif choice == "2":
                    return True, True
                elif choice == "3":
                    return False, False
                else:
                    print("Invalid choice, try again")
            except KeyboardInterrupt:
                return False, False
    
    def process_single_token(self, token_name: str, contract_address: str) -> TokenResult:
        """Extract all transfers for a single token contract."""
        start_time = time.time()
        result = TokenResult(
            token_name=token_name,
            contract_address=contract_address,
            status="PROCESSING"
        )
        
        try:
            # Skip if we already have this one
            existing_file = self._check_existing_output(contract_address)
            if existing_file:
                result.output_file = existing_file
                result.status = "SKIPPED"
                result.processed_at = datetime.now().isoformat()
                return result
            
            # Sanitize token name for filesystem
            safe_name = "".join(c for c in token_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(" ", "_")[:50]
            output_file = f"{self.config.output_dir}/{safe_name}_{contract_address}.csv"
            
            # Call the tracker to fetch all transactions
            total_transfers = self.tracker.track_all_transactions_complete(
                contract_address,
                output_file,
                max_transfers=self.config.max_transfers
            )
            
            # Parse the output file to get detailed stats
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                result.erc20_transfers = len(df[df['transaction_type'] == 'ERC20_TRANSFER'])
                result.eth_transfers = len(df[df['transaction_type'] == 'ETH_NATIVE'])
                result.internal_transfers = len(df[df['transaction_type'] == 'ETH_INTERNAL'])
                result.total_transfers = len(df)
                result.file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
                del df
                gc.collect()
            
            result.output_file = output_file
            result.status = "SUCCESS"
            result.processed_at = datetime.now().isoformat()
            
            # Update checkpoint
            if self.config.enable_checkpoint:
                self.processed_contracts.add(contract_address)
                self._save_checkpoint()
            
            self.logger.info(f"Done: {token_name} - {result.total_transfers} transfers")
            
        except Exception as e:
            result.status = "ERROR"
            result.error = str(e)
            result.processed_at = datetime.now().isoformat()
            self.logger.error(f"Failed: {token_name} - {e}")
        
        finally:
            result.processing_time = time.time() - start_time
            self._check_memory_usage()
        
        return result
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff on failure."""
        for attempt in range(self.config.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise e
                wait_time = self.config.retry_delay * (2 ** attempt)
                self.logger.warning(f"Retry {attempt+1}/{self.config.retry_attempts}, waiting {wait_time}s")
                time.sleep(wait_time)
    
    def process_tokens_parallel(self, tokens_df: pd.DataFrame, skip_existing: bool = False) -> List[TokenResult]:
        """Process multiple tokens in parallel using thread pool."""
        self.logger.info(f"Starting parallel processing with {self.config.max_workers} workers")
        
        # Filter out already-checkpointed tokens
        if self.config.enable_checkpoint and self.processed_contracts:
            before = len(tokens_df)
            tokens_df = tokens_df[~tokens_df[CONTRACT_ADDRESS_COLUMN].isin(self.processed_contracts)]
            skipped = before - len(tokens_df)
            if skipped > 0:
                self.logger.info(f"Skipping {skipped} tokens (from checkpoint)")
        
        # Filter out tokens with existing output files
        if skip_existing:
            stats = self.check_existing_files(tokens_df)
            missing = [f['contract_address'] for f in stats['missing_files_list']]
            tokens_df = tokens_df[tokens_df[CONTRACT_ADDRESS_COLUMN].isin(missing)]
            self.logger.info(f"Processing {len(tokens_df)} missing tokens only")
        
        if len(tokens_df) == 0:
            self.logger.info("Nothing to process - all tokens complete!")
            return []
        
        token_list = [
            (row[TOKEN_NAME_COLUMN], row[CONTRACT_ADDRESS_COLUMN])
            for _, row in tokens_df.iterrows()
        ]
        
        results = []
        self._stats['start_time'] = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_token = {
                executor.submit(
                    self._retry_with_backoff,
                    self.process_single_token,
                    token_name,
                    contract_address
                ): (token_name, contract_address)
                for token_name, contract_address in token_list
            }
            
            for future in as_completed(future_to_token):
                token_name, contract_address = future_to_token[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    self._stats['total_processed'] += 1
                    if result.status == "SUCCESS":
                        self._stats['successful'] += 1
                    elif result.status == "SKIPPED":
                        self._stats['skipped'] += 1
                    else:
                        self._stats['failed'] += 1
                    
                    # Log progress every 5 tokens
                    if self._stats['total_processed'] % 5 == 0:
                        self._log_progress(len(token_list))
                    
                    if self.config.delay_between_tokens > 0:
                        time.sleep(self.config.delay_between_tokens)
                    
                except Exception as e:
                    self.logger.error(f"Worker error for {token_name}: {e}")
                    error_result = TokenResult(
                        token_name=token_name,
                        contract_address=contract_address,
                        status="ERROR",
                        error=str(e),
                        processed_at=datetime.now().isoformat()
                    )
                    results.append(error_result)
                    self._stats['failed'] += 1
        
        return results
    
    def _log_progress(self, total_tokens: int):
        """Log current progress with ETA estimate."""
        elapsed = time.time() - self._stats['start_time']
        processed = self._stats['total_processed']
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total_tokens - processed) / rate if rate > 0 else 0
        
        self.logger.info(
            f"Progress: {processed}/{total_tokens} ({processed/total_tokens*100:.1f}%) - "
            f"ETA: {eta/60:.1f} min"
        )
    
    def save_summary(self, results: List[TokenResult]):
        """Write extraction summary to CSV."""
        try:
            summary_file = f"{self.config.output_dir}/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            data = []
            for result in results:
                data.append({
                    'token_name': result.token_name,
                    'contract_address': result.contract_address,
                    'status': result.status,
                    'total_transfers': result.total_transfers,
                    'erc20_transfers': result.erc20_transfers,
                    'eth_transfers': result.eth_transfers,
                    'internal_transfers': result.internal_transfers,
                    'output_file': result.output_file,
                    'file_size_mb': result.file_size_mb,
                    'processing_time': result.processing_time,
                    'error': result.error
                })
            
            df = pd.DataFrame(data)
            df.to_csv(summary_file, index=False)
            self.logger.info(f"Summary saved: {summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")
    
    def show_final_stats(self, results: List[TokenResult]):
        """Display final extraction statistics."""
        if not results:
            return
        
        total_time = time.time() - self._stats['start_time']
        successful = [r for r in results if r.status == "SUCCESS"]
        
        self.logger.info("=" * 60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total tokens: {len(results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Skipped: {self._stats['skipped']}")
        self.logger.info(f"Failed: {self._stats['failed']}")
        self.logger.info(f"Total time: {total_time/60:.1f} min")
        self.logger.info(f"Peak memory: {self._stats['memory_peak']:.1f}%")
        
        if successful:
            total = sum(r.total_transfers for r in successful)
            self.logger.info(f"Total transfers extracted: {total:,}")
        
        self.logger.info("=" * 60)
    
    def extract_all_tokens(self, auto_continue: bool = False, skip_existing: bool = True) -> List[TokenResult]:
        """Main entry point - extract transfers for all tokens in the CSV."""
        self.logger.info("Starting extraction job")
        
        df = self.read_token_csv()
        if df is None:
            return []
        
        stats = self.check_existing_files(df)
        
        if not auto_continue:
            should_continue, skip_existing = self.ask_user_continue(stats)
            if not should_continue:
                return []
        
        results = self.process_tokens_parallel(df, skip_existing=skip_existing)
        
        if results:
            self.save_summary(results)
        
        self.show_final_stats(results)
        
        return results


# ========================================
# Main Entry Point
# ========================================

if __name__ == "__main__":
    # Etherscan API keys (rotate to avoid rate limits)
    API_KEYS = [
    ]

    config = Config(
        max_workers=len(API_KEYS),  # One worker per API key
        delay_between_tokens=0.1,
        max_transfers=500,
        enable_checkpoint=True
    )
    
    extractor = TokenTransferExtractor(API_KEYS, INPUT_CSV_FILE, config)
    
    print("\n" + "=" * 60)
    print("TOKEN TRANSFER EXTRACTION SYSTEM")
    print("=" * 60)
    print(f"Input CSV: {INPUT_CSV_FILE}")
    print(f"Output directory: {extractor.config.output_dir}")
    print(f"Checkpoint: {'enabled' if config.enable_checkpoint else 'disabled'}")
    print("=" * 60)
    
    print("\n1. Interactive mode (will prompt for options)")
    print("2. Auto mode - process missing tokens only")
    print("3. Auto mode - process all tokens")
    
    try:
        choice = input("\nSelect mode (1-3): ").strip()
        
        if choice == "1":
            results = extractor.extract_all_tokens(auto_continue=False)
        elif choice == "2":
            results = extractor.extract_all_tokens(auto_continue=True, skip_existing=True)
        elif choice == "3":
            results = extractor.extract_all_tokens(auto_continue=True, skip_existing=False)
        else:
            print("Invalid choice, using interactive mode")
            results = extractor.extract_all_tokens(auto_continue=False)
    
    except KeyboardInterrupt:
        print("\nCancelled by user")
        results = []
    
    if results:
        print(f"\nDone! Processed {len(results)} tokens")
    else:
        print("\nNo tokens were processed")
