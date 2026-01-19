#!/usr/bin/env python3
"""
Ethereum Transaction Tracker

Fetches ERC20 token transfers and ETH transactions from Etherscan API v2.
Optimized for memory efficiency via streaming to CSV.

Features:
- API key rotation to avoid rate limits
- Streaming output to minimize memory footprint
- Pagination support for large transaction histories
"""

import requests
import time
import csv
import logging
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any
from pathlib import Path
import gc


class CompleteEthereumTransactionTracker:
    """
    Extracts all transaction types for a given Ethereum address/contract:
    - ERC20 token transfers
    - Native ETH transfers
    - Internal ETH transfers
    
    Writes directly to CSV to keep memory usage low.
    """
    
    def __init__(self, api_keys: List[str]):
        """
        Initialize the tracker with Etherscan API keys.
        
        Args:
            api_keys: List of Etherscan API keys for rotation
        """
        if isinstance(api_keys, str):
            self.api_keys = [api_keys]
        else:
            self.api_keys = api_keys
        
        self.current_api_index = 0
        self.failed_apis = set()
        
        # Track API usage for debugging
        self.api_stats = {
            'calls_per_api': {key: 0 for key in self.api_keys},
            'errors_per_api': {key: 0 for key in self.api_keys}
        }
        
        # Etherscan API v2 configuration
        self.base_url = "https://api.etherscan.io/v2/api"
        self.chain_id = 1  # Ethereum mainnet
        self.rate_limit_delay = 0.22  # ~4.5 requests/sec per key (safe margin)
        self.last_request_time = {key: 0 for key in self.api_keys}
        
        # Thread safety for API key rotation
        import threading
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger('EthereumTracker')
    
    def _get_next_api_key(self) -> Optional[str]:
        """Get the next available API key using round-robin rotation."""
        with self.lock:
            available_keys = [k for k in self.api_keys if k not in self.failed_apis]
            
            if not available_keys:
                self.logger.error("All API keys have failed!")
                return None
            
            api_key = available_keys[self.current_api_index % len(available_keys)]
            self.current_api_index = (self.current_api_index + 1) % len(available_keys)
            
            return api_key
    
    def _rate_limit(self, api_key: str):
        """Enforce rate limiting per API key."""
        current_time = time.time()
        last_time = self.last_request_time.get(api_key, 0)
        time_since_last = current_time - last_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time[api_key] = time.time()
    
    def _make_api_request(self, params: Dict[str, Any], retry_count: int = 3) -> Optional[Dict]:
        """
        Execute API request with automatic retry and key rotation.
        
        Args:
            params: Request parameters
            retry_count: Number of retry attempts
            
        Returns:
            JSON response or None on failure
        """
        for attempt in range(retry_count):
            api_key = self._get_next_api_key()
            
            if not api_key:
                return None
            
            try:
                self._rate_limit(api_key)
                
                # Add API key and chain ID (required for v2)
                params['apikey'] = api_key
                params['chainid'] = self.chain_id
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                self.api_stats['calls_per_api'][api_key] += 1
                
                # Check for valid response
                if data.get('status') == '1' or data.get('message') == 'No transactions found':
                    return data
                
                error_message = data.get('message', '')
                
                if 'rate limit' in error_message.lower():
                    self.logger.warning(f"Rate limit hit for key {api_key[:8]}...")
                    time.sleep(2)
                    continue
                
                if 'invalid api key' in error_message.lower():
                    self.logger.error(f"Invalid API key: {api_key[:8]}...")
                    self.failed_apis.add(api_key)
                    continue
                
                self.logger.warning(f"API Error: {error_message}")
                self.api_stats['errors_per_api'][api_key] += 1
                
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                return data
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}/{retry_count}): {e}")
                self.api_stats['errors_per_api'][api_key] += 1
                
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
            
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                self.api_stats['errors_per_api'][api_key] += 1
                
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        return None
    
    def _timestamp_to_datetime(self, timestamp: str) -> str:
        """Convert Unix timestamp to human-readable datetime string."""
        try:
            dt = datetime.fromtimestamp(int(timestamp))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp
    
    def _wei_to_ether(self, wei_value: str, decimals: int = 18) -> str:
        """Convert Wei to Ether (or token units with custom decimals)."""
        try:
            value = int(wei_value) / (10 ** decimals)
            return f"{value:.18f}".rstrip('0').rstrip('.')
        except:
            return wei_value
    
    def get_erc20_transfers(self, contract_address: str, max_transfers: int = None) -> Generator[Dict, None, None]:
        """
        Fetch all ERC20 token transfers for a contract.
        
        Uses a generator to stream results without loading everything into memory.
        
        Args:
            contract_address: The token contract address
            max_transfers: Optional limit on number of transfers to fetch
            
        Yields:
            Dict containing transfer data
        """
        page = 1
        offset = 1000  # Max results per page
        transfers_count = 0
        
        self.logger.info(f"Fetching ERC20 transfers for {contract_address}")
        if max_transfers:
            self.logger.info(f"  Limit set: {max_transfers} transfers")
        
        while True:
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contract_address,
                'page': page,
                'offset': offset,
                'sort': 'asc'
            }
            
            response = self._make_api_request(params)
            
            if not response:
                self.logger.error(f"Failed to fetch ERC20 transfers (page {page})")
                break
            
            result = response.get('result', [])
            
            if not result or response.get('message') == 'No transactions found':
                if page == 1:
                    self.logger.info("No ERC20 transfers found")
                break
            
            # Handle string response (error message)
            if isinstance(result, str):
                if page == 1:
                    self.logger.warning(f"API returned: {result}")
                break
            
            for tx in result:
                if max_transfers and transfers_count >= max_transfers:
                    self.logger.info(f"Reached limit: {max_transfers} ERC20 transfers")
                    return
                
                try:
                    decimals = int(tx.get('tokenDecimal', '18'))
                    
                    transfer_data = {
                        'transaction_hash': tx.get('hash', ''),
                        'timestamp': self._timestamp_to_datetime(tx.get('timeStamp', '')),
                        'from_address': tx.get('from', '').lower(),
                        'to_address': tx.get('to', '').lower(),
                        'value': self._wei_to_ether(tx.get('value', '0'), decimals),
                        'token_symbol': tx.get('tokenSymbol', ''),
                        'token_name': tx.get('tokenName', ''),
                        'gas_used': tx.get('gasUsed', ''),
                        'gas_price': self._wei_to_ether(tx.get('gasPrice', '0'), 9),  # Gwei
                        'transaction_type': 'ERC20_TRANSFER'
                    }
                    
                    yield transfer_data
                    transfers_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error parsing ERC20 transfer: {e}")
                    continue
            
            # If we got fewer results than requested, we're done
            if len(result) < offset:
                break
            
            page += 1
            self.logger.debug(f"Processed page {page - 1}, continuing...")
    
    def get_eth_transfers(self, address: str, max_transfers: int = None) -> Generator[Dict, None, None]:
        """
        Fetch native ETH transfers for an address.
        
        Args:
            address: Ethereum address
            max_transfers: Optional limit on number of transfers
            
        Yields:
            Dict containing transfer data
        """
        page = 1
        offset = 1000
        transfers_count = 0
        
        self.logger.info(f"Fetching native ETH transfers for {address}")
        if max_transfers:
            self.logger.info(f"  Limit set: {max_transfers} transfers")
        
        while True:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': address,
                'page': page,
                'offset': offset,
                'sort': 'asc'
            }
            
            response = self._make_api_request(params)
            
            if not response:
                self.logger.error(f"Failed to fetch ETH transfers (page {page})")
                break
            
            result = response.get('result', [])
            
            if not result or response.get('message') == 'No transactions found':
                if page == 1:
                    self.logger.info("No native ETH transfers found")
                break
            
            if isinstance(result, str):
                if page == 1:
                    self.logger.warning(f"API returned: {result}")
                break
            
            for tx in result:
                if max_transfers and transfers_count >= max_transfers:
                    self.logger.info(f"Reached limit: {max_transfers} ETH transfers")
                    return
                
                try:
                    # Only include transactions with non-zero ETH value
                    if int(tx.get('value', '0')) > 0:
                        transfer_data = {
                            'transaction_hash': tx.get('hash', ''),
                            'timestamp': self._timestamp_to_datetime(tx.get('timeStamp', '')),
                            'from_address': tx.get('from', '').lower(),
                            'to_address': tx.get('to', '').lower(),
                            'value': self._wei_to_ether(tx.get('value', '0'), 18),
                            'token_symbol': 'ETH',
                            'token_name': 'Ethereum',
                            'gas_used': tx.get('gasUsed', ''),
                            'gas_price': self._wei_to_ether(tx.get('gasPrice', '0'), 9),
                            'transaction_type': 'ETH_NATIVE'
                        }
                        
                        yield transfer_data
                        transfers_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error parsing ETH transfer: {e}")
                    continue
            
            if len(result) < offset:
                break
            
            page += 1
    
    def get_internal_transfers(self, address: str, max_transfers: int = None) -> Generator[Dict, None, None]:
        """
        Fetch internal ETH transfers for an address.
        
        Internal transfers are ETH movements that happen within smart contract execution.
        
        Args:
            address: Ethereum address
            max_transfers: Optional limit on number of transfers
            
        Yields:
            Dict containing transfer data
        """
        page = 1
        offset = 1000
        transfers_count = 0
        
        self.logger.info(f"Fetching internal ETH transfers for {address}")
        if max_transfers:
            self.logger.info(f"  Limit set: {max_transfers} transfers")
        
        while True:
            params = {
                'module': 'account',
                'action': 'txlistinternal',
                'address': address,
                'page': page,
                'offset': offset,
                'sort': 'asc'
            }
            
            response = self._make_api_request(params)
            
            if not response:
                self.logger.error(f"Failed to fetch internal transfers (page {page})")
                break
            
            result = response.get('result', [])
            
            if not result or response.get('message') == 'No transactions found':
                if page == 1:
                    self.logger.info("No internal ETH transfers found")
                break
            
            if isinstance(result, str):
                if page == 1:
                    self.logger.warning(f"API returned: {result}")
                break
            
            for tx in result:
                if max_transfers and transfers_count >= max_transfers:
                    self.logger.info(f"Reached limit: {max_transfers} internal transfers")
                    return
                
                try:
                    if int(tx.get('value', '0')) > 0:
                        transfer_data = {
                            'transaction_hash': tx.get('hash', ''),
                            'timestamp': self._timestamp_to_datetime(tx.get('timeStamp', '')),
                            'from_address': tx.get('from', '').lower(),
                            'to_address': tx.get('to', '').lower(),
                            'value': self._wei_to_ether(tx.get('value', '0'), 18),
                            'token_symbol': 'ETH',
                            'token_name': 'Ethereum',
                            'gas_used': tx.get('gas', ''),
                            'gas_price': tx.get('gasUsed', ''),  # Internal txs have different fields
                            'transaction_type': 'ETH_INTERNAL'
                        }
                        
                        yield transfer_data
                        transfers_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error parsing internal transfer: {e}")
                    continue
            
            if len(result) < offset:
                break
            
            page += 1
    
    def track_all_transactions_complete(self, contract_address: str, output_file: str, max_transfers: int = None) -> int:
        """
        Extract all transaction types and write directly to CSV.
        
        This is the main entry point for full extraction. Streams data to disk
        to minimize memory usage.
        
        Args:
            contract_address: Contract address to track
            output_file: Path for output CSV file
            max_transfers: Optional cap on total transfers (useful for testing)
            
        Returns:
            Total number of transfers extracted
        """
        self.logger.info(f"Starting full extraction for {contract_address}")
        self.logger.info(f"Output file: {output_file}")
        if max_transfers:
            self.logger.info(f"Limited mode: max {max_transfers} total transfers")
        
        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        total_transfers = 0
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'transaction_hash',
                'timestamp',
                'from_address',
                'to_address',
                'value',
                'token_symbol',
                'token_name',
                'gas_used',
                'gas_price',
                'transaction_type'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Phase 1: ERC20 transfers
            self.logger.info("Phase 1: Fetching ERC20 transfers...")
            erc20_count = 0
            remaining = max_transfers if max_transfers else float('inf')
            
            for transfer in self.get_erc20_transfers(contract_address, max_transfers=int(remaining) if remaining != float('inf') else None):
                writer.writerow(transfer)
                erc20_count += 1
                
                if erc20_count % 100 == 0:
                    self.logger.debug(f"  Written {erc20_count} ERC20 transfers...")
                
                if max_transfers:
                    remaining = max_transfers - erc20_count
                    if remaining <= 0:
                        break
            
            self.logger.info(f"ERC20 transfers: {erc20_count}")
            total_transfers += erc20_count
            
            gc.collect()
            
            # Check if we hit the limit
            if max_transfers and total_transfers >= max_transfers:
                self.logger.info(f"Reached total limit: {max_transfers} transfers")
                self.logger.info(f"Extraction complete! Total: {total_transfers}")
                self.logger.info(f"Breakdown: ERC20={erc20_count}, ETH=0, Internal=0")
                return total_transfers
            
            # Phase 2: Native ETH transfers
            self.logger.info("Phase 2: Fetching native ETH transfers...")
            eth_count = 0
            remaining = max_transfers - total_transfers if max_transfers else float('inf')
            
            for transfer in self.get_eth_transfers(contract_address, max_transfers=int(remaining) if remaining != float('inf') else None):
                writer.writerow(transfer)
                eth_count += 1
                
                if eth_count % 100 == 0:
                    self.logger.debug(f"  Written {eth_count} ETH transfers...")
                
                if max_transfers:
                    remaining = max_transfers - total_transfers - eth_count
                    if remaining <= 0:
                        break
            
            self.logger.info(f"Native ETH transfers: {eth_count}")
            total_transfers += eth_count
            
            gc.collect()
            
            if max_transfers and total_transfers >= max_transfers:
                self.logger.info(f"Reached total limit: {max_transfers} transfers")
                self.logger.info(f"Extraction complete! Total: {total_transfers}")
                self.logger.info(f"Breakdown: ERC20={erc20_count}, ETH={eth_count}, Internal=0")
                return total_transfers
            
            # Phase 3: Internal ETH transfers
            self.logger.info("Phase 3: Fetching internal ETH transfers...")
            internal_count = 0
            remaining = max_transfers - total_transfers if max_transfers else float('inf')
            
            for transfer in self.get_internal_transfers(contract_address, max_transfers=int(remaining) if remaining != float('inf') else None):
                writer.writerow(transfer)
                internal_count += 1
                
                if internal_count % 100 == 0:
                    self.logger.debug(f"  Written {internal_count} internal transfers...")
                
                if max_transfers:
                    remaining = max_transfers - total_transfers - internal_count
                    if remaining <= 0:
                        break
            
            self.logger.info(f"Internal ETH transfers: {internal_count}")
            total_transfers += internal_count
        
        self.logger.info(f"Extraction complete! Total: {total_transfers}")
        self.logger.info(f"Breakdown: ERC20={erc20_count}, ETH={eth_count}, Internal={internal_count}")
        
        return total_transfers
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Return API usage statistics for debugging."""
        return {
            'total_keys': len(self.api_keys),
            'failed_keys': len(self.failed_apis),
            'available_keys': len(self.api_keys) - len(self.failed_apis),
            'calls_per_api': self.api_stats['calls_per_api'],
            'errors_per_api': self.api_stats['errors_per_api']
        }
