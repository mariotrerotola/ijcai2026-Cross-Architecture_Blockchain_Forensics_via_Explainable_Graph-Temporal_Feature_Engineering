#!/usr/bin/env python3
"""
Feature Extraction for Token Transaction Networks

Extracts graph-based and statistical features from token transaction CSV files.
Features include network centralities, temporal patterns, and transaction statistics.

Input: CSV files in data/raw/{scam,licit}/ directories
Output: Consolidated feature dataset in data/dataset_with_features/

Supports parallel processing and automatic checkpointing for large datasets.
"""

import os
import sys
import glob
import json
import time
import argparse
import logging
import warnings
import gc
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import networkx as nx

warnings.filterwarnings('ignore')


class Config:
    """Configuration parameters for feature extraction."""
    
    def __init__(self, **kwargs):
        self.input_base_dir = kwargs.get('input_dir', 'data/raw')
        self.output_dir = kwargs.get('output_dir', 'data/dataset_with_features')
        self.max_workers = kwargs.get('workers', max(1, os.cpu_count() - 1))
        self.batch_size = kwargs.get('batch_size', 100)
        self.max_files = kwargs.get('max_files', None)
        self.log_level = kwargs.get('log_level', 'INFO')
        self.resume = kwargs.get('resume', True)
        self.memory_threshold = 0.85  # Trigger GC above this memory usage


def setup_logging(config: Config) -> logging.Logger:
    """Configure logging to file and console."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"feature_extraction_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('FeatureExtractor')


def get_memory_usage() -> float:
    """Return current memory usage as percentage."""
    return psutil.virtual_memory().percent


# ==========================================
# Feature Computation Functions
# ==========================================

def compute_node_centrality_stats(G, centrality_values: Dict) -> Dict[str, float]:
    """Compute summary statistics (mean, std, min, max) for centrality values."""
    if not centrality_values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    values = list(centrality_values.values())
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of numeric values."""
    if len(values) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def extract_features_from_file(csv_file: Path) -> Tuple[Optional[Dict], Optional[nx.DiGraph], Optional[pd.DataFrame]]:
    """
    Load transaction CSV and build the transaction graph.
    
    Returns the base features dict, the NetworkX graph, and the raw dataframe.
    """
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None, None, None
        
        # Clean up the data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
        df = df.sort_values('timestamp')
        
        features = {'token_file': csv_file.name}
        
        # Build directed graph from transactions
        G = nx.DiGraph()
        
        for _, row in df.iterrows():
            u, v = row['from_address'], row['to_address']
            value = float(row['value'])
            ts = row['timestamp']
            
            if G.has_edge(u, v):
                # Aggregate multiple transactions between same addresses
                data = G[u][v]
                data['weight'] += value
                data['count'] += 1
                data['timestamps'].append(ts)
                data['values'].append(value)
            else:
                G.add_edge(u, v, 
                          weight=value, 
                          count=1,
                          timestamps=[ts],
                          values=[value])
        
        features.update({
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_transactions': len(df)
        })
        
        return features, G, df
    
    except Exception as e:
        return None, None, None


def extract_node_centrality_features(G: nx.DiGraph, features: Dict) -> Dict:
    """
    Compute various centrality measures for nodes in the graph.
    
    Includes degree, betweenness, closeness, eigenvector, and Katz centralities.
    """
    if features['num_nodes'] == 0:
        return features
    
    # Standard centrality measures (fast)
    centrality_funcs = {
        'node_degree_centrality': nx.degree_centrality,
        'node_indegree_centrality': nx.in_degree_centrality,
        'node_outdegree_centrality': nx.out_degree_centrality,
    }
    
    for name, func in centrality_funcs.items():
        try:
            cent_map = func(G)
            stats = compute_node_centrality_stats(G, cent_map)
            for stat_name, val in stats.items():
                features[f'{name}_{stat_name}'] = val
        except Exception:
            for stat in ['mean', 'std', 'min', 'max']:
                features[f'{name}_{stat}'] = 0.0

    # More expensive centrality measures (with fallbacks)
    expensive_metrics = [
        ('node_betweenness_centrality', lambda g: nx.betweenness_centrality(g, k=min(100, len(g)))),
        ('node_closeness_centrality', lambda g: nx.closeness_centrality(g)),
        ('node_eigenvector_centrality', lambda g: nx.eigenvector_centrality(g, max_iter=1000, tol=1e-06)),
        ('node_katz_centrality', lambda g: nx.katz_centrality(g, max_iter=1000, tol=1e-06)),
        ('node_clustering_coefficient', lambda g: nx.clustering(g.to_undirected()))
    ]
    
    for name, func in expensive_metrics:
        try:
            val_map = func(G)
            stats = compute_node_centrality_stats(G, val_map)
            for stat_name, val in stats.items():
                features[f'{name}_{stat_name}'] = val
        except Exception:
            # Some metrics may fail on certain graph structures
            for stat in ['mean', 'std', 'min', 'max']:
                features[f'{name}_{stat}'] = 0.0
                
    return features


def extract_node_temporal_features(G: nx.DiGraph, df: pd.DataFrame, features: Dict) -> Dict:
    """
    Extract temporal features based on long-term vs short-term activity.
    
    Splits activity at the median timestamp and computes incoming/outgoing
    statistics for each period.
    """
    if df.empty:
        return features
        
    median_timestamp = df['timestamp'].median()
    
    temporal_data = {
        'incoming_count_long': [], 'incoming_val_long': [],
        'incoming_count_short': [], 'incoming_val_short': [],
        'outgoing_count_long': [], 'outgoing_val_long': [],
        'outgoing_count_short': [], 'outgoing_val_short': []
    }
    
    for node in G.nodes():
        # Analyze incoming edges
        in_edges = G.in_edges(node, data=True)
        long_in = [d for _, _, d in in_edges if d['timestamps'][-1] < median_timestamp]
        short_in = [d for _, _, d in in_edges if d['timestamps'][-1] >= median_timestamp]
        
        temporal_data['incoming_count_long'].append(len(long_in))
        temporal_data['incoming_count_short'].append(len(short_in))
        
        val_long = np.mean([d['weight'] for d in long_in]) if long_in else 0
        val_short = np.mean([d['weight'] for d in short_in]) if short_in else 0
        temporal_data['incoming_val_long'].append(val_long)
        temporal_data['incoming_val_short'].append(val_short)
        
        # Analyze outgoing edges
        out_edges = G.out_edges(node, data=True)
        long_out = [d for _, _, d in out_edges if d['timestamps'][-1] < median_timestamp]
        short_out = [d for _, _, d in out_edges if d['timestamps'][-1] >= median_timestamp]
        
        temporal_data['outgoing_count_long'].append(len(long_out))
        temporal_data['outgoing_count_short'].append(len(short_out))
        
        val_long = np.mean([d['weight'] for d in long_out]) if long_out else 0
        val_short = np.mean([d['weight'] for d in short_out]) if short_out else 0
        temporal_data['outgoing_val_long'].append(val_long)
        temporal_data['outgoing_val_short'].append(val_short)
        
    # Map internal names to final feature names
    feature_mapping = {
        'incoming_count_long': 'node_long_term_incoming_count',
        'incoming_val_long': 'node_long_term_incoming_avg_value',
        'incoming_count_short': 'node_short_term_incoming_count',
        'incoming_val_short': 'node_short_term_incoming_avg_value',
        'outgoing_count_long': 'node_long_term_outgoing_count',
        'outgoing_val_long': 'node_long_term_outgoing_avg_value',
        'outgoing_count_short': 'node_short_term_outgoing_count',
        'outgoing_val_short': 'node_short_term_outgoing_avg_value',
    }
    
    for key, values in temporal_data.items():
        base_name = feature_mapping[key]
        stats = compute_stats(values)
        for stat, val in stats.items():
            features[f'{base_name}_{stat}'] = val
            
    return features


def extract_edge_features(G: nx.DiGraph, df: pd.DataFrame, features: Dict) -> Dict:
    """
    Extract features from edges (transaction flows between addresses).
    
    Includes value metrics, accumulation patterns, and transfer frequency.
    """
    if len(df) == 0:
        return features
        
    total_val = df['value'].sum()
    median_timestamp = df['timestamp'].median()
    max_timestamp = df['timestamp'].max()
    
    edge_metrics = {
        'val': [], 'val_norm': [], 'val_log': [], 'val_harm': [],
        'accum_in_val': [], 'accum_in_count': [],
        'accum_out_val': [], 'accum_out_count': [],
        'ft_in_long': [], 'ft_in_short': [],
        'ft_out_long': [], 'ft_out_short': [],
        'freq': [], 'recency': []
    }
    
    for u, v, data in G.edges(data=True):
        w = data['weight']
        edge_metrics['val'].append(w)
        edge_metrics['val_norm'].append(w / total_val if total_val > 0 else 0)
        edge_metrics['val_log'].append(np.log(w + 1) if w > 0 else 0)
        edge_metrics['val_harm'].append(1 / (w + 1) if w > 0 else 0)
        
        # Accumulation at source and destination
        in_v_val = sum(d['weight'] for _, _, d in G.in_edges(v, data=True))
        out_u_val = sum(d['weight'] for _, _, d in G.out_edges(u, data=True))
        
        edge_metrics['accum_in_val'].append(in_v_val)
        edge_metrics['accum_in_count'].append(G.in_degree(v))
        edge_metrics['accum_out_val'].append(out_u_val)
        edge_metrics['accum_out_count'].append(G.out_degree(u))
        
        edge_metrics['freq'].append(data['count'])
        
        # Days since last transaction on this edge
        recency = (max_timestamp - data['timestamps'][-1]).total_seconds() / 86400
        edge_metrics['recency'].append(recency)
        
        # Long-term vs short-term frequency
        ts_long = [t for t in data['timestamps'] if t < median_timestamp]
        ts_short = [t for t in data['timestamps'] if t >= median_timestamp]
        c_long = len(ts_long)
        c_short = len(ts_short)
        
        edge_metrics['ft_in_long'].append(c_long)
        edge_metrics['ft_in_short'].append(c_short)
        edge_metrics['ft_out_long'].append(c_long)
        edge_metrics['ft_out_short'].append(c_short)

    # Map to final feature names
    map_names = {
        'val': 'edge_transfer_value',
        'val_norm': 'edge_transfer_value_normalized',
        'val_log': 'edge_transfer_value_log',
        'val_harm': 'edge_harmonic_transfer_value',
        'accum_in_val': 'edge_accumulated_incoming_value',
        'accum_in_count': 'edge_accumulated_incoming_count',
        'accum_out_val': 'edge_accumulated_outgoing_value',
        'accum_out_count': 'edge_accumulated_outgoing_count',
        'ft_in_long': 'edge_long_term_accum_incoming_freq',
        'ft_in_short': 'edge_short_term_accum_incoming_freq',
        'ft_out_long': 'edge_long_term_accum_outgoing_freq',
        'ft_out_short': 'edge_short_term_accum_outgoing_freq',
        'freq': 'edge_transfer_frequency',
        'recency': 'edge_transfer_recency'
    }

    for key, values in edge_metrics.items():
        base = map_names[key]
        stats = compute_stats(values)
        for stat, val in stats.items():
            features[f'{base}_{stat}'] = val
            
    return features


def extract_transaction_features(df: pd.DataFrame, features: Dict) -> Dict:
    """
    Extract aggregate transaction statistics.
    
    Includes volume metrics, address counts, timing patterns, and value distribution.
    """
    if len(df) == 0:
        return features
        
    # Basic transaction stats
    features['total_transactions'] = len(df)
    features['total_volume'] = df['value'].sum()
    features['avg_transaction_value'] = df['value'].mean()
    features['std_transaction_value'] = df['value'].std()
    features['min_transaction_value'] = df['value'].min()
    features['max_transaction_value'] = df['value'].max()
    features['median_transaction_value'] = df['value'].median()
    features['q25_transaction_value'] = df['value'].quantile(0.25)
    features['q75_transaction_value'] = df['value'].quantile(0.75)
    features['q95_transaction_value'] = df['value'].quantile(0.95)
    
    # Address statistics
    n_senders = df['from_address'].nunique()
    n_receivers = df['to_address'].nunique()
    all_addrs = set(df['from_address']).union(set(df['to_address']))
    n_unique = len(all_addrs)
    
    features['unique_senders'] = n_senders
    features['unique_receivers'] = n_receivers
    features['unique_addresses'] = n_unique
    
    if n_unique > 0:
        features['sender_ratio'] = n_senders / n_unique
        features['receiver_ratio'] = n_receivers / n_unique
        features['tx_per_unique_address'] = len(df) / n_unique
        features['avg_volume_per_address'] = features['total_volume'] / n_unique
    else:
        features['sender_ratio'] = 0
        features['receiver_ratio'] = 0
        features['tx_per_unique_address'] = 0
        features['avg_volume_per_address'] = 0
        
    # Timing statistics
    max_ts = df['timestamp'].max()
    min_ts = df['timestamp'].min()
    features['time_span'] = (max_ts - min_ts).total_seconds() / 86400  # days
    
    if len(df) > 1:
        diffs = df['timestamp'].diff().dt.total_seconds().dropna()
        features['avg_time_between_tx'] = diffs.mean()
        features['std_time_between_tx'] = diffs.std()
        features['min_time_between_tx'] = diffs.min()
        features['max_time_between_tx'] = diffs.max()
    else:
        features.update({k: 0 for k in ['avg_time_between_tx', 'std_time_between_tx', 
                                      'min_time_between_tx', 'max_time_between_tx']})
                                      
    # Value concentration (Gini coefficient)
    vals = np.sort(df['value'].values)
    n = len(vals)
    if n > 0 and vals.sum() > 0:
        cumsum = np.cumsum(vals)
        features['value_concentration'] = (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n
    else:
        features['value_concentration'] = 0
        
    return features


def determine_label(filepath: str) -> str:
    """Determine label (scam/licit) based on parent directory name."""
    parent = Path(filepath).parent.name.lower()
    if 'scam' in parent:
        return 'scam'
    return 'licit'


def process_file_wrapper(args):
    """
    Wrapper function for parallel processing.
    
    Runs the full feature extraction pipeline on a single file.
    """
    file_path, label = args
    path_obj = Path(file_path)
    
    try:
        base_feats, G, df = extract_features_from_file(path_obj)
        if base_feats is None:
            return None
            
        feats = extract_node_centrality_features(G, base_feats)
        feats = extract_node_temporal_features(G, df, feats)
        feats = extract_edge_features(G, df, feats)
        feats = extract_transaction_features(df, feats)
        
        feats['label'] = label
        feats['source_directory'] = path_obj.parent.name
        
        # Free memory
        del G, df
        return feats
        
    except Exception as e:
        return None


# ==========================================
# Main Application
# ==========================================

class FeatureExtractorApplication:
    """Orchestrates the feature extraction pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.processed_files = set()
        self.columns = []
        
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = Path(self.config.output_dir) / 'checkpoint.json'
        self.output_csv = Path(self.config.output_dir) / 'features_dataset.csv'
        
    def load_checkpoint(self):
        """Resume from previous run if checkpoint exists."""
        if self.config.resume and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
                self.logger.info(f"Checkpoint loaded: {len(self.processed_files)} files already done")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")

    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({'processed_files': list(self.processed_files)}, f)
        except Exception as e:
            self.logger.error(f"Could not save checkpoint: {e}")

    def scan_files(self) -> List[Tuple[str, str]]:
        """Find all transaction CSV files in the input directory."""
        self.logger.info(f"Scanning {self.config.input_base_dir}...")
        
        all_files = []
        search_pattern = os.path.join(self.config.input_base_dir, "**", "*.csv")
        
        for file_path in glob.glob(search_pattern, recursive=True):
            # Skip summary/dataset files
            if 'summary' in file_path or 'dataset' in file_path:
                continue
                
            abs_path = os.path.abspath(file_path)
            if abs_path in self.processed_files:
                continue
                
            label = determine_label(file_path)
            all_files.append((abs_path, label))
            
        self.logger.info(f"Found {len(all_files)} files to process")
        
        if self.config.max_files:
            all_files = all_files[:self.config.max_files]
            self.logger.info(f"Limited to {len(all_files)} files")
            
        return all_files

    def run(self):
        """Execute the feature extraction pipeline."""
        self.logger.info("Starting feature extraction")
        self.load_checkpoint()
        
        files_to_process = self.scan_files()
        total_files = len(files_to_process)
        
        if total_files == 0:
            self.logger.info("No new files to process")
            return

        batch_results = []
        start_time = time.time()
        processed_count = 0
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(process_file_wrapper, f): f[0] 
                for f in files_to_process
            }
            
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        
                    self.processed_files.add(file_path)
                    processed_count += 1
                    
                    # Check memory and trigger GC if needed
                    if get_memory_usage() > self.config.memory_threshold * 100:
                        gc.collect()
                    
                    # Save progress periodically
                    if len(batch_results) >= self.config.batch_size:
                        self.save_batch(batch_results)
                        self.save_checkpoint()
                        batch_results = []
                        
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        eta = (total_files - processed_count) / rate / 60
                        self.logger.info(f"Progress: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%) | ETA: {eta:.1f} min")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")

        # Save any remaining results
        if batch_results:
            self.save_batch(batch_results)
            self.save_checkpoint()
            
        self.logger.info("Feature extraction complete")

    def save_batch(self, batch: List[Dict], write_header: bool = False):
        """Write a batch of results to CSV files, grouped by source directory."""
        if not batch:
            return
            
        df = pd.DataFrame(batch)
        
        # Group results by source directory
        for source_dir, group_df in df.groupby('source_directory'):
            target_dir = Path(self.config.output_dir) / source_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / 'features.csv'
            
            # Write header only if file doesn't exist yet
            header = not target_file.exists()
            
            group_df.to_csv(target_file, mode='a', header=header, index=False)
            
        self.logger.info(f"Saved {len(batch)} records across {len(df['source_directory'].unique())} datasets")


def main():
    parser = argparse.ArgumentParser(description="Extract features from token transaction data")
    parser.add_argument("--input-dir", default="data/raw", help="Directory containing raw CSV files")
    parser.add_argument("--output-dir", default="data/dataset_with_features", help="Output directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=100, help="Records to accumulate before writing")
    parser.add_argument("--max-files", type=int, help="Limit number of files (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint, start fresh")
    
    args = parser.parse_args()
    
    config = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        max_files=args.max_files,
        resume=not args.no_resume
    )
    
    app = FeatureExtractorApplication(config)
    app.run()


if __name__ == "__main__":
    main()
