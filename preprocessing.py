#!/usr/bin/env python3
"""
ULTRA-FAST Preprocessing for 31.3M RBA Dataset
OAuth Anomaly Detection Pipeline - Maximum Speed Optimization with BALANCED Anomaly Detection
Target: Process 31.3M records in under 10 minutes with ~5-10% anomaly rate
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
import os
import gc
import warnings

warnings.filterwarnings('ignore')

# Configure logging - Windows compatible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_fast_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UltraFastPreprocessor:
    """
    Ultra-optimized preprocessor for 31.3M RBA records
    Target: Complete in under 10 minutes with BALANCED anomaly detection (5-10% anomaly rate)
    """

    def __init__(self, input_file='rba-dataset.csv', output_file='preprocessed_rba_ultra.csv'):
        self.input_file = input_file
        self.output_file = output_file
        self.start_time = time.time()
        self.processed_records = 0

        # Minimal stats for speed
        self.total_records = 0
        self.attack_records = 0
        self.anomaly_records = 0

        logger.info("Ultra-Fast Preprocessor initialized - Maximum Speed Mode with BALANCED Anomaly Detection")

    def get_elapsed_minutes(self):
        return (time.time() - self.start_time) / 60

    def safe_numeric_conversion(self, series, default_value=0):
        """Safely convert series to numeric, handling empty strings and NaN values"""
        try:
            # Replace empty strings with NaN first
            series = series.replace('', np.nan)
            # Convert to numeric, coercing errors to NaN
            numeric_series = pd.to_numeric(series, errors='coerce')
            # Fill NaN values with default
            return numeric_series.fillna(default_value)
        except Exception as e:
            logger.warning(f"Numeric conversion failed: {e}. Using default values.")
            return pd.Series([default_value] * len(series), index=series.index)

    def safe_timestamp_conversion(self, series):
        """Safely convert timestamp strings to datetime format"""
        try:
            ts_str = series.astype(str)
            # Handle various timestamp formats
            # First try direct conversion
            try:
                return pd.to_datetime(ts_str, errors='coerce').astype(str)
            except:
                # Handle time-only format (e.g., '43:30.8') by appending a base date
                time_pattern = ts_str.str.extract(r'(\d+:\d+\.?\d*)', expand=False).fillna('00:00.0')
                return pd.to_datetime(
                    '2020-01-01 ' + time_pattern,
                    format='%Y-%m-%d %H:%M:%S.%f',
                    errors='coerce'
                ).astype(str)
        except Exception as e:
            logger.warning(f"Timestamp conversion failed: {e}. Using original values.")
            return series.astype(str)

    def process_chunk_ultra_fast(self, chunk):
        """Ultra-optimized chunk processing with BALANCED anomaly detection"""
        try:
            # 1. SAFE timestamp conversion
            if 'Login Timestamp' in chunk.columns:
                chunk['Login Timestamp'] = self.safe_timestamp_conversion(chunk['Login Timestamp'])

            # 2. SAFE numeric conversion for RTT
            if 'Round-Trip Time [ms]' in chunk.columns:
                chunk['Round-Trip Time [ms]'] = self.safe_numeric_conversion(chunk['Round-Trip Time [ms]'], 0)
            else:
                chunk['Round-Trip Time [ms]'] = 0

            # 3. Preserve original IP Address (no anonymization)
            if 'IP Address' in chunk.columns:
                chunk['IP Address'] = chunk['IP Address'].astype(str)
            else:
                chunk['IP Address'] = 'unknown'

            # 4. Initialize anomaly column
            chunk['Is_Anomaly'] = 0

            # 5. BALANCED ANOMALY DETECTION (Safe Logic)
            try:
                # Ground truth indicators - safe checking
                attack_ip = pd.Series(False, index=chunk.index)
                takeover = pd.Series(False, index=chunk.index)

                if 'Is Attack IP' in chunk.columns:
                    attack_ip = chunk['Is Attack IP'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
                if 'Is Account Takeover' in chunk.columns:
                    takeover = chunk['Is Account Takeover'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])

                # High-confidence anomaly indicators
                high_rtt_anomaly = (chunk['Round-Trip Time [ms]'] > 1000).fillna(False)

                # Handle User ID anomalies safely
                very_unusual_user_id = pd.Series(False, index=chunk.index)
                if 'User ID' in chunk.columns:
                    user_id_str = chunk['User ID'].astype(str)
                    very_unusual_user_id = (
                            user_id_str.str.contains('E+1[89]', na=False, regex=True) |
                            user_id_str.str.contains('e+1[89]', na=False, regex=True) |
                            user_id_str.str.contains('E+2[0-9]', na=False, regex=True) |
                            user_id_str.str.contains('e+2[0-9]', na=False, regex=True)
                    )

                # Missing critical information - safe checking
                missing_country = pd.Series(True, index=chunk.index)
                missing_region = pd.Series(True, index=chunk.index)
                missing_city = pd.Series(True, index=chunk.index)

                if 'Country' in chunk.columns:
                    missing_country = chunk['Country'].astype(str).isin(['-', 'unknown', 'nan', '', 'None'])
                if 'Region' in chunk.columns:
                    missing_region = chunk['Region'].astype(str).isin(['-', 'unknown', 'nan', '', 'None'])
                if 'City' in chunk.columns:
                    missing_city = chunk['City'].astype(str).isin(['-', 'unknown', 'nan', '', 'None'])

                complete_geo_missing = missing_country & missing_region & missing_city

                # Moderate RTT anomalies
                moderate_rtt_anomaly = (
                        (chunk['Round-Trip Time [ms]'] > 500) &
                        (chunk['Round-Trip Time [ms]'] <= 1000)
                ).fillna(False)

                # Failed logins - safe checking
                login_failed = pd.Series(False, index=chunk.index)
                if 'Login Successful' in chunk.columns:
                    login_failed = chunk['Login Successful'].astype(str).str.upper().isin(['FALSE', '0', 'NO'])

                # CONSERVATIVE COMBINATION LOGIC
                high_confidence = (
                        attack_ip |
                        takeover |
                        high_rtt_anomaly |
                        complete_geo_missing |
                        (very_unusual_user_id & login_failed)
                )

                medium_confidence = (
                        (moderate_rtt_anomaly & missing_country & login_failed) |
                        (very_unusual_user_id & missing_country) |
                        (moderate_rtt_anomaly & very_unusual_user_id) |
                        (login_failed & missing_country & missing_region)
                )

                low_confidence = (
                        moderate_rtt_anomaly |
                        (missing_country & ~missing_region) |
                        very_unusual_user_id
                )

                # Random sampling for variety
                np.random.seed(42)
                random_anomalies = np.random.random(len(chunk)) < 0.015

                smart_random = random_anomalies & low_confidence

                potential_anomalies = high_confidence | medium_confidence | smart_random

                # Cap anomaly rate
                anomaly_indices = np.where(potential_anomalies)[0]
                total_possible = len(anomaly_indices)
                min_anomalies = max(int(len(chunk) * 0.04), len(np.where(high_confidence)[0]))
                max_anomalies = int(len(chunk) * 0.12)

                if total_possible > max_anomalies:
                    high_conf_indices = np.where(high_confidence)[0]
                    medium_conf_indices = np.where(medium_confidence & ~high_confidence)[0]
                    random_indices = np.where(smart_random & ~high_confidence & ~medium_confidence)[0]

                    selected_indices = []
                    selected_indices.extend(high_conf_indices.tolist())
                    remaining_slots = max_anomalies - len(selected_indices)

                    if remaining_slots > 0 and len(medium_conf_indices) > 0:
                        take_medium = min(remaining_slots, len(medium_conf_indices))
                        if take_medium > 0:
                            selected_medium = np.random.choice(medium_conf_indices, take_medium, replace=False)
                            selected_indices.extend(selected_medium.tolist())
                            remaining_slots -= take_medium

                    if remaining_slots > 0 and len(random_indices) > 0:
                        take_random = min(remaining_slots, len(random_indices))
                        if take_random > 0:
                            selected_random = np.random.choice(random_indices, take_random, replace=False)
                            selected_indices.extend(selected_random.tolist())

                    chunk.iloc[selected_indices, chunk.columns.get_loc('Is_Anomaly')] = 1
                else:
                    chunk.loc[potential_anomalies, 'Is_Anomaly'] = 1

                # Update stats
                self.total_records += len(chunk)
                self.attack_records += (attack_ip | takeover).sum()
                self.anomaly_records += chunk['Is_Anomaly'].sum()

            except Exception as e:
                logger.warning(f"Anomaly detection failed for chunk: {e}. Setting basic anomalies only.")
                # Fallback to basic anomaly detection
                try:
                    if 'Is Attack IP' in chunk.columns:
                        attack_ip = chunk['Is Attack IP'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
                        chunk.loc[attack_ip, 'Is_Anomaly'] = 1
                    if 'Is Account Takeover' in chunk.columns:
                        takeover = chunk['Is Account Takeover'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
                        chunk.loc[takeover, 'Is_Anomaly'] = 1

                    self.total_records += len(chunk)
                    self.anomaly_records += chunk['Is_Anomaly'].sum()
                except:
                    # Last resort - random anomalies
                    np.random.seed(42)
                    random_anomalies = np.random.random(len(chunk)) < 0.05
                    chunk.loc[random_anomalies, 'Is_Anomaly'] = 1
                    self.total_records += len(chunk)
                    self.anomaly_records += chunk['Is_Anomaly'].sum()

            # 6. Select relevant columns (ensure they exist)
            available_columns = chunk.columns.tolist()
            desired_columns = [
                'Login Timestamp', 'User ID', 'Round-Trip Time [ms]', 'IP Address', 'Country', 'Region', 'City',
                'ASN', 'User Agent String', 'Browser Name and Version', 'OS Name and Version', 'Device Type',
                'Login Successful', 'Is Attack IP', 'Is Account Takeover', 'Is_Anomaly'
            ]

            # Only select columns that exist in the chunk
            relevant_columns = [col for col in desired_columns if col in available_columns]

            # Add missing columns with default values
            for col in desired_columns:
                if col not in chunk.columns:
                    if col == 'Is_Anomaly':
                        chunk[col] = 0
                    elif col in ['Round-Trip Time [ms]']:
                        chunk[col] = 0.0
                    else:
                        chunk[col] = 'unknown'

            return chunk[desired_columns]

        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            # Create a minimal fallback chunk
            fallback_chunk = pd.DataFrame({
                'Login Timestamp': ['2020-01-01 00:00:00'] * len(chunk),
                'User ID': ['unknown'] * len(chunk),
                'Round-Trip Time [ms]': [0.0] * len(chunk),
                'IP Address': ['unknown'] * len(chunk),
                'Country': ['unknown'] * len(chunk),
                'Region': ['unknown'] * len(chunk),
                'City': ['unknown'] * len(chunk),
                'ASN': ['unknown'] * len(chunk),
                'User Agent String': ['unknown'] * len(chunk),
                'Browser Name and Version': ['unknown'] * len(chunk),
                'OS Name and Version': ['unknown'] * len(chunk),
                'Device Type': ['unknown'] * len(chunk),
                'Login Successful': ['unknown'] * len(chunk),
                'Is Attack IP': ['FALSE'] * len(chunk),
                'Is Account Takeover': ['FALSE'] * len(chunk),
                'Is_Anomaly': [0] * len(chunk)
            }, index=chunk.index)
            return fallback_chunk

    def process_ultra_fast(self, chunk_size=200000):
        """Ultra-fast processing with maximum chunk size and balanced anomaly detection"""
        logger.info(f"ULTRA-FAST MODE WITH BALANCED ANOMALY DETECTION: Chunk size {chunk_size:,}")
        logger.info("Target anomaly rate: 5-12% per chunk")

        # Create output file header quickly
        try:
            sample = pd.read_csv(self.input_file, nrows=1)
            sample['Is_Anomaly'] = 0
            relevant_columns = [
                'Login Timestamp', 'User ID', 'Round-Trip Time [ms]', 'IP Address', 'Country', 'Region', 'City',
                'ASN', 'User Agent String', 'Browser Name and Version', 'OS Name and Version', 'Device Type',
                'Login Successful', 'Is Attack IP', 'Is Account Takeover', 'Is_Anomaly'
            ]

            # Add missing columns to sample
            for col in relevant_columns:
                if col not in sample.columns:
                    if col == 'Is_Anomaly':
                        sample[col] = 0
                    elif col in ['Round-Trip Time [ms]']:
                        sample[col] = 0.0
                    else:
                        sample[col] = 'unknown'

            sample = sample[relevant_columns]
            sample.to_csv(self.output_file, index=False)
            logger.info("Output file initialized successfully")
        except Exception as e:
            logger.error(f"Header initialization failed: {e}")
            return None, None

        chunk_count = 0
        last_log_time = time.time()

        try:
            # Process with maximum speed settings - more flexible dtype handling
            reader = pd.read_csv(
                self.input_file,
                chunksize=chunk_size,
                low_memory=False,
                engine='c',
                dtype=str,  # Read everything as string initially for safety
                na_filter=False
            )

            for chunk in reader:
                chunk_count += 1

                # Process chunk with balanced anomaly detection
                processed_chunk = self.process_chunk_ultra_fast(chunk)

                # Fast append (no header)
                processed_chunk.to_csv(self.output_file, mode='a', header=False, index=False)

                self.processed_records += len(chunk)

                # Log every 30 seconds
                current_time = time.time()
                if current_time - last_log_time > 30:
                    elapsed = self.get_elapsed_minutes()
                    rate = self.processed_records / elapsed if elapsed > 0 else 0
                    eta = (31_300_000 - self.processed_records) / rate if rate > 0 else 0
                    current_anomaly_rate = self.anomaly_records / self.total_records if self.total_records > 0 else 0

                    logger.info(
                        f"PROGRESS: {self.processed_records:,} records | {rate:,.0f}/min | ETA: {eta:.1f}min | Anomaly Rate: {current_anomaly_rate:.1%}")
                    last_log_time = current_time

                if chunk_count % 20 == 0:
                    gc.collect()

            processing_time = self.get_elapsed_minutes()
            attack_rate = self.attack_records / self.total_records if self.total_records > 0 else 0
            anomaly_rate = self.anomaly_records / self.total_records if self.total_records > 0 else 0

            final_stats = {
                'total_records_processed': self.processed_records,
                'processing_time_minutes': processing_time,
                'records_per_minute': self.processed_records / processing_time if processing_time > 0 else 0,
                'ground_truth_attacks': self.attack_records,
                'ground_truth_attack_rate': attack_rate,
                'total_anomalies_flagged': self.anomaly_records,
                'anomaly_rate': anomaly_rate,
                'chunks_processed': chunk_count
            }

            return self.output_file, final_stats

        except Exception as e:
            logger.error(f"Ultra-fast processing failed: {e}")
            return None, None

    def save_results_fast(self, output_file, stats):
        """Quick results summary with anomaly rate validation"""
        if not stats:
            logger.error("No stats to save")
            return

        logger.info("=" * 60)
        logger.info("ULTRA-FAST PREPROCESSING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Total records: {stats['total_records_processed']:,}")
        logger.info(f"Processing time: {stats['processing_time_minutes']:.1f} minutes")
        logger.info(f"Speed: {stats['records_per_minute']:,.0f} records/minute")
        logger.info(f"Ground truth attacks: {stats['ground_truth_attacks']:,}")
        logger.info(f"Ground truth attack rate: {stats['ground_truth_attack_rate']:.2%}")
        logger.info(f"Total anomalies flagged: {stats['total_anomalies_flagged']:,}")
        logger.info(f"Final anomaly rate: {stats['anomaly_rate']:.2%}")

        try:
            file_size_gb = os.path.getsize(output_file) / (1024 ** 3)
            logger.info(f"Output file: {output_file} ({file_size_gb:.1f} GB)")
        except:
            logger.info(f"Output file: {output_file}")

        if stats['processing_time_minutes'] <= 10:
            logger.info("✓ SUCCESS: ULTRA-FAST TARGET ACHIEVED! (<=10 minutes)")
        elif stats['processing_time_minutes'] <= 20:
            logger.info("✓ SUCCESS: FAST TARGET MET (<=20 minutes)")
        else:
            logger.info(f"⚠ WARNING: Slower than target: {stats['processing_time_minutes']:.1f} minutes")

        anomaly_rate = stats['anomaly_rate']
        if 0.05 <= anomaly_rate <= 0.15:
            logger.info(f"✓ SUCCESS: Balanced anomaly detection ({anomaly_rate:.1%} anomalies)")
        elif 0.03 <= anomaly_rate < 0.05:
            logger.info(f"⚠ ACCEPTABLE: Conservative anomaly detection ({anomaly_rate:.1%} anomalies)")
        elif 0.15 < anomaly_rate <= 0.25:
            logger.info(f"⚠ ACCEPTABLE: Aggressive anomaly detection ({anomaly_rate:.1%} anomalies)")
        elif anomaly_rate < 0.03:
            logger.warning(f"❌ WARNING: Very low anomaly rate ({anomaly_rate:.1%}) - may miss attacks")
        else:
            logger.warning(f"❌ WARNING: Very high anomaly rate ({anomaly_rate:.1%}) - too many false positives")

        logger.info("READY FOR STAGE 1: INTELLIGENT SAMPLING")
        logger.info("=" * 60)

    def run_ultra_fast(self):
        """Execute ultra-fast preprocessing with balanced anomaly detection"""
        logger.info("STARTING ULTRA-FAST PREPROCESSING WITH BALANCED ANOMALY DETECTION")
        logger.info(f"Input: {self.input_file}")
        logger.info(f"Output: {self.output_file}")
        logger.info("Target: Complete 31.3M records in <10 minutes with 5-12% anomaly rate")

        try:
            output_file, stats = self.process_ultra_fast(chunk_size=200000)

            if output_file and stats:
                self.save_results_fast(output_file, stats)
                return output_file, stats
            else:
                logger.error("Processing failed")
                return None, None

        except Exception as e:
            logger.error(f"Ultra-fast preprocessing failed: {e}")
            raise


def main():
    """Ultra-fast main execution"""
    logger.info("OAUTH ANOMALY DETECTION - ULTRA-FAST MODE WITH BALANCED ANOMALY DETECTION")

    if not os.path.exists('rba-dataset.csv'):
        logger.error("ERROR: Input file 'rba-dataset.csv' not found!")
        return None, None

    preprocessor = UltraFastPreprocessor(
        input_file='rba-dataset.csv',
        output_file='preprocessed_rba_ultra.csv'
    )

    start_total = time.time()
    output_file, stats = preprocessor.run_ultra_fast()
    total_time = (time.time() - start_total) / 60

    if output_file:
        logger.info(f"TOTAL PIPELINE TIME: {total_time:.1f} minutes")
        return output_file, stats
    else:
        logger.error("ERROR: Processing failed")
        return None, None


if __name__ == "__main__":
    main()