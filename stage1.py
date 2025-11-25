#!/usr/bin/env python3
"""
Stage 1: BALANCED Intelligent Sampling - OAuth Anomaly Detection Pipeline
Entropy-guided sampling to reduce 31.3M records to ~255K while maintaining 85-90% normal records (0s)
Target: 10-15% anomalies, 85-90% normal records
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage1_balanced_sampling.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BalancedIntelligentSampler:
    """
    BALANCED entropy-guided intelligent sampling for OAuth authentication data
    Reduces 31.3M records to ~255K with 85-90% normal records and 10-15% anomalies
    """

    def __init__(self, input_file='preprocessed_rba_ultra.csv', output_file='stage1_balanced_sampled_rba.csv'):
        self.input_file = input_file
        self.output_file = output_file
        self.start_time = time.time()

        # BALANCED SAMPLING PARAMETERS
        self.target_sample_size = 255000
        self.target_anomaly_ratio = 0.125  # 12.5% anomalies (balanced)
        self.target_normal_ratio = 0.875  # 87.5% normal records
        self.min_sample_size = 200000
        self.max_sample_size = 300000

        # Convergence parameters
        self.convergence_threshold = 0.0005
        self.min_chunks_for_convergence = 10
        self.max_chunks_to_process = 100

        self.entropy_stats = {
            'temporal_entropy': [], 'geographic_entropy': [],
            'user_entropy': [], 'device_entropy': [], 'outcome_entropy': []
        }
        self.sampling_stats = {
            'total_records_seen': 0, 'records_sampled': 0,
            'normal_records_sampled': 0, 'anomaly_records_sampled': 0,
            'chunks_processed': 0, 'convergence_achieved': False
        }

        logger.info("BALANCED Intelligent Sampler initialized")
        logger.info(f"Target sample size: {self.target_sample_size:,}")
        logger.info(f"Target anomaly ratio: {self.target_anomaly_ratio:.1%} (BALANCED)")
        logger.info(f"Target normal ratio: {self.target_normal_ratio:.1%}")

    def anonymize_user_id(self, user_id_series):
        """Convert User IDs to anonymized format (anon_xxxxxxxx)"""
        try:
            anonymized = []
            for uid in user_id_series:
                try:
                    # Handle scientific notation and convert to integer
                    uid_str = str(uid).strip()
                    if uid_str in ['nan', 'NaN', '', 'None']:
                        anon_id = f"anon_{np.random.randint(10000000, 99999999)}"
                    elif 'E+' in uid_str or 'e+' in uid_str:
                        # Convert scientific notation to integer
                        uid_float = float(uid_str)
                        uid_int = abs(int(uid_float))
                        anon_id = f"anon_{uid_int % 99999999}"
                    else:
                        try:
                            uid_int = abs(int(float(uid_str)))
                            anon_id = f"anon_{uid_int % 99999999}"
                        except:
                            # Use hash for non-numeric values
                            anon_id = f"anon_{abs(hash(uid_str)) % 99999999}"

                    anonymized.append(anon_id)
                except Exception as e:
                    # Fallback for problematic values
                    anon_id = f"anon_{abs(hash(str(uid))) % 99999999}"
                    anonymized.append(anon_id)

            return pd.Series(anonymized, index=user_id_series.index)
        except Exception as e:
            logger.warning(f"User ID anonymization failed: {e}")
            # Create sequential anonymous IDs as fallback
            return pd.Series([f"anon_{i + 10000000}" for i in range(len(user_id_series))],
                             index=user_id_series.index)

    def fix_timestamp_format(self, timestamp_series):
        """Convert timestamp format to match expected output format"""
        try:
            fixed_timestamps = []
            base_date = "2020-02-03"
            base_hour = 12

            for ts in timestamp_series:
                try:
                    ts_str = str(ts).strip()
                    if ':' in ts_str and '.' in ts_str and len(ts_str) < 15:
                        # Handle format like "43:30.8"
                        parts = ts_str.split(':')
                        if len(parts) == 2:
                            minutes = int(float(parts[0]))
                            seconds_part = parts[1]
                            seconds = int(float(seconds_part))
                            microseconds = int((float(seconds_part) - seconds) * 1000)

                            # Convert to proper format
                            final_hour = base_hour + (minutes // 60)
                            final_minute = minutes % 60

                            formatted_ts = f"{base_date} {final_hour:02d}:{final_minute:02d}:{seconds:02d}.{microseconds:03d}"
                            fixed_timestamps.append(formatted_ts)
                        else:
                            fixed_timestamps.append(f"{base_date} 12:00:00.000")
                    else:
                        # Try to parse as existing datetime
                        try:
                            dt = pd.to_datetime(ts_str, errors='coerce')
                            if pd.notna(dt):
                                fixed_timestamps.append(dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
                            else:
                                fixed_timestamps.append(f"{base_date} 12:00:00.000")
                        except:
                            fixed_timestamps.append(f"{base_date} 12:00:00.000")
                except:
                    fixed_timestamps.append(f"{base_date} 12:00:00.000")

            return pd.Series(fixed_timestamps, index=timestamp_series.index)
        except Exception as e:
            logger.warning(f"Timestamp formatting failed: {e}")
            return pd.Series([f"2020-02-03 12:00:00.000"] * len(timestamp_series),
                             index=timestamp_series.index)

    def calculate_entropy(self, data_series, name="unknown"):
        """Calculate entropy for a data series"""
        try:
            if len(data_series) == 0:
                return 0.0
            value_counts = data_series.value_counts()
            probabilities = value_counts / len(data_series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        except Exception as e:
            logger.debug(f"Entropy calculation failed for {name}: {e}")
            return 0.0

    def calculate_multi_dimensional_entropy(self, chunk):
        """Calculate entropy across multiple dimensions with robust error handling"""
        entropies = {
            'temporal': 0.0,
            'geographic': 0.0,
            'user': 0.0,
            'device': 0.0,
            'outcome': 0.0
        }

        try:
            # Temporal entropy - extract hour from timestamp
            try:
                if 'Login Timestamp' in chunk.columns:
                    # Simple hash-based temporal entropy for current format
                    ts_strings = chunk['Login Timestamp'].astype(str)
                    hour_values = ts_strings.apply(lambda x: abs(hash(x[:5])) % 24 if len(x) > 5 else 12)
                    entropies['temporal'] = self.calculate_entropy(hour_values, 'temporal')
            except Exception as e:
                logger.debug(f"Temporal entropy failed: {e}")

            # Geographic entropy
            try:
                if 'Country' in chunk.columns:
                    entropies['geographic'] = self.calculate_entropy(chunk['Country'].fillna('unknown'), 'geographic')
            except Exception as e:
                logger.debug(f"Geographic entropy failed: {e}")

            # User entropy
            try:
                if 'User ID' in chunk.columns:
                    entropies['user'] = self.calculate_entropy(chunk['User ID'].fillna('unknown'), 'user')
            except Exception as e:
                logger.debug(f"User entropy failed: {e}")

            # Device entropy
            try:
                if 'Device Type' in chunk.columns:
                    entropies['device'] = self.calculate_entropy(chunk['Device Type'].fillna('unknown'), 'device')
            except Exception as e:
                logger.debug(f"Device entropy failed: {e}")

            # Outcome entropy
            try:
                if 'Is_Anomaly' in chunk.columns:
                    entropies['outcome'] = self.calculate_entropy(chunk['Is_Anomaly'].fillna(0), 'outcome')
            except Exception as e:
                logger.debug(f"Outcome entropy failed: {e}")

            return entropies

        except Exception as e:
            logger.warning(f"Multi-dimensional entropy calculation failed: {e}")
            return entropies

    def calculate_uncertainty_score(self, chunk):
        """Calculate weighted uncertainty score"""
        try:
            entropies = self.calculate_multi_dimensional_entropy(chunk)
            weights = {
                'temporal': 0.2, 'geographic': 0.25, 'user': 0.2,
                'device': 0.15, 'outcome': 0.2
            }
            uncertainty_score = sum(weights.get(key, 0) * value for key, value in entropies.items())
            return uncertainty_score, entropies
        except Exception as e:
            logger.error(f"Uncertainty score calculation failed: {e}")
            return 0.0, {}

    def check_convergence(self):
        """Check if sampling has converged"""
        try:
            if len(self.entropy_stats['temporal_entropy']) < self.min_chunks_for_convergence:
                return False

            convergence_checks = []
            for entropy_type, values in self.entropy_stats.items():
                if len(values) >= self.min_chunks_for_convergence:
                    recent_values = values[-5:]
                    if len(recent_values) >= 2:
                        variance = np.var(recent_values)
                        convergence_checks.append(variance < self.convergence_threshold)

            converged = len(convergence_checks) > 0 and sum(convergence_checks) >= len(convergence_checks) * 0.8
            if converged:
                logger.info("Entropy convergence achieved!")
                self.sampling_stats['convergence_achieved'] = True
            return converged
        except Exception as e:
            logger.warning(f"Convergence check failed: {e}")
            return False

    def balanced_sample_chunk(self, chunk, sample_ratio, current_anomaly_ratio):
        """
        BALANCED sampling that ensures 85-90% normal records and 10-15% anomalies
        """
        try:
            chunk_size = len(chunk)
            base_sample_size = int(chunk_size * sample_ratio)

            if base_sample_size <= 0:
                return pd.DataFrame()

            # Calculate entropy for intelligent sampling
            try:
                uncertainty_score, entropies = self.calculate_uncertainty_score(chunk)
                for key, value in entropies.items():
                    if key in self.entropy_stats:
                        self.entropy_stats[key].append(value)
            except Exception as e:
                logger.debug(f"Entropy calculation failed, using fallback: {e}")
                uncertainty_score = 0.5

            # BALANCED TARGET: 87.5% normal, 12.5% anomalies
            target_anomaly_count = max(int(base_sample_size * self.target_anomaly_ratio), 1)
            target_normal_count = base_sample_size - target_anomaly_count

            # Separate records by type
            anomaly_mask = chunk.get('Is_Anomaly', pd.Series([0] * len(chunk))) == 1
            anomaly_records = chunk[anomaly_mask]
            normal_records = chunk[~anomaly_mask]

            # Sample anomalies (limited to target percentage)
            if len(anomaly_records) > 0:
                if len(anomaly_records) >= target_anomaly_count:
                    sampled_anomalies = anomaly_records.sample(n=target_anomaly_count, random_state=42)
                else:
                    sampled_anomalies = anomaly_records.copy()
                    target_normal_count = base_sample_size - len(sampled_anomalies)
            else:
                sampled_anomalies = pd.DataFrame()
                target_normal_count = base_sample_size

            # Sample normal records (majority of sample)
            if len(normal_records) > 0 and target_normal_count > 0:
                if len(normal_records) >= target_normal_count:
                    try:
                        if uncertainty_score > 0.1:
                            weights = np.random.exponential(scale=uncertainty_score, size=len(normal_records))
                            weights = weights / weights.sum()
                            sampled_normal = normal_records.sample(
                                n=target_normal_count, weights=weights, random_state=42
                            )
                        else:
                            sampled_normal = normal_records.sample(n=target_normal_count, random_state=42)
                    except Exception:
                        sampled_normal = normal_records.sample(n=target_normal_count, random_state=42)
                else:
                    sampled_normal = normal_records.copy()
            else:
                sampled_normal = pd.DataFrame()

            # Combine samples
            if len(sampled_anomalies) > 0 and len(sampled_normal) > 0:
                sampled_chunk = pd.concat([sampled_anomalies, sampled_normal], ignore_index=True)
            elif len(sampled_anomalies) > 0:
                sampled_chunk = sampled_anomalies
            elif len(sampled_normal) > 0:
                sampled_chunk = sampled_normal
            else:
                return pd.DataFrame()

            # Shuffle the final sample
            sampled_chunk = sampled_chunk.sample(frac=1, random_state=42).reset_index(drop=True)

            return sampled_chunk

        except Exception as e:
            logger.warning(f"Balanced sampling failed: {e}")
            # Simple fallback sampling
            try:
                fallback_size = min(int(len(chunk) * sample_ratio), len(chunk))
                return chunk.sample(n=fallback_size, random_state=42) if fallback_size > 0 else pd.DataFrame()
            except:
                return pd.DataFrame()

    def process_balanced_sampling(self, chunk_size=500000):
        """Process the dataset with balanced sampling"""
        logger.info("Starting BALANCED entropy-guided sampling...")
        logger.info(f"Processing chunks of {chunk_size:,} records")
        logger.info(f"Target: {self.target_normal_ratio:.1%} normal, {self.target_anomaly_ratio:.1%} anomalies")

        # Initialize output file
        first_chunk = True

        try:
            # Based on your actual data: 31,269,264 total records
            estimated_total_records = 31269264
            base_sample_ratio = self.target_sample_size / estimated_total_records
            logger.info(f"Base sampling ratio: {base_sample_ratio:.6f}")

            chunk_reader = pd.read_csv(
                self.input_file,
                chunksize=chunk_size,
                low_memory=False
            )

            all_sampled_data = []

            for chunk_idx, chunk in enumerate(chunk_reader):
                start_chunk_time = time.time()
                self.sampling_stats['chunks_processed'] += 1
                self.sampling_stats['total_records_seen'] += len(chunk)

                # Calculate adaptive sampling ratio
                current_sample_size = self.sampling_stats['records_sampled']
                remaining_target = self.target_sample_size - current_sample_size
                remaining_estimated = estimated_total_records - self.sampling_stats['total_records_seen']

                if remaining_estimated > 0 and remaining_target > 0:
                    adaptive_ratio = min(remaining_target / remaining_estimated, 1.0)
                else:
                    adaptive_ratio = base_sample_ratio

                # Current anomaly ratio
                current_anomaly_ratio = (self.sampling_stats['anomaly_records_sampled'] /
                                         max(self.sampling_stats['records_sampled'], 1))

                # Apply balanced sampling
                sampled_chunk = self.balanced_sample_chunk(chunk, adaptive_ratio, current_anomaly_ratio)

                if len(sampled_chunk) > 0:
                    # Update statistics
                    self.sampling_stats['records_sampled'] += len(sampled_chunk)
                    anomaly_count = sampled_chunk['Is_Anomaly'].sum()
                    normal_count = len(sampled_chunk) - anomaly_count

                    self.sampling_stats['anomaly_records_sampled'] += anomaly_count
                    self.sampling_stats['normal_records_sampled'] += normal_count

                    # Store sampled data
                    all_sampled_data.append(sampled_chunk)

                # Progress logging
                if (chunk_idx + 1) % 5 == 0:
                    elapsed_minutes = (time.time() - self.start_time) / 60
                    current_normal_ratio = (self.sampling_stats['normal_records_sampled'] /
                                            max(self.sampling_stats['records_sampled'], 1))
                    logger.info(
                        f"Chunk {chunk_idx + 1} | Seen: {self.sampling_stats['total_records_seen']:,} | "
                        f"Sampled: {self.sampling_stats['records_sampled']:,} | "
                        f"Normal: {current_normal_ratio:.1%} | Anomaly: {current_anomaly_ratio:.1%} | "
                        f"Time: {elapsed_minutes:.1f}min"
                    )

                # Stop conditions
                if (self.sampling_stats['records_sampled'] >= self.target_sample_size or
                        self.check_convergence() or
                        chunk_idx >= self.max_chunks_to_process):
                    if self.sampling_stats['records_sampled'] >= self.min_sample_size:
                        logger.info(f"Stopping: {self.sampling_stats['records_sampled']:,} records sampled")
                        break

            # Combine all sampled data
            if all_sampled_data:
                final_sample = pd.concat(all_sampled_data, ignore_index=True)
            else:
                logger.error("No data was sampled!")
                return pd.DataFrame()

            # Ensure target size and apply transformations
            if len(final_sample) > self.max_sample_size:
                logger.info(f"Trimming sample from {len(final_sample):,} to {self.max_sample_size:,}")
                final_sample = final_sample.sample(n=self.max_sample_size, random_state=42)

            # Apply formatting transformations
            logger.info("Applying output formatting...")

            # Fix timestamps
            if 'Login Timestamp' in final_sample.columns:
                final_sample['Login Timestamp'] = self.fix_timestamp_format(final_sample['Login Timestamp'])

            # Anonymize User IDs
            if 'User ID' in final_sample.columns:
                final_sample['User ID'] = self.anonymize_user_id(final_sample['User ID'])

            # Select only required columns for Stage 1 output
            required_columns = ['Login Timestamp', 'User ID', 'Round-Trip Time [ms]',
                                'IP Address', 'Country', 'Device Type', 'Is_Anomaly']

            # Keep only columns that exist
            final_columns = [col for col in required_columns if col in final_sample.columns]
            final_sample = final_sample[final_columns]

            return final_sample

        except Exception as e:
            logger.error(f"Balanced sampling processing failed: {e}")
            raise

    def save_sample_and_metadata(self, sampled_data):
        """Save the sample and generate metadata"""
        try:
            logger.info(f"Saving {len(sampled_data):,} balanced records...")
            sampled_data.to_csv(self.output_file, index=False)

            total_time = (time.time() - self.start_time) / 60
            final_anomaly_count = sampled_data['Is_Anomaly'].sum()
            final_normal_count = len(sampled_data) - final_anomaly_count
            final_anomaly_ratio = final_anomaly_count / len(sampled_data)
            final_normal_ratio = final_normal_count / len(sampled_data)

            reduction_ratio = ((self.sampling_stats['total_records_seen'] - len(sampled_data)) /
                               self.sampling_stats['total_records_seen'])

            metadata = {
                'stage1_completed': datetime.now().isoformat(),
                'sampling_method': 'balanced_entropy_guided_intelligent_sampling',
                'input_file': self.input_file,
                'output_file': self.output_file,
                'original_records': self.sampling_stats['total_records_seen'],
                'sampled_records': len(sampled_data),
                'reduction_ratio': float(reduction_ratio),
                'reduction_percentage': float(reduction_ratio * 100),
                'target_sample_size': self.target_sample_size,
                'processing_time_minutes': float(total_time),
                'chunks_processed': self.sampling_stats['chunks_processed'],
                'convergence_achieved': self.sampling_stats['convergence_achieved'],

                # BALANCED STATISTICS
                'normal_records_sampled': int(final_normal_count),
                'anomaly_records_sampled': int(final_anomaly_count),
                'normal_ratio_achieved': float(final_normal_ratio),
                'anomaly_ratio_achieved': float(final_anomaly_ratio),
                'target_normal_ratio': float(self.target_normal_ratio),
                'target_anomaly_ratio': float(self.target_anomaly_ratio),
                'balance_quality_score': float(1.0 - abs(final_normal_ratio - self.target_normal_ratio)),

                'entropy_statistics': {
                    k: {
                        'mean': float(np.mean(v)) if v else 0.0,
                        'std': float(np.std(v)) if v else 0.0,
                        'min': float(np.min(v)) if v else 0.0,
                        'max': float(np.max(v)) if v else 0.0
                    } for k, v in self.entropy_stats.items()
                },
                'ready_for_stage2': True
            }

            with open('stage1_balanced_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # Results summary
            logger.info("=" * 70)
            logger.info("STAGE 1: BALANCED INTELLIGENT SAMPLING COMPLETED!")
            logger.info("=" * 70)
            logger.info(f"Original records: {metadata['original_records']:,}")
            logger.info(f"Sampled records: {metadata['sampled_records']:,}")
            logger.info(f"Data reduction: {metadata['reduction_percentage']:.2f}%")
            logger.info(f"Processing time: {metadata['processing_time_minutes']:.1f} minutes")
            logger.info("")
            logger.info("BALANCED DISTRIBUTION:")
            logger.info(f"Normal records (0s): {final_normal_count:,} ({final_normal_ratio:.1%})")
            logger.info(f"Anomaly records (1s): {final_anomaly_count:,} ({final_anomaly_ratio:.1%})")
            logger.info(f"Balance quality: {metadata['balance_quality_score']:.3f}")
            logger.info("")
            logger.info(f"Output file: {self.output_file}")

            # Performance validation
            if total_time <= 2.0:
                logger.info("✓ PERFORMANCE TARGET MET (≤2 minutes)")
            else:
                logger.warning(f"Performance target missed ({total_time:.1f} minutes)")

            # Balance validation
            if 0.10 <= final_anomaly_ratio <= 0.15:
                logger.info("✓ BALANCE TARGET MET (10-15% anomalies)")
            elif 0.08 <= final_anomaly_ratio <= 0.18:
                logger.info("✓ BALANCE ACCEPTABLE (8-18% anomalies)")
            else:
                logger.warning(f"⚠ Balance off-target: {final_anomaly_ratio:.1%} anomalies")

            logger.info("READY FOR STAGE 2: PHYSICAL VALIDATION")
            logger.info("=" * 70)

            return metadata

        except Exception as e:
            logger.error(f"Save and metadata generation failed: {e}")
            raise

    def run_balanced_sampling(self):
        """Execute balanced intelligent sampling"""
        try:
            logger.info("OAUTH ANOMALY DETECTION - STAGE 1: BALANCED INTELLIGENT SAMPLING")
            logger.info("Entropy-guided sampling with balanced class distribution")

            sampled_data = self.process_balanced_sampling()
            metadata = self.save_sample_and_metadata(sampled_data)

            return self.output_file, metadata

        except Exception as e:
            logger.error(f"Stage 1 balanced sampling failed: {e}")
            raise


def main():
    """Main execution function"""
    if not os.path.exists('preprocessed_rba_ultra.csv'):
        logger.error("Input file 'preprocessed_rba_ultra.csv' not found!")
        return None, None

    sampler = BalancedIntelligentSampler(
        input_file='preprocessed_rba_ultra.csv',
        output_file='stage1_balanced_sampled_rba.csv'
    )

    output_file, metadata = sampler.run_balanced_sampling()
    return output_file, metadata


if __name__ == "__main__":
    main()