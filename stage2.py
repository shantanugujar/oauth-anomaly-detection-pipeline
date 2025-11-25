#!/usr/bin/env python3
"""
Stage 2: Physical Validation - OAuth Anomaly Detection Pipeline
Remove physically impossible authentication sequences while preserving legitimate mobility and anomalies
Target: Complete in 3-5 minutes, ~50% validation rate
IP addresses are preserved for downstream analysis
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import json
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage2_physical_validation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PhysicalValidator:
    """
    Physical constraint validation for OAuth authentication data
    Removes impossible travel patterns while preserving attack indicators and legitimate mobility
    IP addresses are preserved for downstream analysis
    """

    def __init__(self, input_file='stage1_balanced_sampled_rba.csv', output_file='stage2_validated_rba.csv'):
        self.input_file = input_file
        self.output_file = output_file
        self.start_time = time.time()

        # Physical validation parameters
        self.max_travel_speed_kmh = 1200  # Commercial flight + buffer
        self.min_time_threshold_minutes = 3  # Minimum time between locations
        self.max_time_threshold_hours = 24  # Maximum time window

        # RTT thresholds by device type
        self.rtt_thresholds = {
            'desktop': 300,
            'mobile': 500,
            'tablet': 400,
            'unknown': 400
        }

        # Country coordinates (comprehensive list)
        self.country_coords = {
            'US': (39.8283, -98.5795), 'CA': (56.1304, -106.3468), 'GB': (55.3781, -3.4360),
            'DE': (51.1657, 10.4515), 'FR': (46.2276, 2.2137), 'IT': (41.8719, 12.5674),
            'ES': (40.4637, -3.7492), 'NL': (52.1326, 5.2913), 'BE': (50.5039, 4.4699),
            'AU': (-25.2744, 133.7751), 'JP': (36.2048, 138.2529), 'CN': (35.8617, 104.1954),
            'IN': (20.5937, 78.9629), 'BR': (-14.2350, -51.9253), 'RU': (61.5240, 105.3188),
            'ZA': (-30.5595, 22.9375), 'MX': (23.6345, -102.5528), 'AR': (-38.4161, -63.6167),
            'KR': (35.9078, 127.7669), 'TH': (15.8700, 100.9925), 'SG': (1.3521, 103.8198),
            'MY': (4.2105, 101.9758), 'ID': (-0.7893, 113.9213), 'PH': (12.8797, 121.7740),
            'VN': (14.0583, 108.2772), 'EG': (26.0975, 30.0444), 'SA': (23.8859, 45.0792),
            'AE': (23.4241, 53.8478), 'IL': (31.0461, 34.8516), 'TR': (38.9637, 35.2433),
            'PL': (51.9194, 19.1451), 'CZ': (49.8175, 15.4730), 'AT': (47.5162, 14.5501),
            'CH': (46.8182, 8.2275), 'SE': (60.1282, 18.6435), 'NO': (60.4720, 8.4689),
            'DK': (56.2639, 9.5018), 'FI': (61.9241, 25.7482), 'IE': (53.4129, -8.2439),
            'PT': (39.3999, -8.2245), 'GR': (39.0742, 21.8243), 'RO': (45.9432, 24.9668),
            'BG': (42.7339, 25.4858), 'HR': (45.1000, 15.2000), 'HU': (47.1625, 19.5033),
            'SK': (48.6690, 19.6990), 'SI': (46.1512, 14.9955), 'LT': (55.1694, 23.8813),
            'LV': (56.8796, 24.6032), 'EE': (58.5953, 25.0136), 'Unknown': (0.0, 0.0)
        }

        # Statistics tracking
        self.validation_stats = {
            'total_records_processed': 0,
            'records_validated': 0,
            'records_rejected': 0,
            'impossible_travel_rejected': 0,
            'rtt_anomaly_rejected': 0,
            'device_inconsistency_rejected': 0,
            'attack_records_preserved': 0,
            'user_profiles_created': 0,
            'validation_rate': 0.0
        }

        # User profile cache
        self.user_profiles = defaultdict(lambda: {
            'locations': [],
            'travel_speeds': [],
            'rtt_history': [],
            'device_history': [],
            'ip_history': [],
            'max_observed_speed': 0,
            'avg_rtt': 0,
            'is_frequent_traveler': False,
            'last_valid_location': None,
            'last_valid_time': None,
            'last_valid_device': None,
            'last_valid_ip': None
        })

        logger.info("Physical Validator initialized - IP addresses will be preserved")
        logger.info(f"Max travel speed: {self.max_travel_speed_kmh} km/h")
        logger.info(f"Min time threshold: {self.min_time_threshold_minutes} minutes")

    def parse_timestamp(self, timestamp_str):
        """Parse the custom timestamp format (e.g., '20:43.2')"""
        try:
            if pd.isna(timestamp_str) or timestamp_str == '':
                return datetime(2020, 2, 3, 12, 0, 0)

            timestamp_str = str(timestamp_str).strip()

            # Handle the MM:SS.s format
            if ':' in timestamp_str and '.' in timestamp_str:
                parts = timestamp_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds_part = float(parts[1])

                    # Convert to proper datetime (using base date 2020-02-03)
                    base_time = datetime(2020, 2, 3, 12, 0, 0)  # Start at noon
                    total_seconds = minutes * 60 + seconds_part
                    return base_time + timedelta(seconds=total_seconds)

            # Fallback to parsing as regular datetime
            return pd.to_datetime(timestamp_str, errors='coerce')

        except Exception as e:
            logger.debug(f"Timestamp parsing failed for '{timestamp_str}': {e}")
            return datetime(2020, 2, 3, 12, 0, 0)

    def calculate_distance_km(self, coord1, coord2):
        """Calculate distance between two coordinates using Haversine formula"""
        try:
            lat1, lon1 = coord1
            lat2, lon2 = coord2

            # Convert to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            R = 6371  # Earth's radius in km

            return R * c
        except Exception as e:
            logger.debug(f"Distance calculation failed: {e}")
            return 0.0

    def get_country_coordinates(self, country):
        """Get coordinates for a country with fallback"""
        if pd.isna(country) or country == '' or country == 'Unknown':
            return self.country_coords['Unknown']
        return self.country_coords.get(str(country).strip().upper(), self.country_coords['Unknown'])

    def calculate_travel_speed(self, loc1, time1, loc2, time2):
        """Calculate travel speed between two locations and times"""
        try:
            coord1 = self.get_country_coordinates(loc1)
            coord2 = self.get_country_coordinates(loc2)
            distance_km = self.calculate_distance_km(coord1, coord2)

            # Same location or very close
            if distance_km < 1.0:
                return 0.0

            # Parse timestamps
            time1_dt = self.parse_timestamp(time1)
            time2_dt = self.parse_timestamp(time2)

            time_diff_seconds = abs((time2_dt - time1_dt).total_seconds())

            # Too close in time
            if time_diff_seconds < self.min_time_threshold_minutes * 60:
                return float('inf')

            time_diff_hours = time_diff_seconds / 3600
            speed_kmh = distance_km / time_diff_hours
            return speed_kmh

        except Exception as e:
            logger.debug(f"Travel speed calculation failed: {e}")
            return 0.0

    def update_user_profile(self, user_id, location, timestamp, ip_address, device_type, speed=None, rtt=None):
        """Update user profile with new authentication data"""
        try:
            profile = self.user_profiles[user_id]

            # Add location history
            profile['locations'].append({
                'country': location,
                'timestamp': timestamp,
                'coordinates': self.get_country_coordinates(location)
            })

            # Add speed data
            if speed is not None and speed > 0:
                profile['travel_speeds'].append(speed)
                profile['max_observed_speed'] = max(profile['max_observed_speed'], speed)

            # Add RTT history
            if rtt is not None and not pd.isna(rtt):
                profile['rtt_history'].append(float(rtt))
                if profile['rtt_history']:
                    profile['avg_rtt'] = np.mean(profile['rtt_history'])

            # Add device history
            if device_type and not pd.isna(device_type):
                profile['device_history'].append(str(device_type))

            # Add IP history (preserved, not anonymized)
            if ip_address and not pd.isna(ip_address):
                profile['ip_history'].append(str(ip_address))

            # Determine if frequent traveler
            profile['is_frequent_traveler'] = sum(1 for s in profile['travel_speeds'] if s > 500) >= 3

            # Update last valid state
            profile['last_valid_location'] = location
            profile['last_valid_time'] = timestamp
            profile['last_valid_device'] = device_type
            profile['last_valid_ip'] = ip_address

            return profile

        except Exception as e:
            logger.debug(f"User profile update failed for {user_id}: {e}")
            return self.user_profiles[user_id]

    def get_adaptive_speed_threshold(self, user_id):
        """Get adaptive speed threshold based on user travel history"""
        try:
            profile = self.user_profiles[user_id]
            base_threshold = self.max_travel_speed_kmh

            # Higher threshold for frequent travelers
            if profile['is_frequent_traveler']:
                return base_threshold * 1.5

            # Adaptive threshold based on history
            if len(profile['travel_speeds']) >= 3:
                avg_speed = np.mean(profile['travel_speeds'])
                max_speed = max(profile['travel_speeds'])

                if avg_speed > 300 and max_speed < base_threshold:
                    return base_threshold * 1.2

            return base_threshold

        except Exception as e:
            logger.debug(f"Adaptive threshold calculation failed: {e}")
            return self.max_travel_speed_kmh

    def validate_rtt_consistency(self, current_rtt, device_type, user_id):
        """Validate RTT consistency for device type and user history"""
        try:
            current_rtt = float(current_rtt) if not pd.isna(current_rtt) else 0
            device_threshold = self.rtt_thresholds.get(str(device_type).lower(), 400)

            # Check against device type threshold
            if current_rtt > device_threshold * 2:
                return False, f"RTT {current_rtt} ms exceeds device threshold {device_threshold * 2} ms"

            # Check against user history
            profile = self.user_profiles[user_id]
            if profile['avg_rtt'] > 0:
                # Allow some variation but flag extreme outliers
                if current_rtt > profile['avg_rtt'] * 3 and current_rtt > 1000:
                    return False, f"RTT {current_rtt} ms far exceeds user average {profile['avg_rtt']:.1f} ms"

            return True, "RTT within acceptable range"

        except Exception as e:
            logger.debug(f"RTT validation failed: {e}")
            return True, "RTT validation error - allowing"

    def validate_device_consistency(self, current_device, user_id):
        """Validate device consistency (allow reasonable device changes)"""
        try:
            profile = self.user_profiles[user_id]

            # No history - allow any device
            if not profile['device_history']:
                return True, "First device record"

            # Allow device changes but flag rapid switching
            recent_devices = profile['device_history'][-5:]  # Last 5 devices
            if len(set(recent_devices)) > 3:  # More than 3 different devices recently
                return False, f"Too many recent device changes: {set(recent_devices)}"

            return True, "Device change acceptable"

        except Exception as e:
            logger.debug(f"Device validation failed: {e}")
            return True, "Device validation error - allowing"

    def validate_record_sequence(self, current_record):
        """Validate if current record is physically possible"""
        try:
            # Extract record data
            current_country = current_record.get('Country', 'Unknown')
            current_time = current_record.get('Login Timestamp', '')
            current_user = current_record.get('User ID', 'unknown')
            current_rtt = current_record.get('Round-Trip Time [ms]', 0)
            current_device = current_record.get('Device Type', 'unknown')
            current_ip = current_record.get('IP Address', 'unknown')
            is_attack = current_record.get('Is_Anomaly', 0) == 1

            profile = self.user_profiles[current_user]

            # Always preserve attack records
            if is_attack:
                self.validation_stats['attack_records_preserved'] += 1
                self.update_user_profile(current_user, current_country, current_time,
                                         current_ip, current_device, 0, current_rtt)
                return True, "Attack record preserved"

            # First record for user
            if profile['last_valid_location'] is None:
                self.update_user_profile(current_user, current_country, current_time,
                                         current_ip, current_device, 0, current_rtt)
                return True, "First location for user"

            # Same location - always valid
            if current_country == profile['last_valid_location']:
                self.update_user_profile(current_user, current_country, current_time,
                                         current_ip, current_device, 0, current_rtt)
                return True, "Same location"

            # Calculate travel speed
            travel_speed = self.calculate_travel_speed(
                profile['last_valid_location'], profile['last_valid_time'],
                current_country, current_time
            )

            speed_threshold = self.get_adaptive_speed_threshold(current_user)

            # Check impossible travel
            if travel_speed > speed_threshold:
                return False, f"Impossible travel: {travel_speed:.1f} km/h > {speed_threshold:.1f} km/h"

            # Validate RTT consistency
            rtt_valid, rtt_reason = self.validate_rtt_consistency(current_rtt, current_device, current_user)
            if not rtt_valid:
                return False, rtt_reason

            # Validate device consistency
            device_valid, device_reason = self.validate_device_consistency(current_device, current_user)
            if not device_valid:
                return False, device_reason

            # Record is valid - update profile
            self.update_user_profile(current_user, current_country, current_time,
                                     current_ip, current_device, travel_speed, current_rtt)

            return True, f"Valid: {travel_speed:.1f} km/h, RTT: {current_rtt} ms"

        except Exception as e:
            logger.warning(f"Validation failed for record: {e}")
            return True, f"Validation error - allowing: {e}"

    def process_physical_validation(self, chunk_size=50000):
        """Process validation in chunks for efficiency"""
        logger.info("Starting physical constraint validation...")
        logger.info(f"Processing in chunks of {chunk_size:,} records")
        logger.info("IP addresses will be preserved for downstream analysis")

        validated_records = []
        rejected_records = []

        try:
            # Read and process data
            df = pd.read_csv(self.input_file, low_memory=False)
            logger.info(f"Loaded {len(df):,} records from {self.input_file}")

            # Ensure required columns exist
            required_cols = ['Login Timestamp', 'User ID', 'Country', 'Is_Anomaly',
                             'Round-Trip Time [ms]', 'Device Type', 'IP Address']

            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col}, adding default values")
                    if col == 'Is_Anomaly':
                        df[col] = 0
                    elif col == 'Round-Trip Time [ms]':
                        df[col] = 0.0
                    else:
                        df[col] = 'unknown'

            # Sort by user and timestamp for sequential validation
            df['Login Timestamp'] = df['Login Timestamp'].astype(str)
            df = df.sort_values(['User ID', 'Login Timestamp']).reset_index(drop=True)

            # Process records
            for idx, record in df.iterrows():
                self.validation_stats['total_records_processed'] += 1

                is_valid, reason = self.validate_record_sequence(record.to_dict())

                if is_valid:
                    validated_records.append(record.to_dict())
                    self.validation_stats['records_validated'] += 1
                else:
                    rejected_records.append({**record.to_dict(), 'rejection_reason': reason})
                    self.validation_stats['records_rejected'] += 1

                    # Track rejection types
                    if "Impossible travel" in reason:
                        self.validation_stats['impossible_travel_rejected'] += 1
                    elif "RTT" in reason:
                        self.validation_stats['rtt_anomaly_rejected'] += 1
                    elif "device" in reason:
                        self.validation_stats['device_inconsistency_rejected'] += 1

                # Progress logging
                if self.validation_stats['total_records_processed'] % 10000 == 0:
                    elapsed_minutes = (time.time() - self.start_time) / 60
                    validation_rate = (self.validation_stats['records_validated'] /
                                       max(self.validation_stats['total_records_processed'], 1))
                    logger.info(
                        f"Processed {self.validation_stats['total_records_processed']:,} | "
                        f"Validated: {self.validation_stats['records_validated']:,} | "
                        f"Rate: {validation_rate:.1%} | "
                        f"Users: {len(self.user_profiles):,} | "
                        f"Time: {elapsed_minutes:.1f}min"
                    )

            # Final statistics
            self.validation_stats['user_profiles_created'] = len(self.user_profiles)
            self.validation_stats['validation_rate'] = (self.validation_stats['records_validated'] /
                                                        max(self.validation_stats['total_records_processed'], 1))

            # Convert to DataFrames
            validated_df = pd.DataFrame(validated_records) if validated_records else pd.DataFrame()
            rejected_df = pd.DataFrame(rejected_records) if rejected_records else pd.DataFrame()

            return validated_df, rejected_df

        except Exception as e:
            logger.error(f"Physical validation failed: {e}")
            raise

    def save_validated_data_and_metadata(self, validated_df, rejected_df):
        """Save validated data and generate metadata"""
        try:
            logger.info(f"Saving {len(validated_df):,} validated records...")

            # Save validated records
            validated_df.to_csv(self.output_file, index=False)

            # Save rejected records for audit
            rejected_file = self.output_file.replace('.csv', '_rejected.csv')
            if len(rejected_df) > 0:
                rejected_df.to_csv(rejected_file, index=False)
                logger.info(f"Saved {len(rejected_df):,} rejected records to {rejected_file}")

            total_time = (time.time() - self.start_time) / 60
            original_attacks = validated_df['Is_Anomaly'].sum() if len(validated_df) > 0 else 0
            attack_preservation_rate = (self.validation_stats['attack_records_preserved'] /
                                        max(original_attacks, 1))

            # Generate comprehensive metadata
            metadata = {
                'stage2_completed': datetime.now().isoformat(),
                'validation_method': 'adaptive_velocity_rtt_device_validation',
                'input_file': self.input_file,
                'output_file': self.output_file,
                'rejected_file': rejected_file,
                'ip_addresses_preserved': True,

                'processing_results': {
                    'original_records': self.validation_stats['total_records_processed'],
                    'validated_records': len(validated_df),
                    'rejected_records': self.validation_stats['records_rejected'],
                    'validation_rate': float(self.validation_stats['validation_rate']),
                    'processing_time_minutes': float(total_time)
                },

                'rejection_breakdown': {
                    'impossible_travel_rejected': self.validation_stats['impossible_travel_rejected'],
                    'rtt_anomaly_rejected': self.validation_stats['rtt_anomaly_rejected'],
                    'device_inconsistency_rejected': self.validation_stats['device_inconsistency_rejected']
                },

                'attack_preservation': {
                    'attack_records_preserved': self.validation_stats['attack_records_preserved'],
                    'attack_preservation_rate': float(attack_preservation_rate),
                    'total_anomalies_in_output': int(original_attacks)
                },

                'validation_parameters': {
                    'max_travel_speed_kmh': self.max_travel_speed_kmh,
                    'min_time_threshold_minutes': self.min_time_threshold_minutes,
                    'max_time_threshold_hours': self.max_time_threshold_hours,
                    'rtt_thresholds': self.rtt_thresholds,
                    'countries_with_coordinates': len(self.country_coords)
                },

                'user_profile_statistics': {
                    'total_profiles': len(self.user_profiles),
                    'frequent_travelers': sum(1 for p in self.user_profiles.values() if p['is_frequent_traveler']),
                    'avg_locations_per_user': float(np.mean(
                        [len(p['locations']) for p in self.user_profiles.values()])) if self.user_profiles else 0.0,
                    'avg_rtt': float(np.mean([p['avg_rtt'] for p in self.user_profiles.values() if
                                              p['avg_rtt'] > 0])) if self.user_profiles else 0.0
                },

                'ready_for_stage3': True
            }

            # Save metadata
            with open('stage2_physical_validation_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # Results summary
            logger.info("=" * 70)
            logger.info("STAGE 2: PHYSICAL VALIDATION COMPLETED!")
            logger.info("=" * 70)
            logger.info(f"Original records: {metadata['processing_results']['original_records']:,}")
            logger.info(f"Validated records: {metadata['processing_results']['validated_records']:,}")
            logger.info(f"Validation rate: {metadata['processing_results']['validation_rate']:.1%}")
            logger.info(
                f"Impossible travel rejected: {metadata['rejection_breakdown']['impossible_travel_rejected']:,}")
            logger.info(f"RTT anomalies rejected: {metadata['rejection_breakdown']['rtt_anomaly_rejected']:,}")
            logger.info(
                f"Device inconsistencies rejected: {metadata['rejection_breakdown']['device_inconsistency_rejected']:,}")
            logger.info(f"Attack records preserved: {metadata['attack_preservation']['attack_records_preserved']:,}")
            logger.info(f"Processing time: {metadata['processing_results']['processing_time_minutes']:.1f} minutes")
            logger.info(f"IP addresses preserved: {metadata['ip_addresses_preserved']}")
            logger.info(f"Output: {self.output_file}")

            # Performance validation
            if total_time <= 5.0:
                logger.info("✓ PERFORMANCE TARGET MET (under 5 minutes)")
            else:
                logger.warning(f"Performance target missed (took {total_time:.1f} minutes)")

            # Validation rate assessment
            if 0.4 <= metadata['processing_results']['validation_rate'] <= 0.6:
                logger.info("✓ VALIDATION RATE TARGET MET (~50%)")
            elif 0.3 <= metadata['processing_results']['validation_rate'] < 0.4:
                logger.info("⚠ VALIDATION RATE LOWER THAN TARGET (30-40%)")
            elif 0.6 < metadata['processing_results']['validation_rate'] <= 0.7:
                logger.info("⚠ VALIDATION RATE HIGHER THAN TARGET (60-70%)")
            else:
                logger.warning(
                    f"Validation rate significantly off-target: {metadata['processing_results']['validation_rate']:.1%}")

            logger.info("READY FOR STAGE 3: FEATURE ENGINEERING & ANOMALY DETECTION")
            logger.info("=" * 70)

            return metadata

        except Exception as e:
            logger.error(f"Save and metadata generation failed: {e}")
            raise

    def run_physical_validation(self):
        """Run the complete physical validation process"""
        try:
            logger.info("OAUTH ANOMALY DETECTION - STAGE 2: PHYSICAL VALIDATION")
            logger.info("Adaptive velocity, RTT, and device validation with attack preservation")
            logger.info("IP addresses will be preserved for downstream analysis")

            validated_df, rejected_df = self.process_physical_validation()
            metadata = self.save_validated_data_and_metadata(validated_df, rejected_df)

            return self.output_file, metadata

        except Exception as e:
            logger.error(f"Stage 2 physical validation failed: {e}")
            raise


def main():
    """Main execution"""
    import os

    if not os.path.exists('stage1_balanced_sampled_rba.csv'):
        logger.error("Input file 'stage1_balanced_sampled_rba.csv' not found!")
        logger.info("Please ensure Stage 1 has been completed successfully.")
        return None, None

    validator = PhysicalValidator(
        input_file='stage1_balanced_sampled_rba.csv',
        output_file='stage2_validated_rba.csv'
    )

    try:
        output_file, metadata = validator.run_physical_validation()
        return output_file, metadata
    except Exception as e:
        logger.error(f"Physical validation failed: {e}")
        return None, None


if __name__ == "__main__":
    main()