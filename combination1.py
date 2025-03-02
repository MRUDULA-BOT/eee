#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Solar Forecasting Data Merger
This script combines NIST PV CSV data with ERA5 meteorological data into a single dataset
for machine learning and time-series analysis, with improved data preservation.
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import glob
import re
from datetime import datetime, timezone
import pytz
from scipy import stats
import logging
import warnings
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('solar_data_merger')

# Suppress common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def extract_timestamp(filepath):
    """Extract date information from file name or path."""
    # Adjust this regex to match your file naming pattern
    match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', os.path.basename(filepath))
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return None

def process_pv_data(csv_dir, timezone_name='US/Eastern'):
    """
    Process all PV CSV files in the given directory with improved DST handling.
    
    Args:
        csv_dir: Directory containing NIST PV CSV files
        timezone_name: Timezone of the PV data
        
    Returns:
        Pandas DataFrame with combined PV data
    """
    logger.info(f"Processing PV data from: {csv_dir}")
    
    # Get list of all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True)
    logger.info(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    
    # Create empty list to store dataframes
    dfs = []
    
    # Process each CSV file
    for file in csv_files:
        # Extract date from filename
        date_str = extract_timestamp(file)
        if not date_str:
            logger.warning(f"Could not extract date from {file}, skipping...")
            continue
        
        # Read CSV (with error handling)
        try:
            # First try to determine if there's a header
            with open(file, 'r') as f:
                first_line = f.readline().strip()
            
            # If first line doesn't contain numbers, assume it's a header
            has_header = not all(c.isdigit() or c in ',.-+' for c in first_line.replace(',', ''))
            header = 0 if has_header else None
            
            df = pd.read_csv(file, header=header)
            
            # If no header, create column names
            if header is None:
                df.columns = [f"value_{i}" for i in range(len(df.columns))]
                
            # If no timestamp column exists, create one based on the file date and row index
            if 'timestamp' not in df.columns and 'time' not in df.columns and 'date' not in df.columns:
                # Assuming data is at 1-minute intervals
                df['timestamp'] = pd.date_range(start=date_str, periods=len(df), freq='1min')
            elif 'time' in df.columns:
                df.rename(columns={'time': 'timestamp'}, inplace=True)
            elif 'date' in df.columns:
                df.rename(columns={'date': 'timestamp'}, inplace=True)
                
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Add file source for debugging
            df['source_file'] = os.path.basename(file)
            
            dfs.append(df)
            logger.info(f"Processed {file} with {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid data found in any CSV files")
        
    # Combine all dataframes
    logger.info("Combining all PV dataframes")
    combined_pv_data = pd.concat(dfs, ignore_index=True)
    
    # Handle timezone if needed - IMPROVED DST HANDLING
    if timezone_name:
        local_tz = pytz.timezone(timezone_name)
        # Check if timestamps are already timezone-aware
        if combined_pv_data['timestamp'].dt.tz is None:
            # Localize timestamps if they don't have timezone info
            try:
                combined_pv_data['timestamp'] = combined_pv_data['timestamp'].dt.tz_localize(
                    local_tz, nonexistent='shift_forward', ambiguous='NaT'
                )
                
                # Find any NaT values created by ambiguous times and fix them
                nat_mask = pd.isna(combined_pv_data['timestamp'])
                if nat_mask.any():
                    logger.warning(f"Found {nat_mask.sum()} ambiguous timestamps during DST transition")
                    
                    # For ambiguous times (NaT values), set them to UTC first then convert
                    temp_df = combined_pv_data[nat_mask].copy()
                    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp']).dt.tz_localize('UTC')
                    temp_df['timestamp'] = temp_df['timestamp'].dt.tz_convert(local_tz)
                    
                    # Replace the NaT values
                    combined_pv_data.loc[nat_mask, 'timestamp'] = temp_df['timestamp']
                    
            except Exception as e:
                logger.warning(f"Issue with timezone localization: {str(e)}")
                logger.info("Trying alternative approach with UTC first...")
                # Alternative approach: first localize to UTC, then convert
                combined_pv_data['timestamp'] = combined_pv_data['timestamp'].dt.tz_localize('UTC')
                combined_pv_data['timestamp'] = combined_pv_data['timestamp'].dt.tz_convert(local_tz)
        else:
            # If already timezone-aware, convert to the desired timezone
            combined_pv_data['timestamp'] = combined_pv_data['timestamp'].dt.tz_convert(local_tz)
    
    # Sort by timestamp
    combined_pv_data.sort_values('timestamp', inplace=True)
    
    return combined_pv_data

def process_era5_data(era5_dir, target_timezone='US/Eastern'):
    """
    Process ERA5 netCDF or GRIB files and convert to DataFrame.
    Fixed to handle time coordinate conflicts between files.
    
    Args:
        era5_dir: Directory containing ERA5 netCDF/GRIB files
        target_timezone: Target timezone for the timestamps
        
    Returns:
        Pandas DataFrame with ERA5 data
    """
    logger.info(f"Processing ERA5 data from: {era5_dir}")
    
    # Find all netCDF files
    nc_files = glob.glob(os.path.join(era5_dir, "**/*.nc"), recursive=True)
    
    if nc_files:
        logger.info(f"Found {len(nc_files)} netCDF files")
        try:
            # Process netCDF files - try by variable first
            return process_netcdf_files(nc_files, target_timezone)
        except Exception as e:
            logger.error(f"Error opening netCDF files: {str(e)}")
            logger.info("Falling back to GRIB files")
            nc_files = []
    
    # If no netCDF files or they failed to open, try GRIB files
    if not nc_files:
        grib_files = glob.glob(os.path.join(era5_dir, "**/*.grib"), recursive=True)
        grib_files.extend(glob.glob(os.path.join(era5_dir, "**/*.grb"), recursive=True))
        grib_files.extend(glob.glob(os.path.join(era5_dir, "**/*.grib2"), recursive=True))
        
        if not grib_files:
            raise FileNotFoundError(f"No netCDF or GRIB files found in {era5_dir} or its subfolders")
            
        logger.info(f"Found {len(grib_files)} GRIB files")
        return process_grib_files(grib_files, target_timezone)
    
def process_netcdf_files(nc_files, target_timezone):
    """Process netCDF files and handle potential conflicts"""
    try:
        # First try the standard approach
        era5_data = xr.open_mfdataset(nc_files, combine='by_coords')
        logger.info("Successfully opened ERA5 netCDF data using standard approach")
    except Exception as e:
        logger.warning(f"Standard approach failed: {str(e)}")
        logger.info("Trying to process files individually...")
        
        # Process files individually
        datasets = []
        for nc_file in nc_files:
            try:
                ds = xr.open_dataset(nc_file)
                datasets.append(ds)
                logger.info(f"Successfully processed {nc_file}")
            except Exception as file_e:
                logger.warning(f"Could not open {nc_file}: {str(file_e)}")
        
        if not datasets:
            raise ValueError("Could not open any netCDF files")
        
        # Try to merge datasets
        try:
            era5_data = xr.merge(datasets)
            logger.info("Successfully merged individual netCDF datasets")
        except Exception as merge_e:
            logger.warning(f"Could not merge datasets: {str(merge_e)}")
            # Just use the first dataset as fallback
            era5_data = datasets[0]
            logger.info("Using first dataset only due to merge conflict")
    
    # Convert to dataframe
    era5_df = era5_data.to_dataframe().reset_index()
    
    # Handle timezone
    process_timestamps(era5_df, target_timezone)
    
    return era5_df

def process_grib_files(grib_files, target_timezone):
    """Process GRIB files with handling for the time coordinate conflict"""
    
    # Try processing by variable to avoid time conflicts
    try:
        # Step 1: Group files by variable if possible
        variable_groups = {}
        
        # Try to open first file to see variables
        try:
            sample_ds = xr.open_dataset(grib_files[0], engine='cfgrib')
            var_names = list(sample_ds.data_vars)
            logger.info(f"Found variables: {var_names}")
            
            # Initialize empty lists for each variable
            for var in var_names:
                variable_groups[var] = []
        except Exception:
            # If we can't open the file to get variables, use a simpler approach
            logger.warning("Couldn't inspect variables, processing files individually")
            return process_grib_individual_files(grib_files, target_timezone)
        
        # Step 2: Process files individually and collect dataframes
        all_dfs = []
        
        for grib_file in grib_files:
            try:
                logger.info(f"Processing {grib_file}")
                # Open the file specifically for each variable to avoid conflicts
                for var in var_names:
                    try:
                        # Try to filter by variable
                        ds = xr.open_dataset(
                            grib_file, 
                            engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'shortName': var}}
                        )
                        
                        # Convert to dataframe
                        df = ds.to_dataframe().reset_index()
                        
                        # Standardize column names
                        if 'time' in df.columns and 'timestamp' not in df.columns:
                            df.rename(columns={'time': 'timestamp'}, inplace=True)
                        
                        all_dfs.append(df)
                        logger.info(f"Successfully processed variable {var} from {grib_file}")
                    except Exception as var_e:
                        logger.debug(f"Could not process variable {var} from {grib_file}: {str(var_e)}")
            except Exception as e:
                logger.warning(f"Could not process {grib_file}: {str(e)}")
        
        if not all_dfs:
            logger.warning("Variable-based approach failed, trying individual file processing")
            return process_grib_individual_files(grib_files, target_timezone)
        
        # Combine all dataframes
        era5_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined {len(all_dfs)} dataframes with total {len(era5_df)} rows")
        
        # Handle duplicates that might occur from variable overlap
        era5_df = era5_df.drop_duplicates()
        
    except Exception as e:
        logger.warning(f"Variable processing approach failed: {str(e)}")
        logger.info("Falling back to individual file processing")
        return process_grib_individual_files(grib_files, target_timezone)
    
    # Handle timezone
    process_timestamps(era5_df, target_timezone)
    
    return era5_df

def process_grib_individual_files(grib_files, target_timezone):
    """Process GRIB files individually and combine the results"""
    all_dfs = []
    
    for grib_file in grib_files:
        try:
            # Try to open with cfgrib.open_datasets to handle multiple messages
            import cfgrib
            datasets = cfgrib.open_datasets(grib_file)
            
            for ds in datasets:
                df = ds.to_dataframe().reset_index()
                if 'time' in df.columns and 'timestamp' not in df.columns:
                    df.rename(columns={'time': 'timestamp'}, inplace=True)
                all_dfs.append(df)
                logger.info(f"Processed dataset from {grib_file} with {len(df)} rows")
                
        except Exception as e:
            logger.warning(f"cfgrib.open_datasets failed for {grib_file}: {str(e)}")
            
            try:
                # Try simple xarray open as fallback
                ds = xr.open_dataset(grib_file, engine='cfgrib')
                df = ds.to_dataframe().reset_index()
                if 'time' in df.columns and 'timestamp' not in df.columns:
                    df.rename(columns={'time': 'timestamp'}, inplace=True)
                all_dfs.append(df)
                logger.info(f"Processed {grib_file} with {len(df)} rows using fallback method")
            except Exception as e2:
                logger.error(f"All methods failed for {grib_file}: {str(e2)}")
    
    if not all_dfs:
        raise ValueError("Could not process any GRIB files")
    
    # Combine all dataframes
    era5_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined {len(all_dfs)} dataframes with total {len(era5_df)} rows")
    
    # Handle timezone
    process_timestamps(era5_df, target_timezone)
    
    return era5_df


def process_timestamps(df, target_timezone):
    """Process timestamps to ensure consistent timezone handling with improved DST handling"""
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns and 'time' in df.columns:
        df.rename(columns={'time': 'timestamp'}, inplace=True)
    
    # Handle timezone
    if 'timestamp' in df.columns:
        # ERA5 data is typically in UTC
        if df['timestamp'].dt.tz is None:
            # Handle DST transitions safely
            try:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            except Exception as e:
                logger.warning(f"UTC localization issue: {str(e)}")
                # Alternative approach for problem timestamps
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
        # Convert to target timezone with proper DST handling
        if target_timezone:
            try:
                df['timestamp'] = df['timestamp'].dt.tz_convert(target_timezone)
            except Exception as e:
                logger.warning(f"Timezone conversion issue: {str(e)}")
                # Keep UTC if conversion fails
                pass

def find_common_columns(df1, df2):
    """
    Find common columns between two DataFrames, ignoring timestamp/time
    which will be handled separately.
    """
    time_cols = ['timestamp', 'time', 'date']
    common_cols = [col for col in df1.columns if col in df2.columns and col not in time_cols]
    return common_cols

def merge_datasets(pv_df, era5_df, spatial_merge=False, pv_coords=None):
    """
    Merge PV and ERA5 datasets with improved time alignment and DST handling.
    
    Args:
        pv_df: DataFrame with PV data
        era5_df: DataFrame with ERA5 data
        spatial_merge: Whether to merge based on spatial coordinates
        pv_coords: Dict with 'lat' and 'lon' if spatial_merge is True
        
    Returns:
        Merged DataFrame
    """
    logger.info("Merging PV and ERA5 datasets")
    
    # Check and standardize timestamp columns
    for df, name in [(pv_df, 'PV'), (era5_df, 'ERA5')]:
        if 'timestamp' not in df.columns:
            raise ValueError(f"'timestamp' column missing in {name} data")
        
        # Ensure timestamps are properly formatted
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            logger.warning(f"Converting {name} timestamp to datetime")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure both dataframes have the same timezone or no timezone
    tz_pv = pv_df['timestamp'].dt.tz
    tz_era5 = era5_df['timestamp'].dt.tz
    
    # Handle timezone differences with proper DST handling
    if tz_pv is not None and tz_era5 is None:
        try:
            era5_df['timestamp'] = era5_df['timestamp'].dt.tz_localize(tz_pv, ambiguous='NaT')
        except Exception as e:
            logger.warning(f"ERA5 timezone localization issue: {str(e)}")
            # Try alternative approach
            era5_df['timestamp'] = pd.to_datetime(era5_df['timestamp'], utc=True).dt.tz_convert(tz_pv)
    elif tz_pv is None and tz_era5 is not None:
        try:
            pv_df['timestamp'] = pv_df['timestamp'].dt.tz_localize(tz_era5, ambiguous='NaT')
        except Exception as e:
            logger.warning(f"PV timezone localization issue: {str(e)}")
            # Try alternative approach
            pv_df['timestamp'] = pd.to_datetime(pv_df['timestamp'], utc=True).dt.tz_convert(tz_era5)
    elif tz_pv != tz_era5 and tz_pv is not None and tz_era5 is not None:
        # Convert ERA5 timezone to match PV
        try:
            era5_df['timestamp'] = era5_df['timestamp'].dt.tz_convert(tz_pv)
        except Exception as e:
            logger.warning(f"ERA5 timezone conversion issue: {str(e)}")
            # Fall back to UTC for both
            pv_df['timestamp'] = pv_df['timestamp'].dt.tz_convert('UTC')
            era5_df['timestamp'] = era5_df['timestamp'].dt.tz_convert('UTC')
    
    # Round timestamps to nearest hour to improve matching and avoid DST issues
    logger.info("Rounding timestamps to nearest hour for better alignment and to avoid DST issues")
    pv_df['timestamp_rounded'] = pv_df['timestamp'].dt.round('1H')
    era5_df['timestamp_rounded'] = era5_df['timestamp'].dt.round('1H')
    
    # Rest of the merge function remains unchanged...
    # Check for common columns (excluding timestamp)
    common_cols = find_common_columns(pv_df, era5_df)
    if common_cols:
        logger.info(f"Found common columns for alternative merge approach: {common_cols}")
    
    # Frequency analysis for proper resampling
    pv_freq = pd.infer_freq(pv_df['timestamp'])
    era5_freq = pd.infer_freq(era5_df['timestamp'])
    
    logger.info(f"PV data frequency: {pv_freq}, ERA5 frequency: {era5_freq}")
    
    # If frequencies can't be determined automatically, estimate them
    if pv_freq is None:
        pv_timedeltas = pv_df['timestamp'].diff().dropna()
        if not pv_timedeltas.empty:
            most_common_delta = pv_timedeltas.mode().iloc[0]
            pv_freq_str = f"{int(most_common_delta.total_seconds())}S"
            logger.info(f"Estimated PV frequency: {pv_freq_str}")
        else:
            pv_freq_str = '1min'  # Default assumption
    else:
        pv_freq_str = pv_freq
        
    if era5_freq is None:
        era5_timedeltas = era5_df['timestamp'].diff().dropna()
        if not era5_timedeltas.empty:
            most_common_delta = era5_timedeltas.mode().iloc[0]
            era5_freq_str = f"{int(most_common_delta.total_seconds())}S"
            logger.info(f"Estimated ERA5 frequency: {era5_freq_str}")
        else:
            era5_freq_str = '1H'  # Common for ERA5
    else:
        era5_freq_str = era5_freq
    
    # Spatial merge if requested
    if spatial_merge and pv_coords:
        logger.info("Performing spatial merge based on coordinates")
        if 'latitude' in era5_df.columns and 'longitude' in era5_df.columns:
            # Calculate distance to find closest point
            era5_df['distance'] = np.sqrt(
                (era5_df['latitude'] - pv_coords['lat'])**2 + 
                (era5_df['longitude'] - pv_coords['lon'])**2
            )
            
            # Get the closest point data for each timestamp
            closest_points = era5_df.sort_values('distance').groupby('timestamp_rounded').first().reset_index()
            era5_df = closest_points.drop(columns=['distance'])
        else:
            logger.warning("Spatial merge requested but latitude/longitude not found in ERA5 data")
    
    # Try exact timestamp merge first
    logger.info("Attempting merge on rounded timestamps")
    merged_data = pd.merge(
        pv_df, 
        era5_df,
        on='timestamp_rounded',
        how='left'  # Keep all PV data
    )
    
    # Check merge quality
    merge_quality = (merged_data.isnull().sum() / len(merged_data)).mean()
    logger.info(f"Initial merge quality (% non-null values): {(1-merge_quality)*100:.2f}%")
    
    if merge_quality > 0.5:  # If more than 50% of values are missing
        logger.warning("Poor merge quality. Trying alternative merge strategies...")
        
        # Try nearest timestamp approach using merge_asof
        logger.info("Trying merge_asof with nearest timestamp match")
        
        # Sort by timestamp for merge_asof
        pv_df = pv_df.sort_values('timestamp')
        era5_df = era5_df.sort_values('timestamp')
        
        # Calculate maximum allowed time difference (e.g., 1 hour)
        max_time_diff = pd.Timedelta('1 hour')
        
        merged_data = pd.merge_asof(
            pv_df, 
            era5_df,
            on='timestamp',
            direction='nearest',
            tolerance=max_time_diff
        )
        
        # Drop the rounded timestamp columns
        if 'timestamp_rounded_x' in merged_data.columns:
            merged_data.drop(columns=['timestamp_rounded_x'], inplace=True)
        if 'timestamp_rounded_y' in merged_data.columns:
            merged_data.drop(columns=['timestamp_rounded_y'], inplace=True)
        if 'timestamp_rounded' in merged_data.columns:
            merged_data.drop(columns=['timestamp_rounded'], inplace=True)
            
        # Check quality of asof merge
        new_merge_quality = (merged_data.isnull().sum() / len(merged_data)).mean()
        logger.info(f"Merge_asof quality (% non-null values): {(1-new_merge_quality)*100:.2f}%")
        
        if new_merge_quality > merge_quality:
            logger.info("Merge_asof approach yielded better results")
        else:
            logger.warning("Merge_asof approach did not improve results")
            
            # If common columns exist, try merging on those as a last resort
            if common_cols:
                logger.info(f"Trying merge on common columns: {common_cols}")
                merged_data = pd.merge(
                    pv_df, 
                    era5_df,
                    on=common_cols,
                    how='left'
                )
                
                # Check quality of common column merge
                common_merge_quality = (merged_data.isnull().sum() / len(merged_data)).mean()
                logger.info(f"Common column merge quality: {(1-common_merge_quality)*100:.2f}%")
    
    # Remove temporary columns
    if 'timestamp_rounded' in merged_data.columns:
        merged_data.drop(columns=['timestamp_rounded'], inplace=True)
    
    # Ensure we keep timestamp from PV data
    if 'timestamp_x' in merged_data.columns and 'timestamp_y' in merged_data.columns:
        merged_data.rename(columns={'timestamp_x': 'timestamp'}, inplace=True)
        merged_data.drop(columns=['timestamp_y'], inplace=True)
    
    logger.info(f"Final merged dataset has {len(merged_data)} rows")
    
    # Report on data retention
    pv_rows = len(pv_df)
    era5_rows = len(era5_df)
    merged_rows = len(merged_data)
    
    logger.info(f"Original PV rows: {pv_rows}, ERA5 rows: {era5_rows}, Merged rows: {merged_rows}")
    logger.info(f"Data retention rate: {merged_rows/pv_rows:.2%} of PV data")
    
    return merged_data

def clean_dataset(df, critical_columns=None):
    """
    Clean the merged dataset using z-score normalization and imputation instead of removal.
    
    Args:
        df: DataFrame to clean
        critical_columns: List of columns that must be imputed if null
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning merged dataset with data preservation")
    original_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    logger.info(f"Removed {original_rows - len(df)} duplicate rows")
    
    # Handle missing values with imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude columns that shouldn't be imputed
    exclude_from_imputation = ['timestamp', 'year', 'month', 'day', 'hour', 'minute']
    numeric_cols = [col for col in numeric_cols if col not in exclude_from_imputation]
    
    # For critical columns, apply imputation
    if critical_columns:
        imputation_cols = [col for col in critical_columns if col in numeric_cols]
        if imputation_cols:
            logger.info(f"Imputing missing values in critical columns: {imputation_cols}")
            
            # Create imputer - using median for robustness
            imputer = SimpleImputer(strategy='median')
            
            # Apply imputation
            df[imputation_cols] = imputer.fit_transform(df[imputation_cols])
            
            logger.info("Completed imputation of critical columns")
    
    # For other numeric columns with missing values, apply different imputation strategy
    other_numeric_cols = [col for col in numeric_cols if col not in (critical_columns or [])]
    if other_numeric_cols:
        missing_counts = df[other_numeric_cols].isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
        
        if cols_with_missing:
            logger.info(f"Imputing missing values in {len(cols_with_missing)} non-critical columns")
            
            # For time series data, forward fill then backward fill is often good
            df[cols_with_missing] = df[cols_with_missing].fillna(method='ffill').fillna(method='bfill')
            
            # For any remaining missing values, use median imputation
            remaining_missing = df[cols_with_missing].isnull().sum().sum()
            if remaining_missing > 0:
                logger.info(f"Using median imputation for {remaining_missing} remaining missing values")
                imputer = SimpleImputer(strategy='median')
                df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])
    
    # Handle outliers with Z-score normalization instead of removal
    logger.info("Applying Z-score normalization to handle outliers")
    normalized_cols = {}
    
    for col in numeric_cols:
        # Calculate mean and standard deviation
        mean = df[col].mean()
        std = df[col].std()
        
        # Skip columns with no standard deviation
        if std == 0 or pd.isna(std):
            continue
        
        # Identify outliers (|z| > 3)
        z_scores = np.abs((df[col] - mean) / std)
        outliers = (z_scores > 3)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            logger.info(f"Found {outlier_count} outliers in column {col}")
            
            # Store original column
            orig_col_name = f"{col}_orig"
            df[orig_col_name] = df[col].copy()
            
            # Apply normalization to outliers
            # Method 1: Cap at 3 standard deviations
            df.loc[outliers, col] = df.loc[outliers, col].apply(
                lambda x: mean + 3*std if x > mean + 3*std else (mean - 3*std if x < mean - 3*std else x)
            )
            
            # Alternative: IQR method
            """
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            """
            
            normalized_cols[col] = outlier_count
    
    if normalized_cols:
        logger.info(f"Z-score normalization applied to {len(normalized_cols)} columns")
        for col, count in normalized_cols.items():
            logger.info(f"  - {col}: {count} values normalized")
    
    logger.info(f"Final dataset has {len(df)} rows with all data preserved")
    return df

def main():
    """Main function to process and merge datasets."""
    # CONFIGURATION
    # Replace these paths with your actual data paths
    PV_DATA_DIR = r"C:\Users\Mrudula\OneDrive\Desktop\EEE\nist"
    ERA5_DATA_DIR = r"C:\Users\Mrudula\OneDrive\Desktop\EEE\ERA5"
    OUTPUT_FILE = "merged_solar_data.csv"
    
    # Timezone settings - Try to determine automatically or use specified
    # For the specific region (39.3N, 39S, -77E, -77.4W), the timezone is US/Eastern
    PV_TIMEZONE = "US/Eastern"
    
    # Critical columns that must be imputed if null
    CRITICAL_COLUMNS = []  # Add your critical column names here
    
    # PV site coordinates (based on provided bounds)
    # Calculate center point from the provided coordinates
    PV_COORDS = {
        'lat': 39.15,  # Center of 39.3N and 39S
        'lon': -77.2   # Center of -77E and -77.4W
    }
    
    # PROCESSING
    try:
        # Step 1: Process PV data
        logger.info("Step 1: Processing PV data")
        pv_data = process_pv_data(PV_DATA_DIR, PV_TIMEZONE)
        
        # Step 2: Process ERA5 data
        logger.info("Step 2: Processing ERA5 data")
        era5_data = process_era5_data(ERA5_DATA_DIR, PV_TIMEZONE)
        
        # Step 3: Merge datasets
        logger.info("Step 3: Merging datasets")
        merged_data = merge_datasets(pv_data, era5_data, 
                                    spatial_merge=True, 
                                    pv_coords=PV_COORDS)
        
        # Step 4: Clean the merged dataset
        logger.info("Step 4: Cleaning merged dataset")
        cleaned_data = clean_dataset(merged_data, critical_columns=CRITICAL_COLUMNS)
        
        # Step 5: Add time features for ML
        logger.info("Step 5: Adding time features")
        # Extract time features useful for machine learning
        cleaned_data['year'] = cleaned_data['timestamp'].dt.year
        cleaned_data['month'] = cleaned_data['timestamp'].dt.month
        cleaned_data['day'] = cleaned_data['timestamp'].dt.day
        cleaned_data['hour'] = cleaned_data['timestamp'].dt.hour
        cleaned_data['minute'] = cleaned_data['timestamp'].dt.minute
        cleaned_data['dayofweek'] = cleaned_data['timestamp'].dt.dayofweek
        cleaned_data['dayofyear'] = cleaned_data['timestamp'].dt.dayofyear
        # Add solar time features
        cleaned_data['is_day'] = ((cleaned_data['hour'] >= 6) & (cleaned_data['hour'] <= 18)).astype(int)
        
        # Step 6: Save the merged dataset
        logger.info(f"Step 6: Saving merged dataset to {OUTPUT_FILE}")
        cleaned_data.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Successfully saved merged dataset with {len(cleaned_data)} rows")
        
        # Print summary statistics
        logger.info("Summary of merged dataset:")
        logger.info(f"Total rows: {len(cleaned_data)}")
        logger.info(f"Date range: {cleaned_data['timestamp'].min()} to {cleaned_data['timestamp'].max()}")
        logger.info(f"Columns: {', '.join(cleaned_data.columns)}")
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting Solar Forecasting Data Merger")
    try:
        result = main()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error("Exiting with errors")
    finally:
        logger.info("Script execution finished")