#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Solar Forecasting Data Merger
This script combines NIST PV CSV data with ERA5 meteorological data into a single dataset
for machine learning and time-series analysis.
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

def delete_idx_files(directory):
    """
    Delete all .idx files in the specified directory and its subdirectories.
    
    Args:
        directory: Directory to search for .idx files
    """
    logger.info(f"Cleaning up .idx files in: {directory}")
    idx_files = glob.glob(os.path.join(directory, "**/*.idx"), recursive=True)
    
    for idx_file in idx_files:
        try:
            os.remove(idx_file)
            logger.info(f"Deleted: {idx_file}")
        except Exception as e:
            logger.warning(f"Failed to delete {idx_file}: {str(e)}")
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
    match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', os.path.basename(filepath))
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return None

def process_pv_data(csv_dir):
    """
    Process all PV CSV files in the given directory.
    Assumes timestamps are already timezone-aware and correctly defined.
    
    Args:
        csv_dir: Directory containing NIST PV CSV files
        
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
    
    # Sort by timestamp
    combined_pv_data.sort_values('timestamp', inplace=True)
    
    return combined_pv_data

def process_era5_data(era5_dir):
    """
    Process ERA5 netCDF or GRIB files and convert to DataFrame.
    Assumes timestamps are already timezone-aware and correctly defined.
    
    Args:
        era5_dir: Directory containing ERA5 netCDF/GRIB files
        
    Returns:
        Pandas DataFrame with ERA5 data
    """
    logger.info(f"Processing ERA5 data from: {era5_dir}")
    
    # Clean up .idx files before processing
    delete_idx_files(era5_dir)
    
    # Find all netCDF files
    nc_files = glob.glob(os.path.join(era5_dir, "**/*.nc"), recursive=True)
    
    if nc_files:
        logger.info(f"Found {len(nc_files)} netCDF files")
        try:
            # Process netCDF files - try by variable first
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
        
    else:
        # If no netCDF files, try GRIB files
        grib_files = glob.glob(os.path.join(era5_dir, "**/*.grib"), recursive=True)
        grib_files.extend(glob.glob(os.path.join(era5_dir, "**/*.grb"), recursive=True))
        grib_files.extend(glob.glob(os.path.join(era5_dir, "**/*.grib2"), recursive=True))
        
        if not grib_files:
            raise FileNotFoundError(f"No netCDF or GRIB files found in {era5_dir} or its subfolders")
            
        logger.info(f"Found {len(grib_files)} GRIB files")
        era5_df = process_grib_files(grib_files)
    
    return era5_df

def merge_datasets(pv_df, era5_df, spatial_merge=False, pv_coords=None):
    """
    Merge PV and ERA5 datasets.
    Assumes both datasets have timestamps in the same timezone.
    
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
    
    # Round timestamps to nearest hour to improve matching
    logger.info("Rounding timestamps to nearest hour for better alignment")
    pv_df['timestamp_rounded'] = pv_df['timestamp'].dt.round('1H')
    era5_df['timestamp_rounded'] = era5_df['timestamp'].dt.round('1H')
    
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
    
    # Merge on rounded timestamps
    logger.info("Attempting merge on rounded timestamps")
    merged_data = pd.merge(
        pv_df, 
        era5_df,
        on='timestamp_rounded',
        how='left'  # Keep all PV data
    )
    
    # Remove temporary columns
    if 'timestamp_rounded' in merged_data.columns:
        merged_data.drop(columns=['timestamp_rounded'], inplace=True)
    
    # Ensure we keep timestamp from PV data
    if 'timestamp_x' in merged_data.columns and 'timestamp_y' in merged_data.columns:
        merged_data.rename(columns={'timestamp_x': 'timestamp'}, inplace=True)
        merged_data.drop(columns=['timestamp_y'], inplace=True)
    
    logger.info(f"Final merged dataset has {len(merged_data)} rows")
    
    return merged_data

def process_grib_files(grib_files):
    """
    Process GRIB files and convert to DataFrame using xarray.
    Assumes timestamps are already timezone-aware and correctly defined.
    
    Args:
        grib_files: List of GRIB file paths
        
    Returns:
        Pandas DataFrame with ERA5 data
    """
    logger.info("Processing GRIB files")
    
    all_dfs = []
    
    for grib_file in grib_files:
        try:
            # Open GRIB file with xarray
            ds = xr.open_dataset(grib_file, engine='cfgrib')
            df = ds.to_dataframe().reset_index()
            
            # Rename 'time' column to 'timestamp' if necessary
            if 'time' in df.columns and 'timestamp' not in df.columns:
                df.rename(columns={'time': 'timestamp'}, inplace=True)
            
            all_dfs.append(df)
            logger.info(f"Processed {grib_file} with {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to process {grib_file}: {str(e)}")
    
    if not all_dfs:
        raise ValueError("Could not process any GRIB files")
    
    # Combine all dataframes
    era5_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined {len(all_dfs)} dataframes with total {len(era5_df)} rows")
    
    return era5_df
def main():
    """Main function to process and merge datasets."""
    # CONFIGURATION
    PV_DATA_DIR = r"C:\Users\Mrudula\OneDrive\Desktop\EEE\nist"
    ERA5_DATA_DIR = r"C:\Users\Mrudula\OneDrive\Desktop\EEE\ERA5"
    OUTPUT_FILE = "merged_solar_data.csv"
    PV_COORDS = {'lat': 39.15, 'lon': -77.2}  # Center of the provided bounds
    
    try:
        # Step 1: Process PV data
        logger.info("Step 1: Processing PV data")
        pv_data = process_pv_data(PV_DATA_DIR)
        
        # Step 2: Process ERA5 data
        logger.info("Step 2: Processing ERA5 data")
        era5_data = process_era5_data(ERA5_DATA_DIR)
        
        # Step 3: Merge datasets
        logger.info("Step 3: Merging datasets")
        merged_data = merge_datasets(pv_data, era5_data, spatial_merge=True, pv_coords=PV_COORDS)
        
        # Step 4: Save the merged dataset
        logger.info(f"Step 4: Saving merged dataset to {OUTPUT_FILE}")
        merged_data.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Successfully saved merged dataset with {len(merged_data)} rows")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting Solar Forecasting Data Merger")
    try:
        main()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error("Exiting with errors")
    finally:
        logger.info("Script execution finished")