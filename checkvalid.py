import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def validate_solar_merge(csv_path, output_dir="validation_results"):
    """
    Comprehensive validation of the merged solar and ERA5 data.
    
    Args:
        csv_path: Path to the merged CSV file
        output_dir: Directory to save validation results and plots
    
    Returns:
        Dictionary with validation results
    """
    print(f"Loading merged data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "basic_info": {},
        "temporal_validation": {},
        "spatial_validation": {},
        "physical_validation": {},
        "data_quality": {},
        "recommendations": []
    }
    
    # Basic information
    results["basic_info"]["total_rows"] = len(df)
    results["basic_info"]["date_range"] = f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "No timestamp column found"
    results["basic_info"]["columns"] = list(df.columns)
    
    # Convert timestamp if it exists
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. TEMPORAL VALIDATION
    print("Performing temporal validation...")
    if 'timestamp' in df.columns:
        # Check for time continuity
        df = df.sort_values('timestamp')
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        time_diff_stats = df['time_diff'].describe()
        results["temporal_validation"]["time_diff_stats"] = time_diff_stats.to_dict()
        
        # Check for unexpectedly large gaps
        large_gaps = df[df['time_diff'] > time_diff_stats['75%'] * 1.5]
        results["temporal_validation"]["large_gaps_count"] = len(large_gaps)
        
        # Check temporal resolution
        time_resolution = time_diff_stats['50%']  # median time difference
        results["temporal_validation"]["estimated_resolution_seconds"] = time_resolution
        
        # Check for duplicate timestamps
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        results["temporal_validation"]["duplicate_timestamps"] = duplicate_timestamps
        
        # Plot time difference histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df['time_diff'].dropna(), bins=50)
        plt.title('Histogram of Time Differences Between Consecutive Records')
        plt.xlabel('Time Difference (seconds)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_diff_histogram.png")
        plt.close()
    else:
        results["temporal_validation"]["error"] = "No timestamp column found"
    
    # 2. SPATIAL VALIDATION (if latitude/longitude columns exist)
    print("Performing spatial validation...")
    spatial_columns = [col for col in df.columns if col.lower() in ['latitude', 'lat', 'longitude', 'lon', 'long']]
    
    if len(spatial_columns) >= 2:
        # Identify likely lat/lon columns
        lat_col = next((col for col in spatial_columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in spatial_columns if 'lon' in col.lower() or 'long' in col.lower()), None)
        
        if lat_col and lon_col:
            # Check for coordinate range
            results["spatial_validation"]["lat_range"] = [df[lat_col].min(), df[lat_col].max()]
            results["spatial_validation"]["lon_range"] = [df[lon_col].min(), df[lon_col].max()]
            
            # Check for coordinate uniqueness - are we using one consistent location?
            unique_coords = df[[lat_col, lon_col]].drop_duplicates()
            results["spatial_validation"]["unique_coordinates"] = len(unique_coords)
            
            # If multiple coordinates, check for coordinate coverage over time
            if len(unique_coords) > 1 and 'timestamp' in df.columns:
                # Plot coordinates over time
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(df[lon_col], df[lat_col], c=df['timestamp'].astype(np.int64), 
                                     cmap='viridis', alpha=0.5)
                plt.colorbar(scatter, label='Time')
                plt.title('Spatial Distribution of Data Points Over Time')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/spatial_distribution.png")
                plt.close()
    else:
        results["spatial_validation"]["message"] = "No latitude/longitude columns found for spatial validation"
    
    # 3. PHYSICAL VALIDATION
    print("Performing physical validation...")
    
    # Identify potential solar/PV power columns
    pv_columns = [col for col in df.columns if any(term in col.lower() for term in 
                                                 ['power', 'energy', 'output', 'generation', 'watt', 'kw', 'mw'])]
    
    # Identify potential solar radiation columns
    radiation_columns = [col for col in df.columns if any(term in col.lower() for term in 
                                                       ['radiation', 'irradiance', 'ghi', 'dni', 'irrad', 'solar'])]
    
    # Identify potential temperature columns
    temp_columns = [col for col in df.columns if any(term in col.lower() for term in 
                                                  ['temp', 'temperature'])]
    
    results["physical_validation"]["identified_pv_columns"] = pv_columns
    results["physical_validation"]["identified_radiation_columns"] = radiation_columns
    results["physical_validation"]["identified_temperature_columns"] = temp_columns
    
    # If we can identify key columns, check physical relationships
    if pv_columns and (radiation_columns or temp_columns):
        # Select the first column from each category
        pv_col = pv_columns[0]
        radiation_col = radiation_columns[0] if radiation_columns else None
        temp_col = temp_columns[0] if temp_columns else None
        
        # Check for correlations
        corr_columns = [col for col in [pv_col, radiation_col, temp_col] if col is not None]
        if len(corr_columns) > 1:
            correlation = df[corr_columns].corr()
            results["physical_validation"]["correlations"] = correlation.to_dict()
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm')
            plt.title('Correlation Between Key Variables')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_heatmap.png")
            plt.close()
        
        # Check for night-time PV production (physical impossibility)
        if 'timestamp' in df.columns and radiation_col:
            # Assuming radiation is zero or very low at night
            df['hour'] = df['timestamp'].dt.hour
            night_hours = df[(df['hour'] < 6) | (df['hour'] > 20)]
            night_power = night_hours[pv_col].mean()
            night_radiation = night_hours[radiation_col].mean() if radiation_col else np.nan
            
            results["physical_validation"]["night_power_mean"] = night_power
            results["physical_validation"]["night_radiation_mean"] = night_radiation
            
            if night_power > 0 and night_power > df[pv_col].mean() * 0.05:
                results["recommendations"].append(
                    "WARNING: Significant PV production detected during nighttime hours. This suggests possible timestamp misalignment."
                )
        
        # Check for suspicious day/night patterns
        if 'timestamp' in df.columns and radiation_col and pv_col:
            # Check one day of data for visualization
            sample_date = df['timestamp'].dt.date.mode()[0]
            day_data = df[df['timestamp'].dt.date == sample_date].copy()
            
            if len(day_data) > 0:
                # Plot time series for a single day
                plt.figure(figsize=(14, 8))
                plt.plot(day_data['timestamp'], day_data[pv_col], 'b-', label='PV Power')
                
                if radiation_col:
                    # Normalize radiation to same scale as PV for comparison
                    max_radiation = day_data[radiation_col].max()
                    max_pv = day_data[pv_col].max()
                    if max_radiation > 0 and max_pv > 0:
                        normalized_radiation = day_data[radiation_col] * (max_pv / max_radiation)
                        plt.plot(day_data['timestamp'], normalized_radiation, 'r--', 
                                label=f'Normalized {radiation_col}')
                
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title(f'PV Output and Solar Radiation on {sample_date}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/daily_pattern.png")
                plt.close()
                
                # Calculate phase shift between radiation and power
                if radiation_col and len(day_data) > 10:
                    try:
                        # Create hourly resampled data for robust comparison
                        day_data_resampled = day_data.set_index('timestamp').resample('1H').mean().reset_index()
                        
                        # Find peak time for each series
                        pv_peak_time = day_data_resampled.loc[day_data_resampled[pv_col].idxmax(), 'timestamp']
                        rad_peak_time = day_data_resampled.loc[day_data_resampled[radiation_col].idxmax(), 'timestamp']
                        
                        # Calculate time difference in minutes
                        time_diff_minutes = (pv_peak_time - rad_peak_time).total_seconds() / 60
                        
                        results["physical_validation"]["pv_radiation_peak_time_diff_minutes"] = time_diff_minutes
                        
                        if abs(time_diff_minutes) > 90:  # More than 1.5 hours difference
                            results["recommendations"].append(
                                f"WARNING: PV power and radiation peaks are {time_diff_minutes:.1f} minutes apart. This suggests possible temporal misalignment."
                            )
                    except Exception as e:
                        results["physical_validation"]["peak_analysis_error"] = str(e)
    
    # 4. DATA QUALITY CHECKS
    print("Performing data quality checks...")
    
    # Missing value analysis
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    results["data_quality"]["columns_with_missing_values"] = missing_counts[missing_counts > 0].to_dict()
    results["data_quality"]["missing_percentages"] = missing_percentages[missing_percentages > 0].to_dict()
    
    # Check for likely ERA5 vs PV columns based on missing patterns
    if missing_counts.max() > 0:
        # Columns with same missing pattern likely come from same source
        missing_patterns = df.isnull().sum(axis=1).value_counts()
        results["data_quality"]["missing_patterns"] = missing_patterns.to_dict()
        
        if len(missing_patterns) > 1:
            # There are different missing patterns - check if they align with expected merged data patterns
            results["data_quality"]["potential_data_source_groups"] = "Multiple missing value patterns detected, suggesting different data sources were merged"
    
    # Check for outliers in key columns
    key_columns = pv_columns + radiation_columns + temp_columns
    key_columns = [col for col in key_columns if col in df.columns]
    
    if key_columns:
        for col in key_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            results["data_quality"][f"outliers_{col}"] = len(outliers)
            
            if len(outliers) > 0:
                # Plot outliers for a key column
                plt.figure(figsize=(12, 6))
                plt.boxplot(df[col].dropna())
                plt.title(f'Boxplot of {col} Showing Outliers')
                plt.ylabel('Value')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/outliers_{col.replace(' ', '_')}.png")
                plt.close()
    
    # 5. OVERALL ASSESSMENT
    print("Creating overall assessment and recommendations...")
    
    # Add general recommendations based on findings
    if results["temporal_validation"].get("large_gaps_count", 0) > 0:
        results["recommendations"].append(
            f"Found {results['temporal_validation']['large_gaps_count']} large gaps in time series data. Consider checking for missing data periods."
        )
    
    if any(percentage > 10 for percentage in results["data_quality"].get("missing_percentages", {}).values()):
        results["recommendations"].append(
            "Some columns have >10% missing values. Consider imputation techniques or examining why data is missing."
        )
    
    # Save results to file
    with open(f"{output_dir}/validation_summary.txt", "w") as f:
        f.write("SOLAR DATA MERGE VALIDATION SUMMARY\n")
        f.write("==================================\n\n")
        
        f.write("BASIC INFORMATION\n")
        f.write(f"Total rows: {results['basic_info']['total_rows']}\n")
        f.write(f"Date range: {results['basic_info']['date_range']}\n")
        f.write(f"Columns: {', '.join(results['basic_info']['columns'])}\n\n")
        
        f.write("TEMPORAL VALIDATION\n")
        if 'error' not in results['temporal_validation']:
            f.write(f"Estimated data resolution: {results['temporal_validation']['estimated_resolution_seconds']} seconds\n")
            f.write(f"Duplicate timestamps: {results['temporal_validation']['duplicate_timestamps']}\n")
            f.write(f"Large time gaps detected: {results['temporal_validation']['large_gaps_count']}\n\n")
        else:
            f.write(f"Error: {results['temporal_validation']['error']}\n\n")
        
        f.write("SPATIAL VALIDATION\n")
        if 'message' not in results['spatial_validation']:
            f.write(f"Latitude range: {results['spatial_validation']['lat_range']}\n")
            f.write(f"Longitude range: {results['spatial_validation']['lon_range']}\n")
            f.write(f"Unique coordinate points: {results['spatial_validation']['unique_coordinates']}\n\n")
        else:
            f.write(f"{results['spatial_validation']['message']}\n\n")
        
        f.write("PHYSICAL VALIDATION\n")
        f.write(f"Identified PV columns: {', '.join(results['physical_validation']['identified_pv_columns'])}\n")
        f.write(f"Identified radiation columns: {', '.join(results['physical_validation']['identified_radiation_columns'])}\n")
        f.write(f"Identified temperature columns: {', '.join(results['physical_validation']['identified_temperature_columns'])}\n")
        
        if 'night_power_mean' in results['physical_validation']:
            f.write(f"Average night power: {results['physical_validation']['night_power_mean']}\n")
            f.write(f"Average night radiation: {results['physical_validation']['night_radiation_mean']}\n")
        
        if 'pv_radiation_peak_time_diff_minutes' in results['physical_validation']:
            f.write(f"Time difference between PV and radiation peaks: {results['physical_validation']['pv_radiation_peak_time_diff_minutes']} minutes\n\n")
        
        f.write("DATA QUALITY\n")
        f.write("Columns with missing values:\n")
        for col, count in results['data_quality'].get('columns_with_missing_values', {}).items():
            f.write(f"  - {col}: {count} missing values ({results['data_quality']['missing_percentages'].get(col, 0):.2f}%)\n")
        
        f.write("\nRECOMMENDATIONS\n")
        for i, rec in enumerate(results['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
    
    print(f"Validation complete! Results saved to {output_dir}/")
    return results

if __name__ == "__main__":
    # Replace with your merged file path
    merged_file = "merged_solar_data.csv"
    results = validate_solar_merge(merged_file)
    
    # Print key findings
    print("\nKEY FINDINGS:")
    for rec in results['recommendations']:
        print(f"- {rec}")