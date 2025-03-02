import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = r"C:\Users\Mrudula\OneDrive\Desktop\EEE\merged_solar_data.csv"  # Change this to your actual file path
data = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])

def check_missing_data(df):
    """Function to check missing data percentage per column."""
    missing_data = df.isnull().mean() * 100
    return missing_data[missing_data > 0].sort_values(ascending=False)

# Identify missing data
missing_data = check_missing_data(data)
print("\n📌 **Missing Data Percentage per Column:**\n", missing_data)

# Plot missing data
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Data Visualization")
plt.show()

# Handling missing data (example: forward fill, then backward fill as fallback)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# Extract nighttime data (assuming no solar power at night)
night_data = data[data['InvPAC_kW_Avg'] < 0]  # Assuming negative power at night
print(f"\n🌙 **Nighttime data extracted. Entries:** {len(night_data)}")
print(f"\n🌙 **Average night power ('InvPAC_kW_Avg'):** {night_data['InvPAC_kW_Avg'].mean():.2f} kW")

# Time gap analysis
data['time_diff'] = data['TIMESTAMP'].diff().dt.total_seconds() / 60  # Convert to minutes
gaps = data[['TIMESTAMP', 'time_diff']].dropna()

# Find largest gap
largest_gap = gaps.sort_values(by='time_diff', ascending=False).iloc[0]
print(f"\n🔍 **Largest gap detected:** {largest_gap['time_diff']:.2f} minutes")
print(f"\n⏳ **Occurred at:**\n", largest_gap)

# Find gaps greater than 30 minutes
large_gaps = gaps[gaps['time_diff'] > 30]
print(f"\n📊 **Number of large gaps (>30 min):** {len(large_gaps)}")

if len(large_gaps) == 0:
    print("\n✅ No significant gaps detected!")
else:
    print("\n⚠️ **Significant gaps found. Consider investigating!**")

# Final dataset check
print("\n✅ **Data processing complete!** Ready for further analysis.")
