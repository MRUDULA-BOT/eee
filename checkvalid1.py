import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\Mrudula\OneDrive\Desktop\EEE\merged_solar_data.csv"  # Change this to your actual file path
print("\n📌 Attempting to load data...\n")

try:
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    print("\n✅ File loaded successfully! First 5 rows:\n")
    print(df.head(), "\n")
except Exception as e:
    print(f"\n❌ Error loading file: {e}\n")
    exit()

# Print available columns
print("\n📌 Available columns in dataset:\n", df.columns.tolist(), "\n")

# Ensure 'timestamp' column exists
if 'timestamp' not in df.columns:
    print("\n❌ ERROR: 'timestamp' column missing. Check the dataset format.\n")
    exit()

# Ensure timestamp is sorted
df = df.sort_values(by="timestamp")

### **1. CHECK TIME GAPS** ###
df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60  # Convert to minutes
print("\n📊 Time gap analysis completed!\n")

# Identify largest gap
largest_gap = df['time_diff'].max()
largest_gap_row = df[df['time_diff'] == largest_gap]

print(f"\n🔍 **Largest gap detected:** {largest_gap:.2f} minutes\n")
print("⏳ **Occurred at:**\n", largest_gap_row[['timestamp', 'time_diff']], "\n")

# Identify all large gaps (>30 minutes)
large_gaps = df[df['time_diff'] > 30]
print(f"\n📊 **Number of large gaps (>30 min):** {len(large_gaps)}\n")

if not large_gaps.empty:
    print("\n📌 **Largest time gaps found:**\n")
    print(large_gaps[['timestamp', 'time_diff']].tail(10), "\n")  # Show last 10 large gaps
else:
    print("\n✅ No significant gaps detected!\n")

# Plot time gap distribution
plt.figure(figsize=(10, 4))
plt.hist(df['time_diff'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Time gap (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Time Gaps in Data")
plt.show()

### **2. CHECK INVERTER READINGS AT NIGHT** ###
# Identify correct column name for inverter readings
possible_power_cols =  ['InvPAC_kW_Avg', 'InvPDC_kW_Avg', 'InvEtot_kWh_Max']
if possible_power_cols:
    power_col = possible_power_cols[0]  # Use the first matching column
    print(f"\n✅ **Using '{power_col}' as inverter power column.**\n")
else:
    print("\n⚠️ WARNING: No inverter power column found. Skipping power analysis.\n")
    power_col = None

# Extract night-time data (6 PM - 6 AM)
df['hour'] = df['timestamp'].dt.hour
night_data = df[(df['hour'] >= 18) | (df['hour'] <= 6)]
print(f"\n🌙 **Nighttime data extracted. Entries:** {len(night_data)}\n")

if power_col:
    try:
        avg_night_power = night_data[power_col].mean()
        print(f"\n🌙 **Average night power ('{power_col}'):** {avg_night_power:.2f} kW\n")

        # Check if night power is abnormally high
        if avg_night_power > 0.1:
            print("⚠️ **WARNING: High inverter power detected at night. Possible sensor noise!**\n")
    except Exception as e:
        print(f"\n❌ ERROR: Unable to calculate night power: {e}\n")

### **3. CHECK NIGHT RADIATION** ###
# Identify correct column for solar radiation
possible_radiation_cols = [col for col in df.columns if "radiation" in col.lower()]
if possible_radiation_cols:
    radiation_col = possible_radiation_cols[0]  # Use the first match
    print(f"\n✅ **Using '{radiation_col}' as solar radiation column.**\n")
else:
    print("\n⚠️ WARNING: No solar radiation column found. Skipping radiation analysis.\n")
    radiation_col = None

if radiation_col:
    try:
        avg_night_radiation = night_data[radiation_col].mean()
        print(f"\n🌙 **Average night solar radiation ('{radiation_col}'):** {avg_night_radiation:.2f} W/m²\n")

        # Flag ambient light or sensor noise
        if avg_night_radiation > 0.1:
            print("⚠️ **WARNING: Nonzero night radiation detected. Check for ambient light or sensor noise!**\n")

        # Plot night radiation distribution
        plt.figure(figsize=(10, 4))
        plt.hist(night_data[radiation_col].dropna(), bins=50, color='orange', edgecolor='black')
        plt.xlabel("Night Solar Radiation (W/m²)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Night Solar Radiation")
        plt.show()
    except Exception as e:
        print(f"\n❌ ERROR: Unable to calculate night radiation: {e}\n")

### **4. CHECK MISSING DATA** ###
missing_perc = df.isnull().mean() * 100
print("\n📌 **Missing Data Percentage per Column:**\n", missing_perc[missing_perc > 0], "\n")

if missing_perc.sum() == 0:
    print("\n✅ **No missing data detected!**\n")
else:
    print("\n⚠️ **Missing data found. Visualizing...\n")

    # Visualize missing data
    plt.figure(figsize=(12, 5))
    plt.bar(missing_perc.index, missing_perc.values, color='red')
    plt.xticks(rotation=90)
    plt.ylabel("Missing Data Percentage")
    plt.title("Missing Data in Each Column")
    plt.show()
