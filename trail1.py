import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("\n🔍 Attempting to load data...")
df = pd.read_csv(r"C:\Users\Mrudula\OneDrive\Desktop\EEE\merged_solar_data.csv")  # Replace with actual file name
print("✅ File loaded successfully!\n")
print("📌 First 5 rows:")
print(df.head(), "\n")
print("📌 Dataset contains", df.shape[0], "rows and", df.shape[1], "columns.\n")

# Creating output directory
output_dir = "Analysis_Results"
os.makedirs(output_dir, exist_ok=True)
print(f"📂 Output directory created: {output_dir}\n")

# Checking missing values
missing_values = df.isnull().sum()
print("🔍 Checking for missing values...\n")
print(missing_values[missing_values > 0])

# Summary statistics
print("\n📊 Generating summary statistics...\n")
descriptive_stats = df.describe()
print(descriptive_stats, "\n")
summary_file = os.path.join(output_dir, "summary_statistics.csv")
descriptive_stats.to_csv(summary_file)
print(f"📄 Summary statistics saved to {summary_file}\n")

# Time column analysis (assuming 'TIMESTAMP' is datetime)
if 'TIMESTAMP' in df.columns:
    print("🔍 Converting TIMESTAMP column to datetime format...")
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = df.sort_values(by='TIMESTAMP')
    print("✅ TIMESTAMP conversion successful!\n")
    
    print("📊 Analyzing time gaps...")
    df['time_diff'] = df['TIMESTAMP'].diff().dt.total_seconds() / 60  # in minutes
    largest_gap = df['time_diff'].max()
    large_gaps = df[df['time_diff'] > 30]
    print(f"⏳ Largest gap detected: {largest_gap} minutes\n")
    print(f"📌 Number of gaps >30 min: {len(large_gaps)}\n")
else:
    print("⚠️ TIMESTAMP column missing. Skipping time gap analysis.\n")

# Visualization: Missing Data Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
missing_data_plot = os.path.join(output_dir, "missing_data_heatmap.png")
plt.savefig(missing_data_plot)
print(f"📸 Missing data heatmap saved: {missing_data_plot}\n")
plt.close()

# Histogram for numerical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    print(f"📊 Plotting histogram for {col}...")
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    hist_file = os.path.join(output_dir, f"histogram_{col}.png")
    plt.savefig(hist_file)
    print(f"📸 Histogram saved: {hist_file}\n")
    plt.close()

print("✅ Analysis completed! All results saved in", output_dir)
