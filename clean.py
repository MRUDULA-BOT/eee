import pandas as pd
import numpy as np

# Load dataset
file_path = r"C:\Users\Mrudula\OneDrive\Desktop\EEE\merged_solar_data.csv"
df = pd.read_csv(file_path)
print("\n✅ Data Loaded Successfully!")

# 1️⃣ **Check for Duplicate Columns**
print("\n🔍 Checking for duplicate column names...")
duplicate_columns = df.columns[df.columns.duplicated()]
if len(duplicate_columns) > 0:
    print(f"⚠️ Duplicate column names found: {list(duplicate_columns)}")
    df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
    print("✅ Removed duplicate columns!")
else:
    print("✅ No duplicate column names found!")

# 2️⃣ **Check for Duplicate Rows**
print("\n🔍 Checking for duplicate rows...")
duplicate_rows = df.duplicated().sum()
print(f"⚠️ Found {duplicate_rows} exact duplicate rows!")
if duplicate_rows > 0:
    df.drop_duplicates(inplace=True)
    print("✅ Removed duplicate rows!")

# 3️⃣ **Handle Missing Values**
print("\n🔍 Checking for missing values...")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0])

# Fill missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"⚠️ {col} has {df[col].isnull().sum()} missing values")
        if df[col].dtype in ['int64', 'float64']:
            fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            print(f"   ➡️ Filled {col} with median: {fill_value}")
        else:
            fill_value = df[col].mode()[0]
            df[col].fillna(fill_value, inplace=True)
            print(f"   ➡️ Filled {col} with mode: {fill_value}")
        df[f"{col}_missing"] = df[col].isnull().astype(int)

# 4️⃣ **Fix Data Types**
print("\n🔍 Checking Data Types Before Conversion:")
print(df.dtypes)

if 'timestamp' in df.columns:
    print("\n⏳ Converting 'timestamp' to datetime format...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if df['timestamp'].isnull().all():
        print("❌ Timestamp conversion failed! All values became NaT.")
    else:
        print("✅ Successfully converted 'timestamp' to datetime!")
        
        # Extract datetime features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['weekday'] = df['timestamp'].dt.weekday
        print("✅ Extracted date-time features!")

        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return "Morning"
            elif 12 <= hour < 17:
                return "Afternoon"
            elif 17 <= hour < 21:
                return "Evening"
            else:
                return "Night"
        
        df['time_of_day'] = df['hour'].apply(get_time_of_day)
        print("✅ Created 'time_of_day' column!")
else:
    print("❌ 'timestamp' column not found!")

# 5️⃣ **Convert Object Columns to Numeric Where Possible**
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col])
            print(f"✅ Converted {col} to numeric!")
        except:
            print(f"⚠️ Could not convert {col} to numeric!")

print("\n🔍 Data Types After Conversion:")
print(df.dtypes)

# 6️⃣ **Standardize Column Names**
df.columns = [col.lower().replace(" ", "_") for col in df.columns]
print("✅ Standardized Column Names:")
print(df.columns)

# 7️⃣ **Handle Duplicates Again (After Cleaning)**
df['is_duplicate'] = df.duplicated().astype(int)
print(f"\n🔍 Found {df['is_duplicate'].sum()} duplicate rows (Flagged, not removed)")

# 8️⃣ **Final Check**
print("\n🔍 Final DataFrame Overview:")
print(df.head())
print("\n✅ Preprocessing Completed!")

# Optional: Save the cleaned dataset
cleaned_path = r"C:\Users\Mrudula\OneDrive\Desktop\EEE\cleaned_solar_data.csv"
df.to_csv(cleaned_path, index=False)
print(f"\n💾 Cleaned dataset saved to {cleaned_path}!")
