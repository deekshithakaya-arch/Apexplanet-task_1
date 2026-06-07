# =====================================================
# TASK 1 : DATA IMMERSION & WRANGLING
# =====================================================

import pandas as pd
import numpy as np

print("=" * 60)
print("TASK 1 : DATA IMMERSION & WRANGLING")
print("=" * 60)

# =====================================================
# 1. LOAD DATA
# =====================================================

print("\n[1] Loading Dataset...")

df = pd.read_csv("Superstore data.csv", encoding="utf-8")

print("Dataset Loaded Successfully!")

# =====================================================
# 2. DATA PROFILING
# =====================================================

print("\n[2] DATA PROFILING")

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe(include='all'))

# =====================================================
# 3. CHECK MISSING VALUES
# =====================================================

print("\n[3] MISSING VALUES CHECK")

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print("\nTotal Missing Values:", total_missing)

# =====================================================
# 4. CHECK DUPLICATES
# =====================================================

print("\n[4] DUPLICATE CHECK")

duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates Removed!")
else:
    print("No Duplicates Found.")

# =====================================================
# 5. STANDARDIZE COLUMN NAMES
# =====================================================

print("\n[5] STANDARDIZING COLUMN NAMES")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print(df.columns.tolist())

# =====================================================
# 6. DATA TYPE VALIDATION
# =====================================================

print("\n[6] DATA TYPES")

print(df.dtypes)

# =====================================================
# 7. OUTLIER DETECTION
# =====================================================

print("\n[7] OUTLIER DETECTION")

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"{col} : {len(outliers)} outliers")

# =====================================================
# 8. FEATURE ENGINEERING
# =====================================================

print("\n[8] FEATURE ENGINEERING")

# Profit Margin (%)

df["profit_margin"] = np.where(
    df["sales"] != 0,
    (df["profit"] / df["sales"]) * 100,
    0
)

# Discount Category

df["discount_category"] = pd.cut(
    df["discount"],
    bins=[-0.01, 0, 0.20, 1],
    labels=["No Discount", "Low Discount", "High Discount"]
)

print("New Features Created:")
print("1. profit_margin")
print("2. discount_category")

# =====================================================
# 9. DATA DICTIONARY
# =====================================================

print("\n[9] CREATING DATA DICTIONARY")

data_dictionary = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

data_dictionary.to_csv(
    "Data_Dictionary.csv",
    index=False
)

print("Data Dictionary Saved!")

# =====================================================
# 10. SAMPLE INSIGHTS
# =====================================================

print("\n[10] SAMPLE INSIGHTS")

print("\nTotal Sales:")
print(round(df["sales"].sum(), 2))

print("\nTotal Profit:")
print(round(df["profit"].sum(), 2))

print("\nTop Category By Sales:")
print(
    df.groupby("category")["sales"]
      .sum()
      .sort_values(ascending=False)
      .head(1)
)

print("\nTop Region By Profit:")
print(
    df.groupby("region")["profit"]
      .sum()
      .sort_values(ascending=False)
      .head(1)
)

# =====================================================
# 11. SAVE CLEANED DATASET
# =====================================================

print("\n[11] SAVING CLEANED DATASET")

df.to_csv(
    "Cleaned_Superstore.csv",
    index=False
)

print("Cleaned Dataset Saved Successfully!")

# =====================================================
# COMPLETED
# =====================================================

print("\n" + "=" * 60)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 60)