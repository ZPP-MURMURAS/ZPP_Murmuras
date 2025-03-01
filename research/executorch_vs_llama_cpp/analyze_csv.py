import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python analyze_csv.py <csv_filename>")
    sys.exit(1)

csv_filename = sys.argv[1]

try:
    df = pd.read_csv(csv_filename)

    means = df.mean()
    std_devs = df.std()

    for col in df.columns:
        print(f"{col}: Mean = {means[col]:.6f}, Std Dev = {std_devs[col]:.6f}")

except Exception as e:
    print(f"Error: {e}")

