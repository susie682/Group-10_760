import pandas as pd

# Load the CSV file
df = pd.read_csv("keogram_segment_stats2019.csv")

# Assume column C is the 'median' column
# Drop rows where median < 15
filtered_df = df[df['median'] >= 15]

# Save the cleaned CSV
filtered_df.to_csv("filtered_output.csv", index=False)

print("Rows with median < 15 removed. Saved as filtered_output.csv")
