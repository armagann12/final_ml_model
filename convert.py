import pandas as pd

# 1. Load the .pkl file
df = pd.read_pickle('ensemble_results.pkl')

# 2. Save as .xlsx file
# index=False prevents writing row numbers
df.to_excel('output_file4.xlsx', index=False)
