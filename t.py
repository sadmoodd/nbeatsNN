import pandas as pd

df = pd.read_csv('evaluation_results.csv')

maxr2 = df["R²"].max()

