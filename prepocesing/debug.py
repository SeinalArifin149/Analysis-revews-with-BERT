import pandas as pd

df= pd.read_csv("Bukit jaddih.csv")

print(df["stars"].unique())
print(df["stars"].isna().sum())