from func_prepocesing import preprocess
import pandas as pd

df = pd.read_csv("Bukit jaddih")
df ["clean_text"] = df["reviews"].astype(str).apply(preprocess)
df.to_csv("hasil prepocesing.csv", index=False)

print("Done Prepocesingnya bang")

