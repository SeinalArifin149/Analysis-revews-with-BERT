import pandas as pd

slang_df = pd.read_csv("slang.csv")
slang_dict =dict(
    zip(
        slang_df["slang"].str.lower().str.strip(),
        slang_df["formal"].str.lower().str.strip()
        ))