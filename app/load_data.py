import pandas as pd
def load_data():
    path = "Top_Pariwisata-Bangkalan.csv"
    df = pd. read_csv(path)
    return df