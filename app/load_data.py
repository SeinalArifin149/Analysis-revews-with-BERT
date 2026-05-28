import pandas as pd
from pathlib import Path

def load_data():

    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / "Top_Pariwisata-Bangkalan.csv"

    df = pd.read_csv(path)

    return df