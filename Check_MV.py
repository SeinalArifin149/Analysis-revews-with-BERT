import pandas as pd

# baca file CSV
# df = pd.read_csv("Wisata_Syaikhona.csv")
# df = pd.read_csv("Taman Mangrove sepulu.csv")
df = pd.read_csv("Bukit Jaddih(Cleaning).csv")
print("=== CEK MISSING VALUE ===")
print(df.isnull().sum())

print("\n=== TOTAL MISSING VALUE ===")
print(df.isnull().sum().sum())

print("\n=== CEK DATA DUPLIKAT ===")
duplicates = df.duplicated()
print("Jumlah duplikat:", duplicates.sum())

print("\n=== DATA DUPLIKAT ===")
print(df[df.duplicated()])

# cek dimensi (rows, columns)
print("Dimensi data:", df.shape)

# pisahin biar jelas
print("Jumlah baris:", df.shape[0])
print("Jumlah kolom:", df.shape[1])

print("\n=== SAMPLE 6 DATA (HEAD) ===")
print(df.head(6))