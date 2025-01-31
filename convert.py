#write code to convert xlsx to csv
import pandas as pd

df = pd.read_excel('globalterrorismdb_0522dist.xlsx')
df.to_csv('globalterrorismdb_0522dist.csv', index=False)

df = pd.read_excel('globalterrorismdb_2021Jan-June_1222dist.xlsx')
df.to_csv('globalterrorismdb_2021Jan-June_1222dist.csv', index=False)