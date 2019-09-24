import numpy as np
import pandas as pd

df = pd.read_csv("../data/cleaned_tweets_orig.csv", encoding='ISO-8859-1')

print(len(df.loc[df['class'] == 0]))
print(len(df.loc[df['class'] == 1]))
print(len(df.loc[df['class'] == 2]))

df_0 = df.loc[df['class'] == 0]
df_2 = df.loc[df['class'] == 2]

for i in range(6):
    df = df.append(df_0, ignore_index=True)

for i in range(2):
    df = df.append(df_2, ignore_index=True)

print(len(df.loc[df['class'] == 0]))
print(len(df.loc[df['class'] == 1]))
print(len(df.loc[df['class'] == 2]))
print(df.tail())

export_csv = df.to_csv ("../data/cleaned_tweets_oversampled.csv", index = None, header=True) #Don't forget to add '.csv' at the end of the path
