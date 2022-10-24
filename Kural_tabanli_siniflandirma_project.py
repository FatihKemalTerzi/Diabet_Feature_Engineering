import numpy as np
import pandas as pd
import seaborn as sns
#soru   1
df = pd.read_csv("persona.csv")
df.head()

df.describe()

#soru 2
df["SOURCE"].nunique()
#soru 3
df["PRICE"].nunique()
#Soru 4
df["PRICE"].value_counts()
#soru 5
df["COUNTRY"].value_counts()
#soru 6
df.groupby("COUNTRY")["PRICE"].sum()
#soru7
df.groupby("SOURCE")["PRICE"].sum()
#Soru 8
df.groupby("COUNTRY")["PRICE"].mean()
#Soru 9
df.groupby("SOURCE")["PRICE"].mean()
#Soru 10
df.groupby(["SOURCE","COUNTRY"])["PRICE"].mean()
#GÖREV 2
df.groupby(["SOURCE","COUNTRY","SEX","AGE"])["PRICE"].mean()
#GÖREV 3 df.sort_values(by=['col1']) df.sort_values(['job','count'],ascending=False).groupby('job').head(3)
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
#agg_df = df.sort_values(["PRICE"],ascending=False).groupby(["SOURCE", "COUNTRY", "SEX", "AGE"])
agg_df.head(10)

#görev 4
agg_df = agg_df.reset_index()
agg_df.head()

#görev 5
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]


agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

#görev 6
agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()