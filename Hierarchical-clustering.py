import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster

df = pd.read_csv('AirQualityUCI.csv' ,sep=';')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.notnull().sum())


# Data Preprocessing

df.drop(columns=['Date','Time','Unnamed: 15','Unnamed: 16'] , inplace=True)



for col in ['CO(GT)','C6H6(GT)','T','RH','AH']:
    df[col]=df[col].str.replace(',','.').astype(float)



df.replace(-200 , np.nan ,inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

df = df.apply(pd.to_numeric ,errors='coerce')
df.dropna(inplace=True)
# print(df.head())
# print(df.info())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

linked=linkage(scaled_data,method='ward')

plt.figure(figsize=(14,8))
dendrogram(linked, truncate_mode='lastp',p=30,leaf_font_size=10,leaf_rotation=90,show_contracted=True)
plt.title("Hierarhical clustering Dendogram")
plt.xlabel("Sample index or cluster")
plt.ylabel("Euclidean distance ")
plt.tight_layout()
plt.show()

cluster_label = fcluster(linked,3,criterion='maxclust')

df['Hcluster'] = cluster_label

c_summary = df.groupby('Hcluster').mean().round(2)




print("===Cluster Averages ===")
print(c_summary)
