import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# print(df.head())
# print(df.info())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

inertia=[]
k_range=range(1,11)

for k in k_range:
    kmeans = KMeans(n_clusters=k ,random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8,5))
plt.plot(k_range,inertia,marker='o')    
plt.title("Elbow - Optimal K")
plt.xlabel("Number of Cluster(k)")
plt.ylabel("inertia(within-cluster sum of squared)")
plt.grid(True)
plt.show()


kmeans = KMeans(n_clusters=3 ,random_state=42)
cluster=kmeans.fit_predict(scaled_data)

df['cluster']=cluster

pca = PCA(n_components=2)

pca_result =pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(pca_result[:,0],pca_result[:,1],c=cluster , cmap='viridis',alpha=0.7)
plt.title("K Means Cluster visualize with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal component 2")
plt.colorbar(label='cluster')
plt.grid(True)
plt.show()

important_columns = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
    'T', 'RH', 'AH'
]

print("=== K = 3 Cluster Averages ===")
print(df.groupby('cluster')[important_columns].mean().round(2))




# Use a subset for visibility
# subset = df[['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'cluster']]
# sns.pairplot(subset, hue='cluster', palette='tab10')
# plt.suptitle("Pairplot of Clusters", y=1.02)
# plt.show()




