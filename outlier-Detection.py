import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('AirQualityUCI.csv' ,sep=';')



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

kmean = KMeans(n_clusters=3 , random_state=42)
clusters = kmean.fit_predict(scaled_data)

centroids = kmean.cluster_centers_
distance = np.linalg.norm(scaled_data - centroids[clusters] , axis=1)

df['Clusters'] = clusters
df['distance_from_center'] = distance

threshold = np.percentile(distance,95)
df['Anomaly'] = df['distance_from_center'] > threshold

anomalies=df[df['Anomaly']== True]
print(f"the number of Anomalies are :{len(anomalies)}")



pca = PCA(n_components=2 ,random_state=42)
pca_result = pca.fit_transform(scaled_data)

df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

plt.figure(figsize=(10,6))
sns.scatterplot(data=df ,x='PCA1', y='PCA2',hue='Clusters' ,palette= 'Set2',alpha=0.6)
plt.scatter(df[df['Anomaly']== True]['PCA1'],
            df[df['Anomaly']== True]['PCA2'],
            color='red', edgecolors='black', s=100 ,label='Anomalies')
plt.title("Anomaly Detection using PCA + KMeans")
plt.legend()
plt.show()