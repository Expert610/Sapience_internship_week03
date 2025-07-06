import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('clustered_data.csv')
# print(df.head())
# print(df.info())
features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
            'T', 'RH', 'AH']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# apply pca

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(scaled_data)

df['PC1'] = pca_fit[:,0]
df['PC2'] = pca_fit[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Hcluster', palette='Set1', s=60)
plt.title('PCA: Air Quality Data (2D Projection)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print(f"Total Variance Captured by 2 Components: {pca.explained_variance_ratio_.sum():.2f}")


