import numpy as np
import matplotlib.pyplot as pl
import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



#%%
data = pandas.read_csv('data.csv')
pca = PCA(n_components=2)
x = data.iloc[:,1:].values
x = StandardScaler().fit_transform(x)
pcomps = pca.fit_transform(x)
pl.figure(figsize=(4,4))
for n in data.iterrows():
    pl.plot(pcomps[n[0],1],pcomps[n[0],0],'o')
    pl.text(pcomps[n[0],1],pcomps[n[0],0],n[1].iloc[0])
pl.xlabel('PC2')
pl.ylabel('PC1')