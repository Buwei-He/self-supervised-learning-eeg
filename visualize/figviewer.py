# For interactive view of 3d plots in your own computers (with GUI supports)

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

data = np.load('Skoda_tsne_3d.npz')['arr_0'].T
fig = plt.figure(figsize=(8, 8)) 
sns.set_style("whitegrid", {'axes.grid' : False})
ax = plt.axes(projection="3d")
sc = ax.scatter(data[0], data[1], data[2], c=data[3], marker='o')
plt.legend(*sc.legend_elements())
plt.show()