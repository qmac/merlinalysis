import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

n = 50

events = pd.read_csv('events/raw_visual_events.csv', index_col=0)
objects = events.drop(['onset', 'duration', 'order', 'object_id'], axis=1)

dim_red = PCA()
reduced = dim_red.fit_transform(objects)

top_n = reduced[:, :n]
cols = ['PCA_dimension_%d' % i for i in range(n)]
new_df = pd.DataFrame(top_n, columns=cols)
new_df['onset'] = events['onset']
new_df['duration'] = events['duration']
new_df = pd.melt(new_df, id_vars=['onset', 'duration'], var_name='trial_type')
new_df.rename(columns={'value': 'modulation'}, inplace=True)
new_df.to_csv('events/visual_dimred_object_events.csv')

plt.plot(np.cumsum(dim_red.explained_variance_ratio_[:n]))
plt.show()
