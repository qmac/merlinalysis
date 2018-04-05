import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nistats.design_matrix import make_design_matrix
from ridge.utils import zscore

TR = 1.5


def get_design_matrix(event_file, n_scans):
    events = pd.read_csv(event_file, index_col=0)
    events = events[events['onset'] >= 40.5]
    events['onset'] -= 40.5
    print('Raw events shape: ' + str(events.shape))
    start_time = 0.0
    end_time = (n_scans - 1) * TR
    frame_times = np.linspace(start_time, end_time, n_scans)
    fir_delays = [0]
    dm = make_design_matrix(frame_times, events, hrf_model='fir',
                            fir_delays=fir_delays, drift_model=None)
    dm = dm.drop('constant', axis=1)
    return dm


event_files = [# 'events/audio_semantic_events.csv',
               # 'events/audio_speech_events.csv',
               # 'events/audio_energy_events.csv',
               'events/visual_object_events.csv',
               'events/visual_semantic_events.csv',
               # 'events/visual_face_events.csv',
               ]

all_matrices = []
for ev in event_files:
    dm = get_design_matrix(ev, 975)
    all_matrices.append(zscore(dm.as_matrix().T).T)
    print('Done: ' + ev)

big_matrix = np.concatenate(all_matrices, 1)
print(big_matrix.shape)
corrs = np.corrcoef(big_matrix.T)
plt.matshow(corrs)
plt.show()
