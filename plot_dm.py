import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nistats.design_matrix import make_design_matrix

TR = 1.5


def get_design_matrix(event_file):
    events = pd.read_csv(event_file, index_col=0)
    events = events[events['onset'] >= 40.5]
    events['onset'] -= 40.5
    start_time = 0.0
    end_time = (983 - 1) * TR
    frame_times = np.linspace(start_time, end_time, 983)
    fir_delays = [0]
    dm = make_design_matrix(frame_times, events, hrf_model='fir',
                            fir_delays=fir_delays, drift_model=None)
    dm = dm.drop('constant', axis=1)
    return dm


# f, axarr = plt.subplots(3, sharex=True)
# axarr[0].plot(get_design_matrix('events/audio_energy_events.csv'))
# axarr[0].set_title('energy')
# axarr[1].plot(get_design_matrix('events/speech_events.csv'), 'red')
# axarr[1].set_title('speech')
# axarr[2].plot(get_design_matrix('events/audio_semantic_events.csv'))
# axarr[2].set_title('semantics')
# plt.show()


f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(get_design_matrix('events/visual_object_events.csv')['human_delay_0'])
axarr[0].set_title('human_object')
axarr[1].plot(get_design_matrix('events/visual_semantic_events.csv'))
axarr[1].set_title('semantics')
axarr[2].plot(get_design_matrix('events/face_events.csv'), 'red')
axarr[2].set_title('face')
axarr[3].plot(get_design_matrix('events/visual_brightness_events.csv'), 'orange')
axarr[3].set_title('brightness')
plt.show()
