''' Concatenates two event files. '''
import sys
import pandas as pd


def concat_events(event_file1, event_file2):
    e1 = pd.read_csv(event_file1, index_col=0)
    e2 = pd.read_csv(event_file2, index_col=0)

    # This logic avoids naming conflicts between audio/visual semantic events
    if ('glove' or 'semantic' in event_file1) and \
       ('glove' or 'semantic' in event_file2):
        if 'audio' in event_file1:
            e1['trial_type'] = 'audio_' + e1['trial_type']
        else:
            e1['trial_type'] = 'visual_' + e1['trial_type']
        if 'audio' in event_file2:
            e2['trial_type'] = 'audio_' + e2['trial_type']
        else:
            e2['trial_type'] = 'visual_' + e2['trial_type']

    # Concat and return
    combined = pd.concat([e1, e2])
    return combined


if __name__ == '__main__':
    combined_df = concat_events(sys.argv[1], sys.argv[2])
    combined_df.to_csv(sys.argv[3])
