import sys
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python concat_events.py <event_file1> <event_file2> <output_file>')
        sys.exit(0)

    e1 = pd.read_csv(sys.argv[1], index_col=0)
    e2 = pd.read_csv(sys.argv[2], index_col=0)
    if ('glove' or 'semantic' in sys.argv[1]) and ('glove' or 'semantic' in sys.argv[2]):
        if 'audio' in sys.argv[1]:
            e1['trial_type'] = 'audio_' + e1['trial_type']
        else:
            e1['trial_type'] = 'visual_' + e1['trial_type']
        if 'audio' in sys.argv[2]:
            e2['trial_type'] = 'audio_' + e2['trial_type']
        else:
            e2['trial_type'] = 'visual_' + e2['trial_type']
    combined = pd.concat([e1, e2])
    combined.to_csv(sys.argv[3])
