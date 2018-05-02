import pandas as pd
import numpy as np
import sys

def downsample_events(df, chunk_size=10):
    values = []
    for c in range(27, 1002, chunk_size):
        start = c * 1.5
        end = (c + chunk_size - 1) * 1.5
        subframe = df.loc[start:end]
        totals = subframe.sum()
        values.extend([totals.argmax()] * min(chunk_size, 1002 - c))
    return values

def downsample_single_events(df, chunk_size=10):
    values = []
    for c in range(27, 1002, chunk_size):
        start = c * 1.5
        end = (c + chunk_size - 1) * 1.5
        subframe = df.loc[start:end]
        totals = subframe.sum()[0]
        val = 1.0 if totals >= 5 else 0.0
        values.extend([val] * min(chunk_size, 1002 - c))
    return values

if __name__ == '__main__':
    event_file = sys.argv[1]
    output_file = sys.argv[2]
    df = pd.read_csv(event_file, index_col=0)
    df = df[df['onset'] >= 40.5]
    df = df.sort_values('onset').reset_index(drop=True)
    df = df.drop_duplicates()
    pivoted = df.pivot(index='onset', columns='trial_type', values='modulation')
    pivoted = pivoted.fillna(0.0)
    # new_df = pd.DataFrame(downsample_single_events(pivoted), columns=['trial_type'])
    new_df = pd.DataFrame(downsample_events(pivoted), columns=['trial_type'])
    onsets = np.array(range(27, 1002)) * 1.5
    new_df['onset'] = onsets
    new_df['duration'] = 1.5
    new_df.to_csv(output_file)
