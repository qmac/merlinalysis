''' Filters objects in the provided events file according to a
comma separated list in an allowed objects file. '''
import sys
import pandas as pd


def filter_objects(events, allowed_objs):
    filtered = events[events['trial_type'].isin(allowed_objects)]
    return filtered


if __name__ == '__main__':
    events_file = sys.argv[1]
    allowed_file = sys.argv[2]
    with open(allowed_file, 'r') as f:
        allowed_objects = f.read().split(',')

    events = pd.read_csv(events_file, index_col=0)
    events = filter_objects(events, allowed_objects)
    events.to_csv(sys.argv[3])
