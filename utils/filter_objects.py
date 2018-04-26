import sys
import pandas as pd

def filter_objects(allowed_file, output_file):
    with open(allowed_file, 'r') as f:
        allowed_objects = f.read().split(',')
    
    object_event_file = 'events/visual_object_events.csv'
    events = pd.read_csv(object_event_file, index_col=0)
    events = events[events['trial_type'].isin(allowed_objects)]
    events.to_csv(output_file)

if __name__ == '__main__':
    allowed_file = sys.argv[1]
    output_file = sys.argv[2]
    filter_objects(allowed_file, output_file)