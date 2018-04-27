import sys
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

def add_tree_to_graph(G, tree):
    if len(tree) == 1:
        G.add_node(tree[0].name())
        return tree[0].name()
    else:
        name = tree[0].name()
        for branch in tree[1:]:
            branch_name = add_tree_to_graph(G, branch)
            G.add_edge(branch_name, name)
        return name

def make_object_hierarchy():
    with open('events/objects.txt', 'r') as f:
        objects = f.read().split(',')

    G = nx.Graph()
    for obj in objects:
        w = w.split()[-1] # take last word if more than one
        w = w + '.n.1'
        try:
            synset = wn.synset(w)
        except:
            continue
        tree = synset.tree(lambda s:s.hypernyms()) 
        name = add_tree_to_graph(G, tree)
        assert name == synset.name()

    nx.draw_graphviz(G)
    nx.write_dot(G,'events/object_semantic_graph.dot')

def get_objects_level(level):
    G = nx.read_dot('events/object_semantic_graph.dot')
    visited = set(['entity.n.01'])
    curr = ['entity.n.01']
    for i in range(level):
        new = []
        for n in curr:
            neighbors = G.neighbors(n)
            new += [child for child in neighbors if child not in visited]
            visited.update(neighbors)
        curr = new
    return curr

def convert_object_events(event_file, output_file, level):
    events = pd.read_csv(event_file, index_col=0)
    target_objects = get_objects_level(level)
    categories = {}
    for i, obj in enumerate(events['trial_type'].unique()):
        w = obj.split()[-1] # take last word if more than one
        w = w + '.n.1'
        try:
            synset = wn.synset(w)
        except:
            categories[obj] = np.nan
            continue
        tree = synset.tree(lambda s:s.hypernyms())
        target_categories = ''
        for target in target_objects:
            if target in str(tree):
                target_categories += ' ' + target if target_categories else target
        categories[obj] = target_categories

    events = events.replace({'trial_type': categories})
    events = events.dropna()
    events['trial_type'] = events['trial_type'].str.split()
    lst_col = 'trial_type'
    events = pd.DataFrame({
        col:np.repeat(events[col].values, events[lst_col].str.len())
        for col in events.columns.difference([lst_col])
    }).assign(**{lst_col:np.concatenate(events[lst_col].values)})[events.columns.tolist()]
    events = events.groupby(['onset', 'trial_type', 'duration']).max().reset_index()
    events.to_csv(output_file)

if __name__ == '__main__':
    convert_object_events(sys.argv[1], sys.argv[2], int(sys.argv[3]))
