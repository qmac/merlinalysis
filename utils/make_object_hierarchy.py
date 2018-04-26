import networkx as nx
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


with open('../events/objects.txt', 'r') as f:
    objects = f.read().split(',')

G = nx.Graph()
for obj in objects:
    w = obj + '.n.1'
    try:
        synset = wn.synset(w)
    except:
        continue
    tree = synset.tree(lambda s:s.hypernyms()) 
    name = add_tree_to_graph(G, tree)
    assert name == synset.name()

nx.draw_graphviz(G)
nx.write_dot(G,'../events/object_semantic_graph.dot')
