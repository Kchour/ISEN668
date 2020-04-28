#!/usr/bin/env python3

import pandas as pd
import pdb
from utils import *
from search_algorithms import *
import networkx as nx


#### LOAD MISSION SET
path = "./mission_set.ods"
dfMission =pd.read_excel(path, sheet_name=0, engine='odf', index_col='Mission')

#### LOAD CMC SET
path = "./cmc.ods"
dfCMC = pd.read_excel(path, sheet_name=0, engine='odf')

#### LOAD SHIP SET
path = "./ships.ods"
dfShips = pd.read_excel(path, engine='odf', index_col="Ship")

### LOAD REGION SET
path = "./region.ods"
dfRegion = pd.read_excel(path, engine='odf', sheet_name=1)

# Build list and  dictionary of edges. Should be undirected!
# lists are not hashable types, thus cannot be dictionary keys
edgeList = [(dfRegion.loc[i]['Arcs'], dfRegion.loc[i]['Unnamed: 1'], dfRegion.loc[i]['Length(nm)']) for i in range(len(dfRegion))]

edgeDict = {(dfRegion.loc[i]['Arcs'], dfRegion.loc[i]['Unnamed: 1']): dfRegion.loc[i]['Length(nm)'] for i in range(len(dfRegion))  }
edgeDictMirror = {(dfRegion.loc[i]['Unnamed: 1'], dfRegion.loc[i]['Arcs']): dfRegion.loc[i]['Length(nm)'] for i in range(len(dfRegion))  }
edgeDict.update(edgeDictMirror)


### Plotting graph for DEBUG PURPOSES
#G = nx.Graph()
#G.add_weighted_edges_from(edgeList)
#pos = nx.spring_layout(G)
#
## nodes
#nx.draw_networkx_nodes(G, pos, node_size=700)
#
## edges
#nx.draw_networkx_edges(G, pos,
#                       width=6)
## labels
#new_labels = dict(map(lambda x:((x[0],x[1]), str(x[2]['weight'] if x[2]['weight']>0 else "") ), G.edges(data = True)))
#nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
#nx.draw_networkx_edge_labels(G, pos, font_size=10, font_family='sans-serif', edge_labels=new_labels)
#
#plt.axis('off')
#plt.show()


#### GENERATE SCHEDULES
''' We will use Dijkstra repeatedly to get feasible schedules for each ship
    Inputs:
    Planning horizon: 15 days
    Planning time step: 1 day

'''
# USE dijkstra algorithm (astar with 0 heuristic)
graph = MyGraph(edgeDict)
print(graph.neighbors('r1'))

start = 'r1'
goal = None
SA = Search(graph, start, goal)
parents, optimal_cost = SA.a_star_search(h_type='zero', visualize=False)
## RECONSTRUCT THE PATH
#path = reconstruct_path(parents, start, goal)
pdb.set_trace()












