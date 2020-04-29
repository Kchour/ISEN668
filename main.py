#!/usr/bin/env python3

import pandas as pd
import pdb
from utils import *
from search_algorithms import *
import networkx as nx
import random as rdm
import numpy as np

#### LOAD MISSION SET
path = "./mission_set.ods"
dfMission =pd.read_excel(path, sheet_name=0, engine='odf', index_col='Mission')

#### LOAD CMC SET
path = "./cmc.ods"
dfCMC = pd.read_excel(path, sheet_name=0, engine='odf')

#### LOAD SHIP SET
path = "./ships.ods"
dfShips = pd.read_excel(path, engine='odf', index_col="Name")

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
    Assumption:
    -Planning horizon: 15 days
    -Planning time step: 1 day
    -Ship can participate in region if arrives with 16.0 hrs of day remaining
    
    Also:
    -When in Transit, add rTransit region
'''
ship_schedule = {index:[] for index, info in dfShips.iterrows() if info["Avail"] == 'x'}
# convert ship info to dict for ease of use
dfShips_dict = dfShips.to_dict()

# Initialize ship_schedule dict
ship_schedule = {}
ship_speed = 16  #knots
cutoff_time = 16.0  # hrs must remain on arrival
schedule_limit = 10 # limit the number of schedules generated
# Create a graph based on edges given
graph = MyGraph(edgeDict)
for index, info in dfShips.iterrows():
   
    # USE dijkstra algorithm (astar with 0 heuristic)
    #print(graph.neighbors('r1'))
    iter_ = 1
    full_schedule = []  #full schedule for one ship
    while iter_< schedule_limit: 
        # temporary schedule list
        start = info["Start Region"]
        temp = [start]
        while True:
            goal = None
            SA = Search(graph, start, goal)
            parents, optimal_cost = SA.a_star_search(h_type='zero', visualize=False)
            ## RECONSTRUCT THE PATH
            #path = reconstruct_path(parents, start, goal)
           
            # Random selection
            rand_sel = rdm.choice(list(optimal_cost.keys())) 

            # Find Transit Time in days and decide what to do
            time = optimal_cost[rand_sel]/ship_speed/24.0
            
            # round down if fractional part is less than 8 hours
            if time-np.floor(time) <= 1/3:
                time = np.floor(time).astype('int')
            else:
                time = np.ceil(time).astype('int')

            for i in range(time):
                temp.append('rTransit')
            # Append the desired selection
            temp.append(rand_sel)
           
            # start search from previous desired selection
            start = rand_sel

            # if temp is greater than 15 stop
            if len(temp) > 15:
                temp = temp[0:15]
                break
        # update current ship schedule
        full_schedule.append(temp[:])
        
        #update iter_
        iter_ +=1
    pdb.set_trace()   
    
    # assign to each ship name (given by index) candidate schedules
    ship_schedule.update({index: full_schedule[:]})
    
    
    break
pdb.set_trace()










