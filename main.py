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
# need to forward fill for merged cells in the index
dfCMC = pd.read_excel(path, sheet_name=0, engine='odf').fillna(method='ffill')

#### LOAD SHIP SET
path = "./ships.ods"
dfShips = pd.read_excel(path, engine='odf', index_col="Name")

### LOAD REGION SET
path = "./region.ods"
dfRegions = pd.read_excel(path, engine='odf', sheet_name=0)
dfArcs = pd.read_excel(path, engine='odf', sheet_name=1)

#### CREATE Dictionaries for ease of use
#dfMissionDict = dfMission.to_dict('index')
#dfCMC = 

# Build list and  dictionary of edges. Should be undirected!
# lists are not hashable types, thus cannot be dictionary keys
edgeList = [(dfArcs.loc[i]['Arcs'], dfArcs.loc[i]['Unnamed: 1'], dfArcs.loc[i]['Length(nm)']) for i in range(len(dfArcs))]

edgeDict = {(dfArcs.loc[i]['Arcs'], dfArcs.loc[i]['Unnamed: 1']): dfArcs.loc[i]['Length(nm)'] for i in range(len(dfArcs))  }
edgeDictMirror = {(dfArcs.loc[i]['Unnamed: 1'], dfArcs.loc[i]['Arcs']): dfArcs.loc[i]['Length(nm)'] for i in range(len(dfArcs))  }
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

def generate_schedules(dfShips, regenerate=False, saveToDisk=False):
    #### GENERATE SCHEDULES
    ''' We will use Dijkstra repeatedly to get feasible schedules for each ship
        Condition:
        if GEN_SCHEDULES == True, then we will regenerate schedules, else
                                  read from disk
       
        Assumption:
        -Planning horizon: 15 days
        -Planning time step: 1 day
        -Ship can participate in region if arrives with 16.0 hrs of day remaining
        
        Also:
        -When in Transit, add rTransit region
    '''
    if regenerate==True:
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

        ship_inst = 1
        for index, info in dfShips.iterrows():
           
            # USE dijkstra algorithm (astar with 0 heuristic)
            #print(graph.neighbors('r1'))
            iter_ = 0
            full_schedule = []  #full schedule for one ship
            while iter_<= schedule_limit: 
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
            
            # assign to each ship name (given by index) candidate schedules
            ship_schedule.update({index: full_schedule[:]})
           
            # if testing just 1 ship
            if ship_inst >= 3:
                break
            ship_inst += 1
    else:
        #READ schedules from disk if they exist
        test = pd.read_excel("testSchedule.xls")
        ship_schedule = test.to_dict('list')

        # remote 'quotes' from read schedules
        for item in ship_schedule.keys():
            temp = []
            for subitem in ship_schedule[item]:
                temp.append(eval(subitem))
            ship_schedule.update({item: temp})

    # Save schedules to disk
    if saveToDisk == True and regenerate == True:
        dfSave = pd.DataFrame.from_dict(ship_schedule)
        dfSave.to_excel("testSchedule.xls", index=False)
    
    return ship_schedule

#### RETRIEVE CANDIDATE SCHEDULES
ship_schedule = generate_schedules(dfShips, regenerate=False, saveToDisk=False)

#### BUILD MODEL FOR SOLVING
''' TO debug model do:
    model.pprint()
'''
from pyomo.environ import *
from pyomo.opt import SolverFactory


## Create some variables for ease of use
days = np.arange(1,16).tolist()      
regions = dfRegions['Region'].tolist()

# Create Value data
#[(m,n,r,d) for m in 

# only select cmcs from the dfShips list
cmcS = {}
cmcCols = dfCMC.columns.tolist()
for ship in ship_schedule.keys():

    # GET ALL THE CMCS for current ship 
    cmcTemp = [dfShips.loc[ship]['CMC']]
    cmcTemp.append(dfShips.loc[ship]['Unnamed: 8'])
    cmcTemp.append(dfShips.loc[ship]['Unnamed: 9'])
    
    #collect cmcs
    tempArray=[]
    for cmcValue in cmcTemp:
        #remove ship class column
        row = dfCMC.loc[(dfCMC['CMC']==cmcValue) ][cmcCols[1::]]
        tempArray.append(row.to_dict('records')[0])

    # APPEND TRANSIT CMC
    row = dfCMC.loc[(dfCMC['Ship Class']=='ALL SHIPS')][cmcCols[1::]]
    tempArray.append(row.to_dict('records')[0])

    # add all cmcs to current ship
    cmcS.update({ship: {'Class': dfShips.loc[ship]['Class'], 'ALL_CMCs': tempArray[:]}})

#cmcS = {ship: info for ship in ship_schedule.keys() for index, info in dfCMC.iterrows() if info['Ship Class']==dfShips.loc[ship]['Class'] or info['Ship Class'] == 'All SHIPS'}
# initialize model
model = ConcreteModel()

# binary variable for schedule selected for ship
model.schedule = Var( ((ship, schedule) for ship in ship_schedule.keys() for schedule in range(len(ship_schedule[ship]))), within=Binary, initialize=0)

# binary variables for concurrent mission selection. Indexed by ship, cmc, day, and region
pdb.set_trace()
model.cmc = Var( ((ship, cmc['CMC'], day, region) for ship in ship_schedule.keys() for cmc in cmcS[ship]['ALL_CMCs'] for day in days for region in regions), within=Binary, initialize=0)

#for index, row in dfCMC.iterrows():
#    if row["Ship Class"] == prevClass:
#        temp

#### TIMING CODE
from codetiming import Timer



