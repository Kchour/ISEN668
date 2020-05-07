#!/usr/bin/env python3
''' This script will solve a network flow formulation for schedules'''
import pandas as pd
import pdb
from utils import *
from search_algorithms import *
import networkx as nx
import random as rdm
import numpy as np
from optim_utils import *

global ship_speed, shipLimit, dayHorizon, schedule_limit, filename

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
''' get a list of ship names based on quantity '''
def get_ships(shipLimit):
    
    ship_inst = 1
    shipList = []
    for index, info in dfShips.iterrows():
        #skip ship if not available
        if info['Avail'] == 'x':
            if ship_inst > shipLimit:
                    break
            shipList.append(index)
            ship_inst += 1
    return shipList

''' for each ship, region, day...see if there is a reasonable mission/cmc pair '''
def generate_feasible_srd(ships, cmcS, mnrd):
    # ships: ['Monterey', ... ]
    #To access ship cmc: cmcS['Monterey']['ALL_CMCs']

    ship_srd = {}
    # go through mnrd to get type
    for ship in ships:
        for cmc in cmcS[ship]['ALL_CMCs']:
            for row in mnrd:
                # 1st elem is a tuple, [0] is mission type
                type_ = row[0][0]
                day_ = row[0][3]
                region_ = row[0][2]

                # mission value row[1]
                value_ = row[1]
                if value_ > 0 and cmc[type_] > 0:
                    ship_srd.update({(ship, region_, day_): True})
    return ship_srd 
   
''' generate all shortest path distances'''
def generate_asp(regions):
    graph = MyGraph(edgeDict)
    asp = {}
    for r1 in regions:
        SA = Search(graph, r1, None)
        parents, optimal_cost = SA.a_star_search(h_type='zero', visualize=False)
        asp.update({r1: optimal_cost})
    return asp

method = 'GEN' # NETW, GEN: will employ network flow or schedule generation approach
ship_speed = 16  #knots
cutoff_frac = 1/3  # round down if fractional day is less than this
shipLimit = 3      #3,18 SHIPS IS THE MAX. This function is done outside now
dayHorizon = 15
schedule_limit = 10 # 5, 10, 50 limit the number of schedules generated
filename = "./pickle/spd" + '%i' % ship_speed
filename += '_ships' + '%i' % shipLimit
filename += '_days' + '%i' % dayHorizon 
filename += '_method' + method
filename += '_schedules' + '%i' % schedule_limit + '.pkl'
def generate_schedules(dfShips, asp, desiredShips, regenerate=False, saveToDisk=False):
    global ship_speed, shipLimit, dayHorizon, schedule_limit
    #### GENERATE SCHEDULES
    ''' We will use Dijkstra repeatedly to get feasible schedules for each ship
        Condition:
        if GEN_SCHEDULES == True, then we will regenerate schedules, else
                                  read from disk
        Inputs: 
        -dfShips(df), 
        -all shortest paths(dict), 
        -desired ships to use (list)
        -regenerate schedules (bool), 
        -saveToDisk (bool)

        Assumption:
        -Planning horizon: 15 days
        -Planning time step: 1 day
        -Ship can participate in region if arrives with 16.0 hrs of day remaining
        - 
        Also:
        -When in Transit, add rTransit region
    '''
    excelFileName_ = filename + ".xlsx"
    if regenerate==True:
        ship_schedule = {index:[] for index, info in dfShips.iterrows() if info["Avail"] == 'x'}
        # convert ship info to dict for ease of use
        dfShips_dict = dfShips.to_dict()
        # Initialize ship_schedule dict
        ship_schedule = {}
        #ship_speed = 16  #knots
        #cutoff_frac = 1/3  # round down if fractional day is less than this
        ##shipLimit = 18      #18 SHIPS IS THE MAX. This function is done outside now
        #dayHorizon = 15
        #schedule_limit = 10 #100,1000,50000 limit the number of schedules generated
        # Create a graph based on edges given
        #graph = MyGraph(edgeDict)

        ship_inst = 1
        #for index, info in dfShips.iterrows():
        for ship in desiredShips:
            info = dfShips.loc[ship]
            # Skip ship if not available
            # USE dijkstra algorithm (astar with 0 heuristic)
            #print(graph.neighbors('r1'))
            iter_ = 0
            full_schedule = []  #full schedule for one ship
            while iter_< schedule_limit: 
                # temporary schedule list
                start = info["Start Region"]
                temp = []   #container for acceptable region
                day = 0 #current day
                while True:
                    # Incorporate ship start day
                    if day < dfShips.loc[ship]['Start Day']-1:
                        day+=1
                        temp.append("None")
                    else:
                        # if temp is greater than 15 stop
                        if len(temp) >= dayHorizon:
                            break

                        # test mission/cmc is reasonable
                        potential_ = asp[start]
                        feasibleTargets = {}
                        for r in potential_.keys():
                            # round down if fractional part is less than 8 hours
                            #time = asp[start][rand_sel]/ship_speed/24.0
                            time = potential_[r]/ship_speed/24.0
                            if time-np.floor(time) <= cutoff_frac:
                                time = np.floor(time).astype('int')
                            else:
                                time = np.ceil(time).astype('int')
                            day2 = day + time + 1
                            if day2 <= dayHorizon: 
                                try: 
                                    feasible = srd[(ship, r, day2)]
                                except:
                                    feasible = False
                                    ''' may need to update feasibleTargets here wiht current pos'''
                                    pass
                                if feasible == True:
                                    feasibleTargets.update({r: time})
                        # Random selection. modified to use asp instead 
                        #rand_sel = rdm.choice(list(asp[start].keys())) 
                        # Random selection. modified to use asp and test feasibility
                        rand_sel = rdm.choice(list(feasibleTargets.keys()))
                        # Find Transit Time in days and decide what to do
                        #time = optimal_cost[rand_sel]/ship_speed/24.0
                        #time = asp[start][rand_sel]/ship_speed/24.0
                        # now using using feasible Targets, returns time in days
                        time = feasibleTargets[rand_sel]

                        # round down if fractional part is less than 8 hours
                        #if time-np.floor(time) <= cutoff_frac:
                        #    time = np.floor(time).astype('int')
                        #else:
                        #    time = np.ceil(time).astype('int')

                        for i in range(time):
                            temp.append('rTransit')
                        # Append the desired selection
                        temp.append(rand_sel)
                       
                        # start search from previous desired selection
                        start = rand_sel
                        #update current day
                        day += time + 1

                # update current ship schedule
                full_schedule.append(temp[:])
                
                #update iter_
                print(iter_)
                iter_ +=1
            
                # assign to each ship name (given by index) candidate schedules
                #ship_schedule.update({index: full_schedule[:]})
                ship_schedule.update({ship: full_schedule[:]})
               
                # # if testing just 1 ship
                # ship_inst += 1
    else:
        #READ schedules from disk if they exist
        #test = pd.read_excel("Schedule_10.xlsx")
        test = pd.read_excel(excelFileName_)
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
        #dfSave.to_excel("Schedule_10.xlsx", index=False)
        dfSave.to_excel(excelFileName_, index=False)
    
    return ship_schedule

# Generate a network for all possible schedules
def network_flow_genSchedules(ships, days, regions, shipSpeed, cutOff, asp):
    ''' Create all the edges in our network'''
    networkSRD = []
    graph = MyGraph(edgeDict)
    # Potential locations
    potentials = regions[:]         #deep copy
    potentials.append('rTransit')
    # Include intial condition, day = 0
    dayLength = len(days)
    days.insert(0, 0)
    for ship in ships:
        #pdb.set_trace()
        #for row in networkSRD:
        #    print(row)
        for day in days:
            for region in regions:
                # get neighbors of region that can be reached within a day
                # include rTransit 
                # Optional: include ship_srd 
                #neighbors = graph.neighbors(region) 
                # all nodes connected to rTransit
                #neighbors.append('rTransit')
                nextDayList = []
                transitList = []
                for n in potentials:
                    ## TODO GET RID OF SELF LOOPS
                    #if n != region:
                    # dont allow transits at start of 15 day
                    if day == dayLength:
                        if n == 'rTransit':
                            break
                    #compute time required to reach a neighbor
                    if n != 'rTransit':
                        time = asp[region][n]/shipSpeed/24.0
                        if time-np.floor(time) <= cutOff:
                            time = np.floor(time).astype('int')
                        else:
                            time = np.ceil(time).astype('int')
                    else:
                        time = 0.0
                    if time < 1.0:
                        nextDayList.append(n)
                    elif time < 2.0:
                        transitList.append(n)
                    else: 
                        break
                # draw arcs between regions between 1 consecutive day 
                if len(nextDayList)>0:
                    for f in nextDayList:
                        networkSRD.append((ship, day, region, day+1, f))
                if len(transitList) > 0:
                    # Transit list
                    if day < dayLength:
                        for f in transitList:
                            networkSRD.append((ship, day+1, 'rTransit', day+2, f))
    pdb.set_trace()
    return networkSRD
#### Define Data for modeling

## Create some variables for ease of use
days = np.arange(1,dayHorizon+1).tolist()                 #days 1-15  
regions = dfRegions['Region'].tolist()          
missionNums = dfMission.index.values.tolist()   #for m

# Create Value data along with its indices
mnrd = []
valueMNRD = {}
# Also create mission complete variables
mrd = []
Qrd = {}

# process mission set
for n, nInfo in dfMission.iterrows():
    for d in range(nInfo['Start Day'], nInfo['Day End']+1):
        mnrd.append([(nInfo['Type'], n, nInfo['Region'], d), nInfo['Value'], {'Required': (nInfo['Required'], nInfo['Unnamed: 8']) }])
        try:
            # if Qrd is non-empty
            temp = Qrd[(nInfo['Region'],d)]
        except:
            # if Qrd is empty
            temp = [(nInfo['Type'], nInfo['Required'])]
            Qrd.update({ (nInfo['Region'], d): temp})
           
            temp = Qrd[(nInfo['Region'],d)] 
            temp.append((nInfo['Type'], nInfo['Unnamed: 8']))
            Qrd.update({ (nInfo['Region'], d): temp})

        
        temp.append((nInfo['Type'], nInfo['Required']))
        temp.append((nInfo['Type'], nInfo['Unnamed: 8']))
        Qrd.update({ (nInfo['Region'], d): temp})
        
        #Qrd.update( {(nInfo['Region'],d):(nInfo['Type'], nInfo['Required'])}) 
        #Qrd.update( (nInfo['Type'], nInfo['Unnamed: 8']))
        valueMNRD.update({(nInfo['Type'], n, nInfo['Region'], d): nInfo['Value']})

# get a list of ships
shipList = get_ships(shipLimit)

# Create cmc selection variable. Only select cmcs from the dfShips list
cmcS = {}
cmcCols = dfCMC.columns.tolist()
missionTypes = cmcCols[2::]
#for ship in ship_schedule.keys():
for ship in shipList:
    # GET ALL THE CMCS for current ship 
    cmcTemp = [dfShips.loc[ship]['CMC']]
    cmcTemp.append(dfShips.loc[ship]['Unnamed: 8'])
    cmcTemp.append(dfShips.loc[ship]['Unnamed: 9'])
    
    #collect cmcs
    tempArray=[]
    
    #if ship == 'Vandegrift':
    #    pdb.set_trace()
    for cmcValue in cmcTemp:
        #ignore nan
        if str(cmcValue) != 'nan':        
            #remove ship class column
            row = dfCMC.loc[(dfCMC['CMC']==cmcValue) ][cmcCols[1::]]
            tempArray.append(row.to_dict('records')[0])

    # APPEND TRANSIT CMC
    #row = dfCMC.loc[(dfCMC['Ship Class']=='ALL SHIPS')][cmcCols[1::]]
    #tempArray.append(row.to_dict('records')[0])

    # add all cmcs to current ship
    cmcS.update({ship: {'Class': dfShips.loc[ship]['Class'], 'ALL_CMCs': tempArray[:]}})

# get all shortest paths dist pair
asp = generate_asp(regions)

#is feasible mission/cmc on day and region?
srd = generate_feasible_srd(shipList, cmcS, mnrd)

## GET SCHEDULES
# RETRIEVE CANDIDATE SCHEDULES USING GEN METHOD
if method == "GEN":
    pdb.set_trace()
    ship_schedule = generate_schedules(dfShips, asp, shipList, regenerate=True, saveToDisk=True)
# NETWORK FLOW FOR SCHEDULES
elif method == "NETW":
    networkSRD = network_flow_genSchedules(shipList, days, regions, ship_speed, cutoff_frac, asp)  
pdb.set_trace()

#### BUILD MODEL FOR SOLVING
''' TO debug model do:
    model.pprint()
'''
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core.expr import current as EXPR #to allow printing of expressions
'''print(EXPR.expression_to_string(e, verbose=True)) '''

## initialize model
model = ConcreteModel()

## Define Variables
if method=="GEN":
    # binary variable for schedule selected for ship (Y_{ps})
    model.schedule = Var( ((ship, schedule) for ship in ship_schedule.keys() for schedule in range(len(ship_schedule[ship]))), within=Binary, initialize=0)
elif method == "NETW":
    model.schedule = Var( (row[0], row[1], row[2], row[3], row[4]) for row in networkSRD)

# binary variables for concurrent mission selection. Indexed by ship, cmc#, day, and region (W)
model.cmc = Var( ((ship, cmc['CMC'], day, region) for ship in ship_schedule.keys() for cmc in cmcS[ship]['ALL_CMCs'] for day in days for region in regions), within=Binary, initialize=0)

# continuous variables up to 1 for accomplishment level
model.level = Var((row[0] for row in mnrd), bounds = (0, 1), initialize=0)

# binary variables for fully-accomplished mission (V)
model.finished = Var(((row[0][0], row[0][2], row[0][3]) for row in mnrd), within=Binary, initialize=0)


## Build constraints
model.constraints = ConstraintList()

# schedule constraints
if method == "GEN":
    # For each ship, sum over each s in schedule (indexed by natural numbers) (T1)
    for ship in ship_schedule.keys():
        model.constraints.add(
             sum(model.schedule[ship, schedule] for schedule in range(len(ship_schedule[ship])))<=1 
        )
elif method == "NETW":
    for ship in shipList:
        for day in days:
            for region in regions:
                pass  


    
# For each ship, day, and region...a cmc is selected iff a schedule is selected with the same region on the same day (T2)
for ship in ship_schedule.keys():
    for day in days:
        for region in regions:
            ''' lhs '''
            lhs = sum(model.cmc[ship, cmc['CMC'], day, region] for cmc in cmcS[ship]['ALL_CMCs'])
            ''' rhs '''
            rhs = sum(model.schedule[ship, schedule] for schedule in range(len(ship_schedule[ship])) if ship_schedule[ship][schedule][day-1] == region)
            model.constraints.add(lhs == rhs) 
            #for schedule in range(len(ship_schedule[ship])):
            #    # need to use day-1 for index purposes
            #    if ship_schedule[ship][schedule][day-1]==region:
            #        #model.schedule[ship, schedule].pprint()
            # Sum lhs over cmcs availbe for ship
            #model.constraints.add(
            #    sum(model.cmc[ship, cmc['CMC'], day, region] for cmc in cmcS[ship]['ALL_CMCs'] ) <= sum(model.schedule[ship, schedule] for schedule in range(len(ship_schedule[ship])) if ship_schedule[ship][schedule][day]==region )
            #)

# Accomplishment level for each m,r,d sum over n....(T3)
for m in missionTypes:
    for r in regions:
        for d in days:
            ''' old implementation'''
            #''' lhs '''
            #lhs = 0
            #rhs = 0
            #for n in missionNums:
            #    if (m, n, r, d) in model.level:
            #        lhs += model.level[m,n,r,d]
            #        ''' rhs '''
            #        for ship in ship_schedule.keys():
            #            for cmc in cmcS[ship]['ALL_CMCs']:
            #                rhs += cmc[m]*model.cmc[ship, cmc['CMC'], d, r] 
            #        
            #        ''' add constraints '''
            #        model.constraints.add(lhs<= rhs)
            if (m,r,d) in model.finished:
                ''' lhs '''
                lhs =sum(model.level[m,n,r,d] for n in missionNums if (m,n,r,d) in model.level)
                ''' rhs '''
                rhs = sum(cmc[m]*model.cmc[ship, cmc['CMC'], d, r] for ship in ship_schedule.keys() for cmc in cmcS[ship]['ALL_CMCs']) 
                ''' add constraints '''
                model.constraints.add(lhs <= rhs)


### We have completed a task with somemission type, at some region, and some day...(T4hall)
for m in missionTypes:
    for r in regions:
        for d in days:
            ''' lhs '''
            if (m, r, d) in model.finished:
                lhs = model.finished[m,r,d]
                ''' rhs '''
                rhs = sum(model.level[m,n,r,d] for n in missionNums if (m, n, r, d) in model.level )
                '''add constraint'''
                model.constraints.add(lhs <= rhs)

### dependencies...(T5hall)
for m in missionTypes:
    for n in missionNums:
        for r in regions:
            for d in days:
                ''' '''
                for m2 in missionTypes:
                    # THE SECOND N NEEDS TO CHANGE DUDE
                    if (m, n, r, d) in model.level and (m2, r, d) in model.finished and (m, m2) in Qrd[(r, d)]:
                        #if ('SUW','m76','r13',10) == (m,n,r,d) or ('AD','r13',10) == (m2,r,d):
                        #    pdb.set_trace()
                        model.constraints.add( model.level[m, n, r, d] <= model.finished[m2, r, d])
                        


### add objective function

# define objective function with model as input
def obj_rule(model):
    sum_ =  sum(valueMNRD[(m, n, r, d)]*model.level[m, n, r, d] for m in missionTypes for n in missionNums for r in regions for d in days if (m, n, r, d) in model.level)
    return  sum_

# add obj function to model
model.obj = Objective(rule=obj_rule, sense=maximize)


### Solve the problem

pdb.set_trace()
opt = SolverFactory('cbc')
results = opt.solve(model, tee=True)
# store results back in model for ease of access
model.solutions.store_to(results)
results.write()


# Test functions
testa = get_ship_schedules(model.schedule, method)
testb = get_cmc_assigned(model.cmc)
testc= get_accomplishLevel(model.level)
# for row in testc.items(): print(row)

testd= get_finishedMissions(model.finished)
# debug for finished mission types
#for m in missionTypes:
#    for r in regions:
#        for d in days:
#            ''' '''
#            if (m,r,d) in model.finished:
#                print(model.finished[m,r,d].value)
# Add a progress bar for each mission

# Save model and results
#global ship_speed, shipLimit, dayHorizon, schedule_limit
pdb.set_trace()
import cloudpickle
#filename = "./pickle/spd" + '%i' % ship_speed
#filename += '_ships' + '%i' % len(shipList)
#filename += '_days' + '%i' % dayHorizon 
#filename += '_schedules' + '%i' % schedule_limit + '.pkl'
with open(filename, mode='wb') as file: cloudpickle.dump(model,file)
with open(filename+'.result', mode='wb') as file: cloudpickle.dump(results,file)

# To reload
#with open('test.pkl', mode='rb') as file: model = cloudpickle.load(file)
pdb.set_trace()
