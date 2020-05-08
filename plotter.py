#!/usr/bin/env python3
import pdb
import cloudpickle
from optim_utils import *
import pandas as pd
'''Case 1:   method     shipSpeed     shipLimit     dayHorizon  scheduleLimit   GotData/Proc?
             NETW       16              18          15              -               Y/N
             NETW       16              3           15              -               Y/N
             NETW       16              1           15              -               DEBUG
             GEN        16              3           15              5
             GEN        16              3           15              10
             GEN        16              3           15              15
             GEN        16              18          15              5               Y/N
             GEN        16              18          15              10
             GEN        16              18          15              15
             GEN        16              1           5              5               BB case
'''    
method = 'NETW' # NETW, GEN
scheduleLimit = 5  #Only relevant to GEN
shipSpeed = 16
shipLimit = 1
dayHorizon = 5
BB = True       # USE BB RESULTS?

if method == "GEN":
    filename = "./pickle/spd" + '%i' % shipSpeed
    filename += '_ships' + '%i' % shipLimit
    filename += '_days' + '%i' % dayHorizon 
    filename += '_method' + method
    filename += '_schedules' + '%i' % scheduleLimit
elif method == 'NETW':
    filename = "./pickle/spd" + '%i' % shipSpeed
    filename += '_ships' + '%i' % shipLimit
    filename += '_days' + '%i' % dayHorizon 
    filename += '_method' + method
#filename = "./pickle/spd" + '%i' % shipSpeed
#filename += '_ships' + '%i' % shipLimit
#filename += '_days' + '%i' % dayHorizon 
#filename += '_method' + method  #RENAME SOME FILES 
#filename += '_schedules' + '%i' % scheduleLimit + '.pkl'
if BB==True:
    with open(filename+'.bb.model.pkl', mode='rb') as file: model = cloudpickle.load(file)
    with open(filename+'.bb.result.pkl', mode='rb') as file: optResults = cloudpickle.load(file)
else:
    with open(filename+'.model.pkl', mode='rb') as file: model = cloudpickle.load(file)
    with open(filename+'.result.pkl', mode='rb') as file: optResults = cloudpickle.load(file)

# See statistics
#optResults.Problem[0]
###Name
###Lower bound
###Upper bound
###Number of objectives
###Number of constraints
###Number of variables
###Number of binary variables
###Number of integer variables
###Number of continuous variables
###Number of nonzeros
###Sense
#optResults.Solver[0]
####Name
####Status
####Return code
####Message
####User time
####System time
####Wallclock time
####Termination condition
####Termination message
####Statistics
####Error rc
####Time
#optResults.Solution[0]
####Gap
####Status
####Message
####Problem
####Objective
####Variable
####Constraint

## use object.__dict__ to see all members

## Get results from model
# get ship names, just build a dictionary for now

cmcAssigned = get_cmc_assigned(model.cmc)
accompLevel = get_accomplishLevel(model.level)
finishedMiss = get_finishedMissions(model.finished)

# Load generated schedules from spreadsheet (heuristic approach)
if method == "GEN":
    shipSchedule = get_ship_schedules(model.schedule, method, dayHorizon)
    excelFileName_ = filename + ".xlsx"
    dfExcel = pd.read_excel(excelFileName_)
    genSchedules = dfExcel.to_dict('list')

    #Lookup shipSchedule from generated ones
    shipScheduleActual = {}
    for i in shipSchedule.keys():
        sn = shipSchedule[i]                 #schedule number
        sA = genSchedules[i][sn]             #actual schedule
        shipScheduleActual.update({i: eval(sA)})   #update dict
else:
    shipScheduleActual = get_ship_schedules(model.schedule, method, dayHorizon)

# Create Gantt Chart

# generate colors dict with region
regenColors = False


#import plotly.plotly as py
import plotly.figure_factory as ff

# create a list of dict objects
df = []
annots = []
shipInst = 0
for ship in shipScheduleActual.keys():
    startDay = 1
    while startDay <= dayHorizon:    
        delta = 1   #all actions consume at least a day
        # figure out how many days in same place
        #currentReg = eval(shipScheduleActual[ship])[startDay-1]
        currentReg = shipScheduleActual[ship][startDay-1]
        if currentReg == 'None':
            startDay += 1
        else:
            for j in range(startDay-1, dayHorizon-1):
                #nextReg = eval(shipScheduleActual[ship])[j+1]
                nextReg = shipScheduleActual[ship][j+1]
                if currentReg == nextReg:
                   delta += 1
                else:
                    break
            dayFinish = startDay + delta


            #df.append(dict(Task=ship, Start=i, Finish=i+1), CMC=cmcAssigned[ship][i])
            df.append(dict(Task=ship, Start=startDay, Finish=dayFinish, Region=currentReg))
            #annots.append(dict(x = startDay+ (dayFinish-startDay)/2.0, y = shipLimit-shipInst-1, text=currentReg, showarrow=False, font=dict(color='white')))
            
            # label with cmc instead
            if startDay in cmcAssigned[ship]:
                annots.append(dict(x = startDay+ (dayFinish-startDay)/2.0, y = shipLimit-shipInst-1, text=cmcAssigned[ship][startDay][0], showarrow=False, font=dict(color='white')))
            else:
                annots.append(dict(x = startDay+ (dayFinish-startDay)/2.0, y = shipLimit-shipInst-1, text="Transit", showarrow=False, font=dict(color='white')))

            startDay = dayFinish    #next tasks starts immediately


    shipInst+=1

#colors = dict(AD='rgb(220, 0, 0)', SUW='rgb(170, 14, 200)', STRIKE=(1, 0.9, 0.16))
if regenColors == True:
    colors = {}
    import random
    # need 1-16
    for i in range(1,17):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        colors.update({'r%i' %i: 'rgb(%i, %i, %i)'%(r, g, b)})

    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    colors.update({'rTransit': 'rgb(%i, %i, %i)'%(100, 100, 100)})
else:
    colors = {'r1': 'rgb(12, 39, 234)', 'r2': 'rgb(56, 231, 86)', 'r3': 'rgb(81, 131, 222)', 'r4': 'rgb(238, 121, 108)', 'r5': 'rgb(252, 129, 116)', 'r6': 'rgb(70, 18, 137)', 'r7': 'rgb(217, 184, 0)', 'r8': 'rgb(9, 91, 210)', 'r9': 'rgb(210, 174, 226)', 'r10': 'rgb(120, 202, 94)', 'r11': 'rgb(197, 219, 1)', 'r12': 'rgb(200, 192, 120)', 'r13': 'rgb(19, 152, 229)', 'r14': 'rgb(252, 172, 205)', 'r15': 'rgb(110, 69, 43)', 'r16': 'rgb(9, 174, 94)', 'rTransit': 'rgb(100, 100, 100)'}

# Need a way to reorder the colorbar
fig = ff.create_gantt(df, colors=colors, index_col='Region', show_colorbar=True, showgrid_x=True, showgrid_y=True, group_tasks=True, bar_width=0.5)

fig['layout']['xaxis']['rangeselector']['visible'] = True
fig['layout']['xaxis']['rangeslider'] = dict(bgcolor='#E2E2E2')
fig['layout']['xaxis']['type'] = 'linear'
fig['layout']['annotations'] = annots
fig['layout']['title'] = 'Ship Capability Assignment over 15-day horizon'
fig.write_html("gantt.html")
fig.show()

##df = [dict(Task="Ship A", Start='2009-01-01', Finish='2009-02-28', Mission='AD'),
##      dict(Task="Ship B", Start='2008-12-05', Finish='2009-04-15', Mission='SUW'),
##      dict(Task="Ship C", Start='2009-02-20', Finish='2009-05-30', Mission='STRIKE')]
#df = [dict(Task="Ship A", Start=1, Finish=2, Mission='AD', ),
#      dict(Task="Ship B", Start='2', Finish='5', Mission='SUW'),
#      dict(Task="Ship C", Start='4', Finish='14', Mission='STRIKE'),
#      dict(Task="Ship A", Start=2, Finish=3, Mission='STRIKE')]
#
#colors = dict(AD='rgb(220, 0, 0)', SUW='rgb(170, 14, 200)', STRIKE=(1, 0.9, 0.16))
#fig = ff.create_gantt(df, colors=colors, index_col='Mission', show_colorbar=True, showgrid_x=True,group_tasks=True)
#
##fig = ff.create_gantt(df,colors='RdBu', index_col='Complete',show_colorbar=True)
#fig['layout']['xaxis']['rangeselector']['visible'] = True
#fig['layout']['xaxis']['rangeslider'] = dict(bgcolor='#E2E2E2')
##fig['layout']['xaxis']['type'] = 'date'
#fig['layout']['xaxis']['type'] = 'linear'
#
#
## add annotations
#annots =  [dict(x=8,y=0,text="Region r1", showarrow=False, font=dict(color='white')),
#           dict(x=(5-2)/2+2,y=1,text="Region r2", showarrow=False, font=dict(color='White')),
#           dict(x=5+4,y=2,text="Region r3", showarrow=False, font=dict(color='White'))]
#
#fig['layout']['annotations'] = annots
#fig.show()
#
##py.iplot(fig, filename='gantt-simple-gantt-chart')

pdb.set_trace()
