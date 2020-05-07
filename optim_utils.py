import pdb
### older functions process results
#def get_ship_schedules(varSchedules):
#    solShipSchedule = {}
#    for ship in ship_schedule.keys():
#        for num in range(len(ship_schedule[ship])):
#            if varSchedules[ship, num] == 1:
#                solShipSchedule.update({ship: ship_schedule[ship][num]})
#    return solShipSchedule 
#

def get_ship_schedules(varSchedules, method):
    solShipSchedule = {}
    if method == "GEN":
        # WHEN FORMAT WAS GEN SCHEDULES
        for row in varSchedules:
            if varSchedules[row].value == 1:
                ship = row[0]
                schedule = row[1]
                solShipSchedule.update({ship: schedule})
    elif method == "NETW":
        path = []
        sum_ = 0
        for row in varSchedules:
            if varSchedules[row].value == 1:
                ship = row[0]
                sum_ +=1 
                try: 
                    path = solShipSchedule[ship]
                    path.append(row[2])
                    path.append(row[4])
                except:
                    path.append(row[2])
                    path.append(row[4])
                # NEED TO ORDER THE PATH ACCORDING TO DAY
                solShipSchedule.update({ship: path[:]}) #deep copy
    print("path elements: ",sum_)
    return solShipSchedule
#def get_cmc_assigned(varCMC):
#    solShipCMC = {}
#    for ship in ship_schedule.keys():
#        for cmc in cmcS[ship]['ALL_CMCs']:
#            for day in days:
#                for region in regions:
#                    if varCMC[ship, cmc['CMC'], day, region] == 1:
#                        # find row in dfCMC corresponding to cmc number
#                        # ignore the first two columns
#                        cmcSpecs = dfCMC.loc[(dfCMC['CMC']==cmc['CMC'])][cmcCols[2::]]
#                        solShipCMC.update({ship: {'num': cmc['CMC'],'spec':cmcSpecs}})
#    return solShipCMC

def get_cmc_assigned(varCMC):
    solShipCMC = {}
    for row in varCMC:
        if varCMC[row].value == 1:
            ship = row[0]
            cmc = row[1]
            day = row[2]
            reg = row[3]
            try:
                # if solShipCMC is not empty
                temp = solShipCMC[ship]
                temp.update( {day: (cmc, reg)} ) 
            except: 
                # on first iteration
                temp = {day: (cmc, reg)}
            solShipCMC.update({ship: temp})
    return solShipCMC

#def get_accomplishLevel(varLevel):
#    accomplishment = {}
#    for m in missionTypes:
#        for n in missionNums:
#            for r in regions:
#                for d in days:
#                    ''' '''
#                    if (m, n, r, d) in varLevel:
#                        if varLevel[m,n,r,d].value > 0:
#                             accomplishment.update({(m,n,r,d): varLevel[m,n,r,d].value})
#    return accomplishment

def get_accomplishLevel(varLevel):
    accomplishment = {}
    for row in varLevel:
        if varLevel[row].value > 0:
            accomplishment.update({row: varLevel[row].value})
    return accomplishment

#def get_finishedMissions(varFin):
#    finished = {}
#    for m in missionTypes:
#        for r in regions:
#            for d in days:
#                ''' '''
#                if (m,r,d) in varFin:
#                    if varFin[m,r,d].value == 1:
#                        finished.update({(m,r,d): varFin[m,r,d].value})
#    return finished

def get_finishedMissions(varFin):
    finished = []
    for row in varFin:
        if varFin[row].value == 1:
            finished.append(row)
    return finished
