# Introduction
This project implements both the schedule-generation scheme and network flow formulation to solve a ship 
routing-scheduling problem. Specifically, a Mixed Integer Linear Program (MILP) is solved. The problem is as follows. There are multiple ships (n ships), with different capabilities (0-100%), going towards the completion of several missions (m missions). The goal is to complete as many missions as possible, which span a geographical region (weighted graph G=(R,E, w)), for some time-span (d days). 

Formally, we are given a graph `G=(R,E,w)`, `N` ships, `M` missions, `c_n` capabilities (a an array of size 11, each element 0-100%) of each ship n \in N, and `D` a planning horizon (i.e. 1-15 days), `v_m` value of mission m \in M. Each ship is assumed to travel at the same average speed, during transits between regions on the graph. A ship cannot be at more than 2 locations on any given day. Also, if transits take more than 8 hours (can be defined arbitrarily) for a given ship, the ship is considered to be in a psuedo-region `rTransit`.

The problem is modeled using Pyomo, a python-based library for modeling optimization problems. Essentially, it is a front-end,
which allows access to different solvers (GLPK, CBC, SCIP, etc.).

Finally, the branch-and-bound algorithm is implemented with best-first-search approach. 

Further work will look into modeling logistics and improving the implementation process

# Requirements
This project was developed on an Ubuntu 18.04 system, running Python3 version 3.6.9. An earlier version of both should be fine, but there is no guarantee the files will execute

## Python dependencies
Python dependencies can be installed using `requirements.txt` as follows:
`$ python3 -m pip install -r requirements.txt`

## System dependecies
System dependecies include the optimization solvers, and their dependecies

i.e. to use Sandia's COIN-OR/CBC solver, first install it: `sudo apt-get install -y coinor-cbc`

i.e to use GLPK: `sudo apt-get install -y glpk-utils`

i.e. to use SCIP: May need to install from source

Other sys deps may be related `open office` or `pandas` module: `python3 -m pip install odfpy`

# Setup Overview:

The problem instances (ship name, number, concurrent mission capabilities (CMC), mission numbers) are stored in the `.ods` (openOffice) spreadsheets. They are as follows, with the most relevant information highlighted:

`ships.ods`: contains ship number, `name`, `availability`, class, type, `start day`, `start region`, and `CMCs`

`regions.ods`: defines a graph over our geographical location (Korean Peninsula), by providing edges and respective weights (nautical miles)

`mission_set.ods`: defines our mission numbers (m1-m80) to be completed. Contains `Mission`, `include`, `Type`, `Region`, `Start Day`, `End Day`, `Value`, and `Required`

`cmc.ods`: defines accomplishment levels for each CMC number. There are 11 mission types here: `AD`, `TBMD`, `ASW`, `SUW`, `Strike`, `NSFS`, `MIO`, `MCM`, `Mine`, `Intel`, `SubIntel`

Processed data is stored in the folder `pickle`, which mainly relies on the `cloudPickle` module to serialize python objects. However, not everything in this folder is a cpickle object. We will come back to this later

To run, there are 2 files of interest: 

`main_Heuristic.py`: utilizes the schedule generation scheme, a heuristical approach for solving scheduling problems

`main_networkFlow.py`: utilizes a network flow formulation to our problem formulation, and solves the problem to optimality. Obviously, this approach gives us better bounds compared to using the previous approach, but takes more time to terminate. However, the heuristic approach is also implemented here. The approach can be adjusted with a flag

Either can be run by invoking python3 i.e.`python3 main_FILENAME.py`. 


## Code setup
First, we load the problem instances into memory with `pandas`. Again, they are the mission sets, cmcs, ships, and regions. Most of the next few functions are pretty self explanatory.

Open `main_networkFlow.py`. From line line `109-114`, we can set a few parameters:

`method`: either use schedule generation or network flow (values: GEN, NETW)

`ship_speed`: average transit speed in nautical miles (default `16`)

`cutoff_frac`: duration counted as fraction of a day. If transit time takes more time than this duration, it is counted as a full day (default `1/3`)

`shipLimit`:  number of ships to include in our problem (default `18`)

`dayHorizon`: planning horizon (default 15)

`schedule_limit`:  number of schedules to generate per ship (default `5`)

The next function `generate_schedules` as its name implies, creates feasible schedules for each ship based on its current location on the graph G. We first use `dijkstra` to calculate the shortest path between any 2 regions on the graph. Together with `ship_speed`, we can approximate the transit time between any two regions. Then, for each ship, we generate several arrays, each of length defined by `dayHorizon`. The i-th position in the array corresponds to the the i-th day. We then continually append a region `r` from G at random, starting from the ships `start day`, while taking into account transit times. This is repeated until all ships have the same number of schedules

Note, a schedule is considered feasible for a ship if: 1) that ship has the capability of accomplishing some mission in that region on that day 2) The missions available to the ship that day and region, has a nonzero value

## Visualization
We can visualize results with a Gantt Chart. A premade one is already given by `gantt.html`

After generating results using the main script from above, we can use `plotter.py` to create a gantt chart. Adjust the parameters in line `19-23` to match with results generated from above.

Keep BB (branch and bound experiment) and LP (linear program experiment) as false

## Debugging

python debugger is used. See https://docs.python.org/3/library/pdb.html

