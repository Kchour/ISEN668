# Introduction

This project implements both the schedule-generation scheme and network flow formulation to solve a ship 
routing-scheduling problem. Specifically, a Mixed Integer Linear Program (MILP) is solved.

The problem is modeled using Pyomo, a python-based library for modeling optimization problems. Essentially, it is a front-end,
which allows access to different solvers (GLPK, CBC, SCIP, etc.).

Finally, the branch-and-bound algorithm is implemented with best-first-search approach. 

Further work will look into modeling logistics and improving the implementation process


# Dependencies

TBD


