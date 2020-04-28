#!/usr/bin/env python
from utils import PriorityQueue
import pdb
import matplotlib.pyplot as plt
import numpy as np

class Search:
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal

    def set_start(self, start):
        self.start = start

    def set_goal(self, goal):
        self.goal = goal

    def heuristic(self, a, b, type_='manhattan'):
        (x1, y1) = a
        (x2, y2) = b
        if type_ == 'manhattan':
            return abs(x1 - x2) + abs(y1 - y2)
        elif type_ == 'euclidean':
            v = [x2 - x1, y2 - y1]
            return np.hypot(v[0], v[1])
        elif type_ == 'diagonal_uniform':
            return max(abs(x1 - x2), abs(y1 - y2))
        elif type_ == 'diagonal_nonuniform':
            dmax = max(abs(x1 - x2), abs(y1 - y2))
            dmin = min(abs(x1 - x2), abs(y1 - y2))
            return 1.414*dmin + (dmax - dmin)
        elif type_ == 'zero':
            return 0
    def a_star_search(self, h_type='zero', visualize=False):
        frontier = PriorityQueue()       # The OPENLIST
        frontier.put(self.start, 0)      # PUT START IN THE OPENLIST
        parent = {}              # parent, {loc: parent}
        # g function dict, {loc: f(loc)}, CLOSED LIST BASICALLY
        g = {}
        parent[self.start] = None
        g[self.start] = 0
        self.h_type = h_type
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
        while not frontier.empty():
            current = frontier.get()  # update current to be the item with best priority

            if visualize:
                ax.plot(current[0], current[1], "xc")

            # early exit if we reached our goal
            if current == self.goal:
                break
            for next in self.graph.neighbors(current):
                g_next = g[current] + self.graph.cost(current, next)
                # if next location not in CLOSED LIST or its cost is less than before
                # Newer implementation
                if next not in g or g_next < g[next]:
                    g[next] = g_next
                    if self.h_type == 'zero':
                        priority = g_next 
                    else:
                        priority = g_next + self.heuristic(self.goal, next, self.h_type)
                    frontier.put(next, priority)
                    parent[next] = current

        if visualize:
            ax.plot(parent[0], parent[1], "-r")
            fig.show()

        self.parent = parent
        self.g = g
        return parent, g

    def breadth_first_search():
        print("WIP")

    def djikstra_search():
        print("WIP")
