import png
import pdb
import numpy as np
import collections
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import alphashape
import elkai
np.set_printoptions(suppress=True)
import copy #for deep copy purposes

import sys
np.set_printoptions(threshold=sys.maxsize)
def png_to_ogm(filename, normalized=False, origin='lower'):
    """
    Convert a png image to occupancy data.
    :param filename: the image filename
    :param normalized: whether the data should be normalised, i.e. to be in value range [0, 1]
    :param origin:
    :return:
    """
    r = png.Reader(filename)
    img = r.read()
    img_data = list(img[2])

    out_img = []
    bitdepth = img[3]['bitdepth']

    for i in range(len(img_data)):

        out_img_row = []

        for j in range(len(img_data[0])):
            if j % img[3]['planes'] == 0:
                if normalized:
                    out_img_row.append(img_data[i][j] * 1.0 / (2**bitdepth))
                else:
                    out_img_row.append(img_data[i][j])

        out_img.append(out_img_row)

    if origin == 'lower':
        out_img.reverse()

    return out_img

def tubular_path(path,l, grid_dim, grid_size):
    # returns all grids that are within some diagonal distance
    
    # Get all points in our grid
    meshlist = mesh_grid_list(grid_dim, grid_size)
    points = []
    # TODO Vectorize?
    for p in path:
        bool_list = (abs(p[0] - meshlist[:,0]) <= l)*(abs(p[1] - meshlist[:,1]) <= l)
        points.append(meshlist[bool_list])
        meshlist = np.delete(meshlist, bool_list, axis=0)
    points_st = np.vstack(points)
    points_uq = np.unique(points_st, axis=0) 
    return points_uq

''' Create a region of at least l thickness away around path p 
    tubular_path_shift: returns shifted path
    return_concave_hull: uses alphashape library
'''

def tubular_path_shift(path, l):
    # Shift all points up-left 
    #s = l/np.sqrt(2)
    s = np.ceil(l)
    up_left = path + np.array([-s, s])
    up_right = path + np.array([s, s])
    down_right = path + np.array([s, - s])
    down_left = path + np.array([-s,-s])
    stack = np.vstack((up_left, up_right, down_right, down_left))
    stack_uq = np.unique(stack, axis=0)
    return stack_uq

def return_convex_hull(points):
    hull = ConvexHull(points)
    return hull

def return_concave_hull(points):
    # use alphashape library
    #points=[(0., 0.), (0., 1.), (1., 1.), (1., 0.),
    #      (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)]
    #points = np.array([[10,10],[20,10],[10,20],[20,20]])
    #alpha = 0.50*alphashape.optimizealpha(points)
    hull = alphashape.alphashape(points, 0.05)   #tune this as needed
    hull_pts = hull.exterior.coords.xy
    pts = np.vstack((hull_pts[0], hull_pts[1])).T
    return pts

def filter_points(path, points, grid_dim):
    # for each point in 'points', find its closest point on path
    x = points[:,0]
    y = points[:,1]
    thresh = points[ (grid_dim[0] <= x)*(x <= grid_dim[1])* (grid_dim[2] <= y)*(y <= grid_dim[3])]
    return thresh

''' A class for adding regions in a plot
    grid_dim:  grid world dimensions [minX, maxX, minY, maxY]
    grid_size: grid resolution [m]
    add_region(verticies): specify ordered vertices of polygon
    returns: all the points lying within the polygonal region
'''
class Regions:
    def __init__(self, grid_dim, grid_size):
        self.grid_dim = grid_dim
        self.grid_size = grid_size
        # Get all the points in our grid
        self.points = mesh_grid_list(grid_dim, grid_size)
    
    def add_region(self, vertices):
        # vertices is a list of tuples (x,y)
        addR = self.Add(vertices, self.points)
        return addR.return_points()
    class Add:
        def __init__(self, vertices, points):
            # vertices is a list of tuples (x,y)
            self.add_poly_vertices(vertices, points)
        def add_poly_vertices(self, vertices, points):
            # vertices is a list of tuples (x,y)
            p = Path(vertices)
            grid_ind = p.contains_points(points)   #a bool of all points contained 
            self.poly_points = points[grid_ind]
        def return_points(self):
            return self.poly_points
         

def init_grid(grid_dim, grid_size, init_val):
    # Add 1 to even out world coordinates
    # Add np ceil to ensure actual grid size is not bigger than desired
    # since for some values, the grid dim wont divide evenly by grid size
    xvals = int(np.ceil((grid_dim[1] - grid_dim[0] + 1) / grid_size))
    yvals = int(np.ceil((grid_dim[3] - grid_dim[2] + 1) / grid_size))
    if init_val != 0:
        return init_val * np.ones((yvals, xvals))
    else:
        return np.zeros((yvals, xvals))

# A list of all the points in our grid
def mesh_grid_list(grid_dim, grid_size):
    # Make a mesh grid with the specified grid size/dim
    xvals = np.arange(grid_dim[0], grid_dim[1], grid_size)
    yvals = np.arange(grid_dim[2], grid_dim[3], grid_size)
    x, y = np.meshgrid(xvals, yvals)
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    return points

def get_world(indx, indy, grid_size, grid_dim):
    # Convert from index to world coordinates
    x = (indx) * grid_size + grid_dim[0]
    y = (indy) * grid_size + grid_dim[2]
    return (x, y)

def get_index(x, y, grid_size, grid_dim):
    # Convert from world coordinates to index
    #indx = int((x - grid_dim[0])/grid_size)
    #indy = int((y - grid_dim[2])/grid_size)
    # Make sure x,y fall within the grid physical boundaries
    if np.any((grid_dim[0] <= x) *
              (x <= grid_dim[1]) == 0) and np.any((grid_dim[2] <= y) *
                                                  (y <= grid_dim[3]) == 0):
        raise NameError('(x,y) world coordinates must be within boundaries')
    indx = ((x - grid_dim[0]) / grid_size).astype(int)
    indy = ((y - grid_dim[2]) / grid_size).astype(int)
    return (indx, indy)


# TODO: visualize class wip
class Animate:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        self.ax.set_xlim(xmin, xmax), self.ax.set_xticks([])
        self.ax.set_ylim(ymin, ymax), self.ax.set_yticks([])
        self.states = []
        self.time = []

    def updateIter(self, frame_number, state, time):
        self.states.append(states)
        self.time.append(time)
        self.ax.scatter(time[0:frame_number], states[0:frame_number])

    def update(self, frame_number, states, time):
        self.ax.scatter(time[0:frame_number], states[0:frame_number])

    def updateXYIter(self, frame_number, states):
        self.states.append(states)
        self.ax.scatter(states[0][0:frame_number], states[1][0:frame_number])

    def updateXY(self, frame_number, XY):
        self.ax.scatter(states[0][0:frame_number], states[1][0:frame_number])

    def animate(self, type_):
        if type_ == "TimeSeries":
            interval = len(self.time)
            return animation.FuncAnimation(
                self.fig,
                self.updateXY,
                init_func=[],
                interval=interval,
                blit=True)

# DEFINE GRAPH RELATED DATA STRUCTURES

class Queue:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()


class PriorityQueue:
    def __init__(self):
        self.elements = {}

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        self.elements[item] = priority

    def get(self):
        # Iterate through dictionary to find the item with the best priority
        best_item, best_priority = None, None
        for item, priority in self.elements.items():
            if best_priority is None or priority < best_priority:
                best_item, best_priority = item, priority

        # Remove the best item from the OPEN LIST
        del self.elements[best_item]

        # return
        return best_item


class SquareGrid:
    def __init__(self, grid, grid_dim, grid_size, type_=4):
        self.xwidth = grid_dim[1] - grid_dim[0]
        self.yheight = grid_dim[3] - grid_dim[2]
        self.grid = grid
        self.grid_dim = grid_dim
        self.grid_size = grid_size
        self.neighbor_type = type_

    def in_bounds(self, ind, type_='map'):
        if type_ == 'world':
            (x, y) = ind
            return self.grid_dim[0] <= x <= self.grid_dim[1] and self.grid_dim[2] <= y <= self.grid_dim[3]
        else:
            # grid indices
            (indx, indy) = ind
            return 0 <= indx <= self.xwidth and 0 <= indy <= self.yheight

    def not_obstacles(self, ind):
        (indx, indy) = ind
        return self.grid[indy, indx] == 0

    def neighbors(self, xxx_todo_changeme):
        # TODO: implement the 8 neighbors next time
        # Convert world coordinates to indices
        (x, y) = xxx_todo_changeme
        (indx, indy) = get_index(x, y, self.grid_size, self.grid_dim)
        if self.neighbor_type == 4:
            results = [(indx + 1, indy), (indx, indy - 1),
                       (indx - 1, indy), (indx, indy + 1)]
        elif self.neighbor_type == 8:
            results = [(indx + 1, indy), (indx, indy - 1),
                       (indx - 1, indy), (indx, indy + 1),
                       (indx + 1, indy + 1), (indx + 1, indy - 1),
                       (indx - 1, indy - 1), (indx - 1, indy + 1)]
        # Only return indices that are in range
        results = filter(self.in_bounds, results)

        # Only return results that are not obstacles
        results = filter(self.not_obstacles, results)
        # convert results to world coordinates
        results = map(
            lambda v: get_world(
                v[0],
                v[1],
                self.grid_size,
                self.grid_dim),
            results)
        return results

    # Cost of moving from one node to another (edge cost)
    def cost(self, from_node, to_node):
        a = from_node
        b = to_node
        v = (b[0] - a[0], b[1] - a[1])
        return np.hypot(v[0], v[1])


class MyGraph:
    def __init__(self, edge_dict):
        '''edge_dict: is dict containing all edges and weights
                      {('v1', 'v2'): 5.0}

		edges = [('v1', 'v2'), ('v1','v3')]
		weights = [0.5, 0.4]

		g = MyGraph()
        '''
        self.edge_dict = edge_dict 

    '''neighbors returns a list [] '''
    def neighbors(self, v):
        # Create a list of edges containing v
        edgeList = [key for key, val in self.edge_dict.items() if v in key]


        # Go through each tuple in test2 and select neighbors 
        neighs  = [node for edge in edgeList for node in edge if node != v]

        # remove duplicates by creating a dictionary
        neighs = list(dict.fromkeys(neighs))
        
        return neighs
    
    ''' cost returns edge weight '''
    def cost(self, from_node, to_node):
        pdb.set_trace
        weight = self.edge_dict[(from_node, to_node)]
        return weight

def reconstruct_path(parents, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = parents[current]
    path.append(start)
    path.reverse()
    return np.array(path)

# Coverage utilities

# lkh gives "optimal" order
def lkh_solve(array, dummy=None):
    # Introduce dummy node that's after the last node to visit 
    # The last node should consist of the very next one
    # Only return 0:len(r)-2. Dont return the dummy node and next visit
    try:
        val = (dummy != None).all()
    except:
        val = (dummy != None)
    # val is true if dummy node is desired
    if val:
        array = np.vstack((array, dummy))
    try:
        adj = np.zeros((len(array),len(array)))
        for i in range(len(array)):
            p1 = array[i]
            for j in range(len(array)):
                p2 = array[j]
                diff_ = (p1[0] - p2[0], p1[1] - p2[1])
                hyp = np.hypot(diff_[0], diff_[1])
                adj[i][j] = hyp
                # add dummy node that's far from every node except the ends
                if val:
                    if i==len(array)-1 or j==len(array)-1:
                        if j == 0 or i==0 or i==j or i==len(array)-2 or j==len(array)-2:
                            adj[i][j] = 0
                        else:
                            adj[i][j] =  1e9
                
        r_order = elkai.solve_float_matrix(adj, runs=10)
        # remove dummy and next node from the result
        if val:
            r_order.remove(len(array)-1)
            r_order.remove(len(array)-2)
            if len(r_order)>1:
                r_order.reverse()
                r_order.remove(0)
                r_order.insert(0,0)
            #rm_ind = np.array([len(array)-1, len(array)-2])
            #set_diff = np.setdiff1d(np.array(r_order), rm_ind)  #this just sorts,we dont want that! wrong!
    except:
        r_order = 0
    return r_order

##### We will implement a class that keeps track of our tree using a hash table
class MyTree:
    def __init__(self):
        self.tree = {}
        self.currentNode = ""
        self.lowestLevel = 0
    def create_node(self, parent, child, lower, upper, m):
        model = copy.deepcopy(m)
        ''' user inputs
            child:   a string for child's name, naming convention: P0, P1, P2,...,
            parent: a string for parent's name
            objective:  objective value from solver
            lower:  a lower bound on objective value    (rounded down sol)
            upper:  an upper bound on objective value   (LP relax)
            model:      a pyomo model object that contains all variablesa
            status:     '', pruned
                    prune by 'infeasible', 'bounds', 'optimal'
                    node name is given by 'child key'

            
            dict/hash_table varibles:
            layer:  int describing depth level each node belongs to 
            In general,always branch from thenode with the maximum upper bound
            If not feasible, then dont add create_node
        '''
        # child points to the parent
        # check parent to update level
        if parent in self.tree:
            parentLevel = self.tree[parent]['level']
            childLevel =  parentLevel + 1
        else:
            childLevel = 0
        self.tree.update({child: {'parent': parent, 'lower': lower, 'upper': upper, 'model': model, 'level': childLevel}})          
        self.currentNode = child 
          
        #update lowest level
        if childLevel > self.lowestLevel:
            self.lowestLevel = childLevel
          
    #def set_status(self, status, node):
    #    '''status:   prune by 'infeasible', 'bounds', 'optimal'
    #        node name is given by 'child key'

    #    '''
    #    temp = self.tree[node]
    #    temp['status'] = status
    #    self.tree.update({node: temp})

    def return_children(self, node):
        ''' return a child from given node
        '''

        # find all nodes with parent name given by "node"
        #children = []
        #for t in self.tree.keys():
        #    if node == self.tree[t]['parent']
        #       children.append(t)
        # list comprehension is nicer
        children = [t for t in self.tree.keys() if node == self.tree[t]['parent']]

        # TODO: we should break the search after finding two childrens...
        return children
    
    def return_model(self, node):
        m = self.tree[node]['model']
        return copy.deepcopy(m)  
    
    def delete_node(self, node):
        if node in self.tree:
            del self.tree[node]
        
    def find_maximum_upper_node(self):
        maxUpperB = -1000
        instance = ""
        for t in self.tree.keys():
            if self.tree[t]['upper'] > maxUpperB:
                maxUpperB = self.tree[t]['upper']
                instance = t
        return instance, maxUpperB
# Helper class to floor pyomo model values
# implemented only for Flow Variables
from pyomo.environ import Var
def pyomo_floor(m):
    #results = copy.deepcopy(r)
    model = copy.deepcopy(m)
    for v in model.component_data_objects(Var):
        #currently only looking at schedule
        if 'schedule' in v.name:    
            v.value = np.floor(v.value)
    # need to test to see if this is persistent?
    # YES IT IS NEED DEEP COPY
    # we will have two different models then
    #for v in results.solution[0]['Variable'].keys():
    #    #currently only looking at schedule
    #    if 'schedule' in v:    
    #        results.solution[0]['Variable'][v]['Value'] = np.floor(results.solution[0]['Variable'][v]['Value'])
    return model


def pyomo_nonInteger_var(model):
    #return all nonInteger variables
    arr = []
    for v in model.component_data_objects(Var):
        if 'schedule' in v.name:
            # check if current variable is integer
            if v.value - np.floor(v.value) > 0:
                temp_string=[]
                for k in v.index():
                     temp_string.append(k)
                arr.append(tuple(temp_string))
                #if v.domain == NonNegativeIntegers
    return arr[:]

