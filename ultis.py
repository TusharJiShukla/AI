import heapq
from graph import graph
import numpy as np

class Search:
    
    total_expanded_nodes = 0

    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal

    def set_start(self, start):
        self.start = start

    def set_goal(self, goal):
        self.goal = goal

    @classmethod
    def update_expanded_nodes(cls):
        cls.total_expanded_nodes +=1

    @classmethod
    def reset(cls):
        """Reset all class variables """
        cls.total_expanded_nodes = 0

class BestFirstSearch(Search):
    """Standard A-Star or Dijkstra search on any generic graph, though heuristics are only defined for grid-based graphs atm"""
    def __init__(self, graph, start, bound=None, goal=None, heuristic_type='zero', depth_limit=None, repaired_edges=None):
        Search.__init__(self, graph, start, goal)
        self.heuristic_type = heuristic_type
        self.cost_type = 'unimpeded_cost' if bound == 'lower_bound' else 'impeded_cost'
        self.repaired_edges = repaired_edges if repaired_edges is not None else set()

        # A star initialize openList, closedList
        self.frontier = PriorityQueueHeap()
        self.frontier.put(self.start, 0, 0)      # PUT START IN THE OPENLIST
        self.parent = {}              # parent, {loc: parent}

        # g function dict, {loc: f(loc)}, CLOSED LIST BASICALLY
        self.g = {}
        self.parent[self.start] = None
        self.g[self.start] = 0

        #depth limit
        self.depth_limit = depth_limit

        self.boundary_nodes = []

    def heuristic(self, a, b, type_='manhattan'):
        """ Grid based heuristics """
        (x1, y1) = a
        (x2, y2) = b
        if type_ == 'manhattan':
            return abs(x1 - x2) + abs(y1 - y2)
        elif type_ == 'euclidean':
            v = [x2 - x1, y2 - y1]
            return np.hypot(v[0], v[1])
        elif type_ == 'diagonal_uniform':
            # Chebyshev Distance
            return max(abs(x1 - x2), abs(y1 - y2))
        elif type_ == 'diagonal_nonuniform':
            dmax = max(abs(x1 - x2), abs(y1 - y2))
            dmin = min(abs(x1 - x2), abs(y1 - y2))
            return 1.414*dmin + (dmax - dmin)

    def compute_custom_heuristic(self, current_node, repaired_edges):
        """Compute realistic heuristic considering repaired edges"""
        if self.goal is None:
            return 0
            
        # Create a temporary graph with updated edge costs based on repairs
        temp_graph = self.graph.copy()
        
        for edge in temp_graph.edges():
            edge_key = tuple(sorted(edge))
            if edge_key in repaired_edges:
                # If edge is repaired, use unimpeded cost
                temp_graph.edges[edge]['temp_cost'] = temp_graph.edges[edge]['unimpeded_cost']
            else:
                # If edge is not repaired, use impeded cost
                temp_graph.edges[edge]['temp_cost'] = temp_graph.edges[edge]['impeded_cost']
        
        # Run Dijkstra with the updated costs
        try:
            heuristic_value = nx.shortest_path_length(temp_graph, source=current_node, 
                                                    target=self.goal, weight='temp_cost')
            return heuristic_value
        except:
            # Fallback to original heuristic if path doesn't exist
            return self.heuristic(self.goal, current_node, 'euclidean')

    def use_algorithm(self, custom_heuristic_edges=None):
        """ Usage:
            - call to runs full algorithm until termination
            Returns:
            - a linked list, 'parent'
            - hash table of nodes and their associated min cost, 'g'
        """
        
        frontier = self.frontier
        parent = self.parent
        g = self.g

        while not frontier.empty():
            _, current = frontier.get()  # update current to be the item with best priority

            # BestFirstSearch.update_expanded_nodes()

            # early exit if we reached our goal
            if current == self.goal: 
                break
            # early exit if all of our goals in the closed set
            if self.goal is not None: 
                if len(set(self.goal).intersection(set(g)-set(frontier.elements))) == len(self.goal):
                    break

            # expand current node and check neighbors
            for next in self.graph.neighbors(current):
                g_next = g[current] + self.graph.edges[(current,next)][self.cost_type]
                # if next location not in CLOSED LIST or its cost is less than before
                # Newer implementation
                if next not in g or g_next < g[next]:
                    if self.heuristic_type == 'zero' or self.heuristic_type == None or self.goal == None:
                        priority = g_next 
                    elif self.heuristic_type == 'custom' and custom_heuristic_edges is not None:
                        # Use custom heuristic with repaired edges information
                        priority = g_next + self.compute_custom_heuristic(next, custom_heuristic_edges)
                    else:
                        priority = g_next + self.heuristic(self.goal, next, self.heuristic_type)
                    
                    if self.depth_limit is not None: 
                        if priority <= self.depth_limit:
                            g[next] = g_next
                            frontier.put(next, priority, priority)
                            parent[next] = current
                        else:
                            self.boundary_nodes.append(current)
                    else:
                        g[next] = g_next    
                        frontier.put(next, priority, priority)
                        parent[next] = current

        return parent, g

class PriorityQueueHeap:
    """ A min Priority Queue O(log(n)) with heapq library
    
    Modified to break ties based on put order

    References:
        Thanks RedBlobGames
    
    """

    def __init__(self):
        self.elements = []
        self.ecount = 0
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority, tie_break):
        """ tie-breaking is done based on ecount
        """
        # print(priority, -tie_break, item)
        heapq.heappush(self.elements, (priority, -tie_break, -self.ecount, item))
        
        self.ecount+=1

    def get(self):
        """ return the item with min priority value, specifically a tuple (value, item)
        """
        # return heapq.heappop(self.elements)
        # only return priority and item
        ret = heapq.heappop(self.elements)
        return (ret[0], ret[3])

    # Only necessary for removing redundant items
    def delete(self, item):
        for i in self.elements:
            if i[1] == item:
                self.elements.remove(i)
                break