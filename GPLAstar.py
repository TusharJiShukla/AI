from networkx.algorithms.reciprocity import reciprocity
from networkx.algorithms.tree.mst import prim_mst_edges
import numpy as np
import sys
from ultis import PriorityQueueHeap, BestFirstSearch
from graph import graph
import networkx as nx
import time

class labels():
    
    def __init__(self,i,j,Ti,tj,V,cost,parent,sv_terminate=False,gv_wait=False):
        self.gv_pos = i
        self.sv_pos = j
        self.gv_time = Ti           #arrival time @ node i
        self.sv_time = tj           #arrival time @ node j
        self.V = V                  #dict with visited impeded edge as key and time as value
        self.cost = cost
        self.parent = parent 
        self.sv_terminate = sv_terminate
        self.gv_wait = gv_wait

class GPLAstar(labels):
    
    def __init__(self, graph, GV_start, GV_goal, SV_start, impeded_edges, upper_bound, UB_parents, lower_bound, run_time_limit, heuristic_type='original'):

        self.GV_start = GV_start
        self.GV_goal = GV_goal
        self.SV_start = SV_start
        self.Graph = graph
        self.impeded_edges = impeded_edges

        self.upper_bound = upper_bound
        self.heuristics = lower_bound
        self.heuristic_type = heuristic_type  # 'original' or 'topk' or 'custom'

        self.expanded_labels = 0
        self.run_time_limit = run_time_limit

        self.labels_dict = {}                       # label_dict with key (i,j) and value as a list of non dominated labels 
        self.labels_heap = PriorityQueueHeap()      # heap of all non-dominated labels queued based on f-value = g+h
        
        self.Final_path = None
        self.UB_cost = self.upper_bound[self.GV_start[0]]
        self.final_label = None
        self.final_cost = None
        self.rollout_path = None

        self.UB_parents = UB_parents
        
        # Statistics for comparison
        self.heuristic_computation_time = 0
        self.original_heuristic_calls = 0
        self.custom_heuristic_calls = 0

        #Top-k heuristic support & caches
        self.topk_k = 2
        try:
            self.dist_c, self.pred_c = nx.single_source_dijkstra(self.Graph, self.GV_goal[0], weight='unimpeded_cost')
        except Exception:
            self.dist_c = self.heuristics if self.heuristics is not None else {}
            self.pred_c = {}
        self.dist_s_all = {}
        self.heuristic_cache = {}
        
        self.algorithm()

    def compute_custom_heuristic(self, gv_pos, repaired_edges):
        """
        ADMISSIBLE CUSTOM HEURISTIC - Always optimistic but more informed
        Uses minimum possible cost considering which edges MIGHT be repaired
        """
        start_time = time.time()
        self.custom_heuristic_calls += 1
        
        try:
            # ✅ ADMISSIBLE APPROACH: Always use the most optimistic cost
            # For repaired edges, use unimpeded cost (best case)
            # For unrepaired edges, use the minimum of impeded vs potential repair cost
            
            def optimistic_cost(u, v, edge_data):
                edge_key = tuple(sorted((u, v)))
                
                # If already repaired, best case is unimpeded cost
                if edge_key in repaired_edges:
                    return edge_data['unimpeded_cost']
                
                # If not repaired yet, consider two possibilities:
                # 1. GV goes through impeded (impeded_cost)
                # 2. SV repairs it first (unimpeded_cost + some service travel)
                # We take the minimum for admissibility
                impeded = edge_data['impeded_cost']
                unimpeded = edge_data['unimpeded_cost']
                
                # Most optimistic: assume repair happens instantly at no extra cost
                # But to be safe, we can add a small penalty to ensure admissibility
                return min(impeded, unimpeded)
            
            # Compute shortest path with optimistic costs
            heuristic_value = nx.shortest_path_length(
                self.Graph, 
                source=gv_pos, 
                target=self.GV_goal[0],
                weight=optimistic_cost
            )
            
        except (nx.NetworkXNoPath, KeyError):
            # If no path exists, fall back to original heuristic
            heuristic_value = self.heuristics.get(gv_pos, float('inf'))
        
        # ✅ CRITICAL: Ensure heuristic is NEVER greater than true cost
        # Compare with original heuristic and take the minimum to ensure admissibility
        original_h = self.heuristics.get(gv_pos, float('inf'))
        final_heuristic = min(heuristic_value, original_h)
        
        self.heuristic_computation_time += (time.time() - start_time)
        return final_heuristic

    def compute_custom_topk(self, gv_pos, sv_pos, repaired_edges, k=None):
        """Top-k predictive heuristic (fast) - robust path reconstruction using nx.shortest_path."""
        start_time = time.time()

        if k is None:
            k = self.topk_k

        # canonicalize repaired edges set (ensure tuples)
        try:
            repaired_fset = frozenset(tuple(sorted(e)) for e in repaired_edges)
        except Exception:
            # if repaired_edges is not iterable or contains unexpected items, make empty set
            repaired_fset = frozenset()

        cache_key = (gv_pos, sv_pos, repaired_fset, k)
        if cache_key in self.heuristic_cache:
            return self.heuristic_cache[cache_key]

        # baseline unimpeded distance
        base = self.dist_c.get(gv_pos, float('inf'))
        if base == float('inf'):
            # if unreachable in unimpeded graph, fallback to original baseline if available
            h_val = self.heuristics.get(gv_pos, float('inf')) if self.heuristics is not None else float('inf')
            self.heuristic_cache[cache_key] = h_val
            self.heuristic_computation_time += (time.time() - start_time)
            return h_val

        # reconstruct unimpeded path nodes using NetworkX (robust)
        try:
            path_nodes = nx.shortest_path(self.Graph, source=gv_pos, target=self.GV_goal[0], weight='unimpeded_cost')
        except nx.NetworkXNoPath:
            path_nodes = []

        # convert consecutive node pairs to edge tuples (sorted ordering)
        path_edges = []
        for idx in range(len(path_nodes) - 1):
            u = path_nodes[idx]
            v = path_nodes[idx + 1]
            edge = (u, v) if u < v else (v, u)
            path_edges.append(edge)

        # collect up to k unrepaired impeded edges on that path
        unrepaired_impeded = []
        for e in path_edges:
            if e in self.impeded_edges and e not in repaired_fset:
                unrepaired_impeded.append(e)
                if len(unrepaired_impeded) >= k:
                    break

        if not unrepaired_impeded:
            h_val = base
            self.heuristic_cache[cache_key] = h_val
            self.heuristic_computation_time += (time.time() - start_time)
            return h_val

        # compute total optimistic saving (sum of Ti - Tu for those edges)
        total_saving = 0
        start_nodes = []
        for e in unrepaired_impeded:
            # ensure edge exists in Graph with named attributes
            total_saving += (self.Graph.edges[e]['impeded_cost'] - self.Graph.edges[e]['unimpeded_cost'])
            # choose as service target the node earlier on the path (path_nodes[idx])
            # we use the u endpoint (i.e., the node the convoy reaches first)
            start_nodes.append(e[0])

        # lazy compute service distances from sv_pos (and cache)
        if sv_pos not in self.dist_s_all:
            self.dist_s_all[sv_pos] = nx.single_source_dijkstra_path_length(self.Graph, sv_pos, weight='SV_cost')
        dist_map = self.dist_s_all[sv_pos]

        # optimistic service travel = minimum distance to any of start_nodes (admissible)
        service_travel = min(dist_map.get(u, float('inf')) for u in start_nodes)

        # predictive heuristic
        h_predictive = base - total_saving + service_travel
        if h_predictive < 0:
            h_predictive = 0

        self.heuristic_cache[cache_key] = h_predictive
        self.heuristic_computation_time += (time.time() - start_time)
        return h_predictive


    def heuristic(self, node, repaired_edges=None, sv_pos=None):
        """Unified heuristic function that supports original, topk, and custom variants."""
        if self.heuristic_type == 'custom':
            if repaired_edges is None:
                self.original_heuristic_calls += 1
                return self.heuristics.get(node, 0) if self.heuristics is not None else 0
            else:
                return self.compute_custom_heuristic(node, repaired_edges)

        elif self.heuristic_type == 'topk':
            if repaired_edges is None or sv_pos is None:
                self.original_heuristic_calls += 1
                return self.heuristics.get(node, 0) if self.heuristics is not None else 0
            else:
                h_predictive = self.compute_custom_topk(node, sv_pos, repaired_edges, k=self.topk_k)
                self.original_heuristic_calls += 1
                return max(self.heuristics.get(node, 0) if self.heuristics is not None else 0, h_predictive)
        else:
            self.original_heuristic_calls += 1
            return self.heuristics.get(node, 0) if self.heuristics is not None else 0

    def algorithm(self):
        '''Initialize lable dict'''
        
        gv_start_time = [0]*len(self.GV_start)
        sv_start_time = [0]*len(self.SV_start)
        start_label = labels(self.GV_start[0],self.SV_start[0],gv_start_time[0],sv_start_time[0],{},0,None)
        
        self.labels_dict[(self.GV_start[0], self.SV_start[0])] = [start_label]
        
        # Use appropriate heuristic for start label
        start_heuristic = self.heuristic(self.GV_start[0], start_label.V.keys(), sv_pos=self.SV_start[0])

        self.labels_heap.put(start_label, start_label.cost + start_heuristic, start_label.cost)      

        start_time = time.time()
        
        while True:

            try:
                _,curr_label = self.labels_heap.get()
            except:
                if len(self.labels_heap.elements)==0:
                    print("open set is empty")
                    break
                else:
                    print("Unknown error")

            self.expanded_labels +=1

            self.simulation(curr_label)

            if curr_label.cost + self.heuristics[curr_label.gv_pos] == self.UB_cost:
                print('fcost equal to upper bound cost')
                print('openlist len', len(self.labels_heap.elements))
                break 
                        
            if (time.time()-start_time) <= self.run_time_limit:
                self.REF(curr_label)
                
            else:
                print('Run out of time')
                print('Output best feasible solution found so far')
                break
        
        # Print heuristic statistics
        self.print_heuristic_stats()
        
        print('expanded labels in GPLAstar',self.expanded_labels)

        self.Final_path = self.final_path()
        self.final_cost = self.UB_cost

    def print_heuristic_stats(self):
        """Print statistics about heuristic usage"""
        print("\n=== HEURISTIC STATISTICS ===")
        print(f"Heuristic type: {self.heuristic_type}")
        print(f"Original heuristic calls: {self.original_heuristic_calls}")
        print(f"Custom heuristic calls: {self.custom_heuristic_calls}")
        print(f"Heuristic computation time: {self.heuristic_computation_time:.4f} seconds")
        if self.custom_heuristic_calls > 0:
            print(f"Average custom heuristic time: {self.heuristic_computation_time/self.custom_heuristic_calls:.6f} seconds per call")

    def simulation(self, label):
        '''
        This is used to generate feasible solution for the GV using the pre-computed shortest paths
        '''
        rollout_path = []       # path for vehicles from current postion to GV destition
        rollout_cost = 0    
        
        gv_pos = label.gv_pos

        while gv_pos != self.GV_goal[0]:
            
            new_pos = self.UB_parents[gv_pos]
            edge = (gv_pos, new_pos) if gv_pos<new_pos else (new_pos,gv_pos)
            
            if edge in label.V:
                wait_cost = max(0, label.V[edge] - label.gv_time - rollout_cost)
                rollout_cost += min(self.Graph.edges[edge]['impeded_cost'], wait_cost + self.Graph.edges[edge]['unimpeded_cost'])
            else:
                rollout_cost += self.Graph.edges[edge]['impeded_cost']
            
            gv_pos = new_pos
            rollout_path.append((new_pos,label.sv_pos,label.gv_time+rollout_cost,label.sv_time))

        total_sim_cost = label.cost + rollout_cost

        if total_sim_cost < self.UB_cost:
            self.UB_cost = total_sim_cost
            self.final_label = label
            self.rollout_path = rollout_path

    def REF(self, curr_label):
        
        gv_curr = curr_label.gv_pos
        sv_curr = curr_label.sv_pos

        gv_neighbors, sv_neighbors = self.motion_model(curr_label)
        
        new_labels = []

        for sv_next in sv_neighbors:
            
            V_ = curr_label.V.copy()
            sv_time = curr_label.sv_time                                         # Update the visited edge dict
            
            wait_label = None

            if sv_next != sv_curr:
                sv_time += self.Graph.edges[(sv_curr,sv_next)]['SV_cost']        # Time for sv to travel the edge
                sv_edge = (sv_curr,sv_next) if sv_curr < sv_next else (sv_next,sv_curr)                                   
                if sv_edge in self.impeded_edges:
                    if sv_edge not in V_ :
                        sv_time +=  self.Graph.edges[sv_edge]['service_cost']         
                        V_[sv_edge] = sv_time
              

            for gv_next in gv_neighbors:
                gv_time = curr_label.gv_time
                visited_edges = curr_label.V.keys()
                gv_edge = (gv_curr,gv_next) if gv_curr < gv_next else (gv_next,gv_curr)

                if gv_edge in self.impeded_edges :                                  #impeded edge check   
                         
                    if gv_edge in visited_edges :
                        visit_time = curr_label.V[gv_edge]

                        wait_time = max(0,visit_time-curr_label.gv_time)
                         # GV picks the lower of visited or unvisited costs
                        gv_edge_cost = min( self.Graph.edges[gv_edge]['impeded_cost'], wait_time + self.Graph.edges[gv_edge]['unimpeded_cost'])
                        gv_time += gv_edge_cost
                        cost = gv_time + sv_time

                        new_labels.append(labels(gv_next,sv_next,gv_time,sv_time,V_,cost,curr_label,sv_next==sv_curr))
                    
                    else:
                        '''create 2 labels'''
                        if sv_next!=sv_curr:                    # generate waiting label only if sv is not terminated
                            gv_time_wait = max(gv_time,sv_time)
                            cost = gv_time_wait + sv_time
                            wait_label = labels(gv_curr,sv_next,gv_time_wait,sv_time,V_,cost,curr_label,False,True)

                        if curr_label.gv_wait is False:
                            gv_time += self.Graph.edges[gv_edge]['impeded_cost']
                            cost = gv_time + sv_time
                            new_labels.append(labels(gv_next,sv_next,gv_time,sv_time,V_,cost,curr_label,sv_next==sv_curr))

                else:
                    if curr_label.gv_wait is False:
                        gv_time += self.Graph.edges[gv_edge]['unimpeded_cost']
                        cost = gv_time + sv_time
                        new_labels.append(labels(gv_next,sv_next,gv_time,sv_time,V_,cost,curr_label,sv_next==sv_curr))

            if wait_label!=None:
                new_labels.append(wait_label)

        self.update_labels(new_labels)

    def update_labels(self, new_labels):
        ''' Check for non Dominated labels 
            Add non dominated labels in label_heap with priority including heuristics
            Add upper bound filter and other filters
        '''

        for label in new_labels:
            # Use appropriate heuristic based on type
            if self.heuristic_type == 'custom':
                h_value = self.heuristic(label.gv_pos, label.V.keys())
            elif self.heuristic_type == 'topk':
                h_value = self.heuristic(label.gv_pos, label.V.keys(), sv_pos=label.sv_pos)
            else:
                h_value = self.heuristic(label.gv_pos)
                
            if label.cost + h_value >= self.UB_cost:        
                continue
         
            if (label.gv_pos,label.sv_pos) in self.labels_dict:
                if self.non_dominance(label):
                    self.labels_dict[(label.gv_pos,label.sv_pos)].append(label)
                    self.labels_heap.put(label, label.cost + h_value, label.cost)
            
            else:
                self.labels_dict[(label.gv_pos,label.sv_pos)] = [label]
                self.labels_heap.put(label, label.cost + h_value, label.cost)

    def non_dominance(self,new_label):
        '''check here if any existing label is able to dominate new_label, if it does then return False else True'''
                
        for label in self.labels_dict[(new_label.gv_pos,new_label.sv_pos)]:
            if label.gv_time <= new_label.gv_time and label.sv_time <= new_label.sv_time:
                if set(new_label.V.keys()).issubset(set(label.V.keys())):
                    flag = 0
                    for edge in new_label.V.keys():
                        if label.V[edge] > new_label.V[edge]:
                            flag = 1
                    if flag == 0:
                        return False

        return True 

    def final_path(self):
        if self.final_label != None:
            path = [(self.final_label.gv_pos, self.final_label.sv_pos, 
                    self.final_label.gv_time, self.final_label.sv_time)]
            
            parent_label = self.final_label.parent

            while parent_label != None:
                path.append((parent_label.gv_pos,parent_label.sv_pos, 
                            parent_label.gv_time, parent_label.sv_time)) 
                parent_label = parent_label.parent

            path.reverse()
            
            if self.rollout_path != None:
                path = path +self.rollout_path
                
            return path
        else:
            return None

    def motion_model(self,curr_label):
        '''get the current node and return list of GV and SV neighbors
        '''
        gv_neighbour = []
        sv_neighbour = []
        parent_label = curr_label.parent
        
        if parent_label == None or curr_label.sv_terminate == True:                    #for start label
            sv_neighbour = [curr_label.sv_pos]
        else:
            if set(curr_label.V.keys()) != set(parent_label.V.keys()):
                sv_neighbour = [curr_label.sv_pos]

        for node_ in self.Graph.neighbors(curr_label.gv_pos):
            if parent_label != None and parent_label.gv_pos == node_:
                continue
            gv_neighbour.append(node_)
        
        if curr_label.sv_terminate == False:
            for node_ in self.Graph.neighbors(curr_label.sv_pos):
                sv_neighbour.append(node_)
             
        return gv_neighbour, sv_neighbour