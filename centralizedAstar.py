from networkx.algorithms.reciprocity import reciprocity
from networkx.algorithms.tree.mst import prim_mst_edges
import numpy as np
import sys
from ultis import PriorityQueueHeap
from graph import graph
import time

# ... rest of the centralizedAstar.py code remains the same ...
# Just remove the incorrect import line at the top

# def main():
#     perm_label_algo()


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



class centralizedAstar(labels):
    
    def __init__(self, graph, GV_start, GV_goal, SV_start, impeded_edges, upper_bound, lower_bound):

        self.GV_start = GV_start
        self.GV_goal = GV_goal
        self.SV_start = SV_start
        self.Graph = graph
        self.impeded_edges = impeded_edges

        self.upper_bound = upper_bound
        self.heuristics = lower_bound

        self.expanded_labels = 0

        self.labels_dict = {}                       # label_dict with key (i,j) and value as a list of non dominated labels 
        self.labels_heap = PriorityQueueHeap()      # heap of all non-dominated labels queued based on f-value = g+h
        self.GV_label_heap = PriorityQueueHeap()

        self.Final_path = None
        self.Final_cost = None
        self.final_label = None
        self.algorithm()
        # print(self.upper_bound[(0,0)],self.upper_bound[(1,0)],self.upper_bound[(0,1)],self.upper_bound[(1,1)])
    
    
    def algorithm(self):
        '''Initialize lable dict'''
        
        gv_start_time = [0]*len(self.GV_start)
        sv_start_time = [0]*len(self.SV_start)
        start_label = labels(self.GV_start[0],self.SV_start[0],gv_start_time[0],sv_start_time[0],{},0,None)
        
        self.labels_dict[tuple(self.GV_start+self.SV_start)] = [start_label]
        
        self.labels_heap.put(start_label,start_label.cost + self.heuristic(self.GV_start[0]), start_label.cost)      

        start_time = time.time()

        while (time.time()-start_time) <= 1800:

            try:
                _,curr_label = self.labels_heap.get()
            except:
                if len(self.labels_heap.elements)==0:
                    print("open set is empty")
                    break
                else:
                    print("Unknown error")
            
            if curr_label.gv_pos == self.GV_goal[0]:
                print('Reached Goal')
                if self.final_label == None or curr_label.cost < self.Final_cost :
                    self.final_label = curr_label
                    self.Final_cost = curr_label.cost  
                break
            
            
            self.REF(curr_label, vehicle=None)
            self.expanded_labels += 1                                                                                                                                                                                                                                                                                                                                                           



        print('expanded labels in Centralized A*', self.expanded_labels)

        self.Final_path = self.final_path()
            


            
    def REF(self, curr_label, vehicle):
        
        gv_curr = curr_label.gv_pos
        sv_curr = curr_label.sv_pos

        gv_neighbors, sv_neighbors = self.motion_model(curr_label, vehicle)
        # print('neighbors',gv_neighbors,sv_neighbors)
        
        new_labels = []

        ############## SV extensions ###############################
        gv_time = curr_label.gv_time

        for sv_next in sv_neighbors:
            V_ = curr_label.V.copy()
            sv_time = curr_label.sv_time                                         # Update the visited edge dict
            
            if sv_next != sv_curr:
                sv_time += self.Graph.edges[(sv_curr,sv_next)]['SV_cost']        # Time for sv to travel the edge
                sv_edge = (sv_curr,sv_next) if sv_curr < sv_next else (sv_next,sv_curr)                                   
                if sv_edge in self.impeded_edges:
                    if sv_edge not in V_ :
                        sv_time +=  self.Graph.edges[sv_edge]['service_cost']         
                        V_[sv_edge] = sv_time
            
            cost = gv_time + sv_time
            new_labels.append(labels(gv_curr,sv_next,gv_time,sv_time,V_,cost,curr_label,sv_next==sv_curr))
        
        ############## GV extensions ###############################
        sv_time = curr_label.sv_time
        sv_term = curr_label.sv_terminate
        V_ = curr_label.V.copy()

        for gv_next in gv_neighbors:
            gv_time = curr_label.gv_time
            visited_edges = curr_label.V.keys()
            gv_edge = (gv_curr,gv_next) if gv_curr < gv_next else (gv_next,gv_curr)

            if gv_edge in self.impeded_edges :                                  #impeded edge check   
                    
                if gv_edge in visited_edges :
                    visit_time = curr_label.V[gv_edge]

                    wait_time = max(0,visit_time-curr_label.gv_time)
                        # GV picks the lower of visited or unvisited costs
                    gv_edge_cost = min(self.Graph.edges[gv_edge]['impeded_cost'], self.Graph.edges[gv_edge]['unimpeded_cost'] + wait_time)
                    gv_time += gv_edge_cost
                    cost = gv_time + sv_time

                    new_labels.append(labels(gv_next,sv_curr,gv_time,sv_time,V_,cost,curr_label,sv_term))
                
                else:
                    if sv_term is False:
                        gv_time_wait = max(gv_time,sv_time)
                        cost = gv_time_wait + sv_time
                        new_labels.append(labels(gv_curr,sv_curr,gv_time_wait,sv_time,V_,cost,curr_label,False,True))
                    
                    if curr_label.gv_wait is False:
                        gv_time += self.Graph.edges[gv_edge]['impeded_cost']
                        cost = gv_time + sv_time
                        new_labels.append(labels(gv_next,sv_curr,gv_time,sv_time,V_,cost,curr_label,sv_term))
                        # print('new_label ',gv_next,sv_next,gv_time,sv_time,V_,cost,curr_label,sv_next==sv_curr)

            else:
                if curr_label.gv_wait is False:
                    gv_time += self.Graph.edges[gv_edge]['impeded_cost']
                    cost = gv_time + sv_time
                    new_labels.append(labels(gv_next,sv_curr,gv_time,sv_time,V_,cost,curr_label,sv_term))
                      

        # input()
        self.update_labels(new_labels, vehicle)




    def update_labels(self,new_labels,vehicle):
        ''' Check for non Dominated labels 
            Add non dominated labels in label_heap with priority including heuristics
            Add upper bound filter and other filters
        '''

        for label in new_labels:
            if label.cost + self.heuristic(label.gv_pos) > self.upper_bound[self.GV_start[0]]:        
                continue

            if (label.gv_pos,label.sv_pos) in self.labels_dict:
                if self.non_dominance(label):
                    self.labels_dict[(label.gv_pos,label.sv_pos)].append(label)
                    self.labels_heap.put(label, label.cost + self.heuristic(label.gv_pos),label.cost)
                          
            else:
                self.labels_dict[(label.gv_pos,label.sv_pos)] = [label]
                self.labels_heap.put(label, label.cost + self.heuristic(label.gv_pos),label.cost)
        
                    
    
    def heuristic(self, node):
        if self.heuristics == None:
            return 0
        else:
            return self.heuristics[node]


        
            
    def non_dominance(self,new_label):
        '''check here if any label is able to dominate new_label, if it does then return False else True'''
                
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

            return path
        else:
            return None


    def motion_model(self,curr_label, vehicle):
        '''get the current node and return list of GV and SV neighbors
        '''
        gv_neighbour, sv_neighbour = [], []
        parent_label = curr_label.parent
        


        if parent_label == None or curr_label.sv_terminate == True:                    #for start label
            sv_neighbour = [curr_label.sv_pos]
        else:
            if set(curr_label.V.keys()) != set(parent_label.V.keys()):
                sv_neighbour = [curr_label.sv_pos]
        
        if curr_label.sv_terminate == False:
            for node_ in self.Graph.neighbors(curr_label.sv_pos):
                # if self.Graph.nodes[node_]['obs']==0 :
                sv_neighbour.append(node_)


        for node_ in self.Graph.neighbors(curr_label.gv_pos):
            # if self.Graph.nodes[node_]['obs']==0 :
            gv_neighbour.append(node_)
        
        
             
    
        return gv_neighbour, sv_neighbour

