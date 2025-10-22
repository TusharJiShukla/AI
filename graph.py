import networkx as nx
import numpy as np
import random

from urllib3 import Retry



class graph():

    def __init__(self, grid_size, frac_imp, cuts, unimp_cost_range, imp_cost_range, SV_cost_range, service_cost_range):
        
            
        self.grid_size = grid_size
        self.cuts = cuts            # this can be an int indicating number of cuts or list of edges that are chosen by the user 
        m,n = grid_size
        self.impeded_edges = []
        

        self.G = nx.grid_2d_graph(m,n)
        self.inverse_mapping={}
        self.node_relabel()
        self.GV_start = [1]
        self.GV_goal = [m*n]
        self.SV_start = [np.random.randint(1,m*n+1)]

        if isinstance(self.cuts, int):
            self.random_cuts()
        else:
            self.impeded_edges = self.cuts
        
        self.add_obstacles() # if we need to obstacle nodes in the graph

        self.add_impeded_edges(frac_imp,unimp_cost_range ,imp_cost_range,SV_cost_range,service_cost_range)


        
    def GV_terminals(self):
        all_dist = nx.shortest_path_length(self.G, source=0, target=None, weight='None', method='dijkstra')
        # print(all_dist[0])
        max_dist_node = max(all_dist,key=all_dist.get)
        self.GV_start = [max_dist_node]
        all_dist = nx.shortest_path_length(self.G, source=max_dist_node, target=None, weight='None', method='dijkstra')
        self.GV_goal = [max(all_dist,key=all_dist.get)]
        # print(self.GV_start,self.GV_goal)
        # print(all_dist[self.GV_goal[0]])
        # input()

    def random_cuts(self):

        node_set = set(self.G.nodes)
        node_set.difference_update(self.GV_start+self.GV_goal)
        # print(node_set)

        def is_sep_set(n1,n2):
            n1, n2 =  self.inverse_mapping[n1],self.inverse_mapping[n2]
            
            v1, v2  = slope*n1[0]-n1[1]+const, slope*n2[0]-n2[1]+const

            if v1*v2<0:
                return True
            elif v1*v2 == 0:
                if v1>0 or v2>0:
                    return True
                else:
                    return False
            else:
                return False
        
        
        for _ in range(self.cuts):
            valid_cut = False
            while valid_cut is False:
                cos = 0
                while cos==0:
                    cos = random.uniform(-0.01,0.01)
                slope = random.uniform(0.99,1)/cos
                node_ = self.inverse_mapping[random.sample(node_set,1)[0]]
                # slope = 3
                # node_ = (1,0)
                const = -slope*node_[0] + node_[1]

                if is_sep_set(self.GV_start[0],self.GV_goal[0]):
                    valid_cut = True
    
            for edge in self.G.edges:
                if is_sep_set(edge[0],edge[1]):
                    self.impeded_edges.append(edge)
        
        # print(self.impeded_edges)
        # print(slope,node_)

    def pick_GV_start(self):
        '''This method can be used to randomly pick the GV start node'''
        node_set = set(self.G.nodes)
        return random.sample(node_set, 1)
    
    def pick_GV_goal(self):
        '''This method can be used to randomly pick the GV goal node'''
        node_set = set(self.G.nodes)
        return random.sample(node_set, 1)

    def pick_SV_start(self):
        '''This method can be used to randomly pick the SV start node'''
        node_set = set(self.G.nodes)
        return random.sample(node_set, 1)
    
    def add_impeded_edges(self, percent_impeded, unimp_cost_range ,imp_cost_range, SV_cost_range, service_cost_range):
        
        num_imp_edge = int(percent_impeded*len(self.G.edges))
        
        edges = list(self.G.edges)
        if self.cuts == 0:
            self.impeded_edges = random.sample( edges, k = num_imp_edge)
       
        self.impeded_edges = list(set(self.impeded_edges))
        unimpeded_edges = list(set(edges)-set(self.impeded_edges))

        for edge in self.impeded_edges:
            
            serv_cost = np.random.randint(service_cost_range[0],service_cost_range[1])
            imp_cost = np.random.randint(imp_cost_range[0] , imp_cost_range[1])
            unimp_cost = np.random.randint(unimp_cost_range[0] , unimp_cost_range[1])
            sv_cost = np.random.randint(SV_cost_range[0],SV_cost_range[1])
            
            ''' Need unimp_cost <= imp_cost-serv_cost '''

            self.G.add_edge(edge[0],edge[1],impeded_cost = imp_cost, unimpeded_cost = unimp_cost, SV_cost = sv_cost, service_cost = serv_cost, color = 'r')
            # self.G.add_edge(edge[0],edge[1],impeded_cost = 10, unimpeded_cost = 1, SV_cost = 1, service_cost = 0, color = 'r')

        for edge in unimpeded_edges:
            unimp_cost = np.random.randint(unimp_cost_range[0] , unimp_cost_range[1])
            imp_cost = unimp_cost
            sv_cost = np.random.randint(SV_cost_range[0],SV_cost_range[1])
            serv_cost = 0

            self.G.add_edge(edge[0],edge[1],impeded_cost = imp_cost, unimpeded_cost = unimp_cost, SV_cost = sv_cost, service_cost = serv_cost, color = 'k')
            # self.G.add_edge(edge[0],edge[1],impeded_cost = 10, unimpeded_cost = 10, SV_cost = 1, service_cost = 0, color = 'r')


    def add_obstacles(self, obstacle=None):
        if obstacle == None:
            remove_edges = []
            for node in self.G.nodes:
                if float(np.random.randint(11))/10 > 1:
                    self.G.nodes[node]['obs'] = 1
                    remove_edges.extend(self.G.edges(node)) 
                else:
                    self.G.nodes[node]['obs'] = 0    
            self.G.remove_edges_from(remove_edges)
        else:
            remove_edges = []
            for node in self.G.nodes:
                if node in obstacle:
                    self.G.nodes[node]['obs'] = 1
                    remove_edges.extend(self.G.edges(node))
                else:
                    self.G.nodes[node]['obs'] = 0
            self.G.remove_edges_from(remove_edges)


    def node_relabel(self):
        
        m,n = self.grid_size
        
        mapping = {}
        self.inverse_mapping = {}
        
        for node in self.G.nodes:
            label = node[0]+1+node[1]*m
            mapping[node] = label
            self.inverse_mapping[label] = node 
        
        self.G = nx.relabel_nodes(self.G,mapping)


