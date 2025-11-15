import networkx as nx
import matplotlib.pyplot as plt
from networkx.generators.trees import prefix_tree
import numpy as np

from animationV2 import AnimateV2
from GPLAstar import GPLAstar
from centralizedAstar import centralizedAstar

import csv, time
from ultis import BestFirstSearch
from graph import graph

def main():
    planner()
    

class planner():
    def __init__(self):
        '''Define the instance'''
        self.generate_graph = graph(grid_size= [5,5], frac_imp=0.5, cuts = 0, unimp_cost_range = [10,16], 
                                imp_cost_range = [40,51], SV_cost_range = [1,2], service_cost_range = [1,6] )
        
        self.GV_start = self.generate_graph.GV_start
        self.GV_goal = self.generate_graph.GV_goal
        self.SV_start = self.generate_graph.SV_start
        self.Graph = self.generate_graph.G
        self.impeded_edges = self.generate_graph.impeded_edges
        self.perm_label_path ,self.lifted_graph_path = [], []

        self.colors = ['b','g','y','m','k','c']

        print('GV_start',self.GV_start)
        print('GV_goal',self.GV_goal)
        print('SV_start',self.SV_start)
        print('Impeded_edges', len(self.impeded_edges), self.impeded_edges)

        # precompute one-time bounds using Dijkstra (paper baseline)
        self.upper_bound = nx.shortest_path_length(self.Graph, source=self.GV_goal[0], target=None, weight='impeded_cost', method='dijkstra')
        self.lower_bound = nx.shortest_path_length(self.Graph, source=self.GV_goal[0], target=None, weight='unimpeded_cost', method='dijkstra')

        UB = self.upper_bound[self.GV_start[0]]
        LB = self.lower_bound[self.GV_start[0]]
        print('upper bound', self.upper_bound[self.GV_start[0]])
        print('lower bound', self.lower_bound[self.GV_start[0]])

        # precompute parents for UB rollout path (used by simulation rollouts)
        upper_bound_search = BestFirstSearch(self.Graph,self.GV_goal[0])
        self.UB_parents, _ = upper_bound_search.use_algorithm()

        frac_imp_edges = len(self.impeded_edges)/len(self.Graph.edges)
        time_limit = 900

        # Run with original heuristic
        print('\n' + '='*60)
        print('############ Starting GPLAstar with ORIGINAL heuristic ############# ')
        print('='*60)
        starttime = time.time()
        GPLAsim_original = GPLAstar(self.Graph,self.GV_start,self.GV_goal,self.SV_start,self.impeded_edges,
                                  self.upper_bound, self.UB_parents, self.lower_bound, time_limit, heuristic_type='original')
        GPLAsim_cost_original = GPLAsim_original.UB_cost 
        GPLAsim_time_original = time.time()-starttime 

        print('Vehicle Trajectory (gv_pos, sv_pos, gv_time, sv_time)',GPLAsim_original.Final_path)
        print('Run Time ', GPLAsim_time_original,  'Cost ', GPLAsim_cost_original)

        # Run with Top-k predictive heuristic
        print('\n' + '='*60)
        print('############ Starting GPLAstar with TOP-K predictive heuristic ############# ')
        print('='*60)
        starttime = time.time()
        GPLAsim_topk = GPLAstar(self.Graph,self.GV_start,self.GV_goal,self.SV_start,self.impeded_edges,
                                self.upper_bound, self.UB_parents, self.lower_bound, time_limit, heuristic_type='topk')
        GPLAsim_cost_topk = GPLAsim_topk.UB_cost 
        GPLAsim_time_topk = time.time()-starttime 

        print('Vehicle Trajectory (gv_pos, sv_pos, gv_time, sv_time)',GPLAsim_topk.Final_path)
        print('Run Time ', GPLAsim_time_topk,  'Cost ', GPLAsim_cost_topk)
        
        # ✅ ADDED: Run with custom heuristic
        print('\n' + '='*60)
        print('############ Starting GPLAstar with CUSTOM heuristic ############# ')
        print('='*60)
        starttime = time.time()
        GPLAsim_custom = GPLAstar(self.Graph,self.GV_start,self.GV_goal,self.SV_start,self.impeded_edges,
                                self.upper_bound, self.UB_parents, self.lower_bound, time_limit, heuristic_type='custom')
        GPLAsim_cost_custom = GPLAsim_custom.UB_cost 
        GPLAsim_time_custom = time.time()-starttime 

        print('Vehicle Trajectory (gv_pos, sv_pos, gv_time, sv_time)',GPLAsim_custom.Final_path)
        print('Run Time ', GPLAsim_time_custom,  'Cost ', GPLAsim_cost_custom)

        print('############ Starting Centralized A* algo ############# ')
        starttime = time.time()
        cen_algo = centralizedAstar(self.Graph,self.GV_start,self.GV_goal,self.SV_start,self.impeded_edges,self.upper_bound,self.lower_bound)
        cen_cost = cen_algo.Final_cost
        cen_time = time.time()-starttime
        
        print('Vehicle Trajectory (gv_pos, sv_pos, gv_time, sv_time)',cen_algo.Final_path)
        print('Run Time ', cen_time,  'Cost ', cen_cost)

        # ✅ UPDATED: Comparison Results with ALL algorithms including Centralized A*
        print('\n' + '='*140)
        print('####################### COMPREHENSIVE COMPARISON RESULTS #######################')
        print('='*140)
        print(f"{'Metric':<25} {'Original':<12} {'Top-K':<12} {'Custom':<12} {'Centralized A*':<15} {'Best Value':<12}")
        print(f"{'-'*25:<25} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12} {'-'*15:<15} {'-'*12:<12}")
        
        # Final Cost Comparison
        costs = {
            'Original': GPLAsim_cost_original,
            'Top-K': GPLAsim_cost_topk, 
            'Custom': GPLAsim_cost_custom,
            'Centralized A*': cen_cost
        }
        best_cost = min(costs.values())
        best_cost_algo = [k for k, v in costs.items() if v == best_cost][0]
        print(f"{'Final Cost':<25} {GPLAsim_cost_original:<12} {GPLAsim_cost_topk:<12} {GPLAsim_cost_custom:<12} {cen_cost:<15} {best_cost} ({best_cost_algo})")
        
        # Run Time Comparison  
        times = {
            'Original': GPLAsim_time_original,
            'Top-K': GPLAsim_time_topk,
            'Custom': GPLAsim_time_custom,
            'Centralized A*': cen_time
        }
        best_time = min(times.values())
        best_time_algo = [k for k, v in times.items() if v == best_time][0]
        print(f"{'Run Time (s)':<25} {GPLAsim_time_original:<12.4f} {GPLAsim_time_topk:<12.4f} {GPLAsim_time_custom:<12.4f} {cen_time:<15.4f} {best_time:.4f} ({best_time_algo})")
        
        # Expanded Labels Comparison
        labels_data = {
            'Original': GPLAsim_original.expanded_labels,
            'Top-K': GPLAsim_topk.expanded_labels,
            'Custom': GPLAsim_custom.expanded_labels,
            'Centralized A*': cen_algo.expanded_labels
        }
        best_labels = min(labels_data.values())
        best_labels_algo = [k for k, v in labels_data.items() if v == best_labels][0]
        print(f"{'Expanded Labels':<25} {GPLAsim_original.expanded_labels:<12} {GPLAsim_topk.expanded_labels:<12} {GPLAsim_custom.expanded_labels:<12} {cen_algo.expanded_labels:<15} {best_labels} ({best_labels_algo})")
        
        # Heuristic Calls (N/A for Centralized A*)
        print(f"{'Heuristic Calls':<25} {GPLAsim_original.original_heuristic_calls:<12} {GPLAsim_topk.original_heuristic_calls:<12} {GPLAsim_custom.original_heuristic_calls + GPLAsim_custom.custom_heuristic_calls:<12} {'N/A':<15} {'-'*12}")

        # ✅ UPDATED: Determine overall best performer
        print('\n' + '='*140)
        print('🏆 OVERALL PERFORMANCE ANALYSIS')
        print('='*140)
        
        # Cost performance (primary metric)
        if best_cost_algo == 'Custom':
            print("🎉 CUSTOM HEURISTIC FOUND THE OPTIMAL SOLUTION!")
        elif best_cost_algo == 'Top-K':
            print("🎉 TOP-K HEURISTIC FOUND THE OPTIMAL SOLUTION!") 
        elif best_cost_algo == 'Original':
            print("🎉 ORIGINAL HEURISTIC FOUND THE OPTIMAL SOLUTION!")
        else:
            print("🎉 CENTRALIZED A* FOUND THE OPTIMAL SOLUTION!")
        
        # Speed performance
        if best_time_algo == 'Custom':
            print("🚀 CUSTOM HEURISTIC WAS THE FASTEST")
        elif best_time_algo == 'Top-K':
            print("🚀 TOP-K HEURISTIC WAS THE FASTEST")
        elif best_time_algo == 'Original':
            print("🚀 ORIGINAL HEURISTIC WAS THE FASTEST")
        else:
            print("🚀 CENTRALIZED A* WAS THE FASTEST")
            
        # Efficiency performance (labels expanded)
        if best_labels_algo == 'Custom':
            print("💪 CUSTOM HEURISTIC WAS MOST EFFICIENT (fewest labels expanded)")
        elif best_labels_algo == 'Top-K':
            print("💪 TOP-K HEURISTIC WAS MOST EFFICIENT (fewest labels expanded)")
        elif best_labels_algo == 'Original':
            print("💪 ORIGINAL HEURISTIC WAS MOST EFFICIENT (fewest labels expanded)")
        else:
            print("💪 CENTRALIZED A* WAS MOST EFFICIENT (fewest labels expanded)")

        print('\n' + '='*140)
        
        # 🎬 ANIMATION TRIGGER - UPDATED WITH CUSTOM HEURISTIC
        self.run_animations(GPLAsim_original, GPLAsim_topk, GPLAsim_custom, cen_algo)

    def run_animations(self, GPLAsim_original, GPLAsim_topk, GPLAsim_custom, cen_algo):
        """Run animations for all algorithms"""
        print("\n" + "="*70)
        print("🎬 STARTING ANIMATIONS")
        print("="*70)
        
        # 1. ORIGINAL HEURISTIC ANIMATION
        if GPLAsim_original and GPLAsim_original.Final_path:
            print("\n📹 Animating ORIGINAL HEURISTIC...")
            self.animate_algorithm_path(GPLAsim_original.Final_path, "Original Heuristic")
        else:
            print("❌ No path found for Original Heuristic")
        
        # 2. TOP-K HEURISTIC ANIMATION
        if GPLAsim_topk and GPLAsim_topk.Final_path:
            print("\n📹 Animating TOP-K HEURISTIC...")  
            self.animate_algorithm_path(GPLAsim_topk.Final_path, "Top-K Heuristic")
        else:
            print("❌ No path found for Top-K Heuristic")
            
        # ✅ ADDED: CUSTOM HEURISTIC ANIMATION
        if GPLAsim_custom and GPLAsim_custom.Final_path:
            print("\n📹 Animating CUSTOM HEURISTIC...")
            self.animate_algorithm_path(GPLAsim_custom.Final_path, "Custom Heuristic")
        else:
            print("❌ No path found for Custom Heuristic")
            
        # 4. CENTRALIZED A* ANIMATION
        if cen_algo and cen_algo.Final_path:
            print("\n📹 Animating CENTRALIZED A*...")
            self.animate_algorithm_path(cen_algo.Final_path, "Centralized A*")
        else:
            print("❌ No path found for Centralized A*")

    def animate_algorithm_path(self, path, algorithm_name):
        """Animate a single algorithm's path using existing animate_motion"""
        print(f"   Algorithm: {algorithm_name}")
        print(f"   Path length: {len(path)} steps")
        print(f"   Final cost: {path[-1][2] if path else 'N/A'}")
        
        # Store the path for animation
        self.perm_label_path = path
        
        # Initialize plot
        self.initialize_plot()
        
        # Start animation
        print("   🚀 Starting animation... (Close window to continue)")
        self.animate_motion('perm_label')
        
        print("   ✅ Animation completed\n")

    def initialize_plot(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # ✅ SIMPLER APPROACH - Direct matplotlib plotting
        # Create mapping from node number to coordinates
        node_to_coord = {}
        for node in self.Graph.nodes:
            row = (node - 1) % 5
            col = 4 - ((node - 1) // 5)
            node_to_coord[node] = (row, col)
        
        # Plot edges
        for edge in self.Graph.edges:
            u, v = edge
            u_coord = node_to_coord[u]
            v_coord = node_to_coord[v]
            
            if edge in self.impeded_edges:
                ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 'r-', linewidth=2)
            else:
                ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 'k-', alpha=0.3)
        
        # Plot nodes
        for node, coord in node_to_coord.items():
            ax.plot(coord[0], coord[1], 'ko', markersize=8, alpha=0.5)
        
        # Plot start and goal
        start_coord = node_to_coord[self.GV_start[0]]
        goal_coord = node_to_coord[self.GV_goal[0]]
        sv_start_coord = node_to_coord[self.SV_start[0]]
        
        ax.plot(start_coord[0], start_coord[1], 'bs', markersize=15, label='GV Start')
        ax.plot(goal_coord[0], goal_coord[1], 'bo', markersize=15, label='GV Goal') 
        ax.plot(sv_start_coord[0], sv_start_coord[1], 'g^', markersize=15, label='SV Start')
        
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Path Planning Visualization')
        
        plt.show(block=False)
        plt.pause(0.1)
        
        # Simple AnimateV2 initialization
        AnimateV2.init_figure(fig, ax)

    def animate_motion(self, algo):
        """Simplified animation using direct matplotlib"""
        if algo != 'perm_label':
            return
            
        print("   🎬 Running simplified animation...")
        
        # Convert node numbers to coordinates
        node_to_coord = {}
        for node in self.Graph.nodes:
            row = (node - 1) % 5
            col = 4 - ((node - 1) // 5)
            node_to_coord[node] = (row, col)
        
        GV_path = [x[0] for x in self.perm_label_path]
        SV_path = [x[1] for x in self.perm_label_path]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot static elements
        for edge in self.Graph.edges:
            u, v = edge
            u_coord = node_to_coord[u]
            v_coord = node_to_coord[v]
            
            if edge in self.impeded_edges:
                ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 'r-', linewidth=2)
            else:
                ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 'k-', alpha=0.3)
        
        # Plot nodes
        for node, coord in node_to_coord.items():
            ax.plot(coord[0], coord[1], 'ko', markersize=8, alpha=0.5)
        
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Path Planning Animation')
        
        # Initialize vehicle markers
        gv_marker, = ax.plot([], [], 'bs', markersize=15, label='GV')
        sv_marker, = ax.plot([], [], 'g^', markersize=15, label='SV')
        gv_trail, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2)
        sv_trail, = ax.plot([], [], 'g-', alpha=0.5, linewidth=2)
        
        ax.legend()
        
        # Animation data
        gv_trail_x, gv_trail_y = [], []
        sv_trail_x, sv_trail_y = [], []
        
        # Animate step by step
        for i in range(len(GV_path)):
            gv_coord = node_to_coord[GV_path[i]]
            sv_coord = node_to_coord[SV_path[i]] if i < len(SV_path) else node_to_coord[SV_path[-1]]
            
            # Update markers
            gv_marker.set_data([gv_coord[0]], [gv_coord[1]])
            sv_marker.set_data([sv_coord[0]], [sv_coord[1]])
            
            # Update trails
            gv_trail_x.append(gv_coord[0])
            gv_trail_y.append(gv_coord[1])
            sv_trail_x.append(sv_coord[0])
            sv_trail_y.append(sv_coord[1])
            
            gv_trail.set_data(gv_trail_x, gv_trail_y)
            sv_trail.set_data(sv_trail_x, sv_trail_y)
            
            # Add step info
            ax.set_title(f'Path Planning Animation - Step {i+1}/{len(GV_path)}')
            
            plt.draw()
            plt.pause(0.5)  # Pause between steps
        
        print("   ✅ Animation completed")
        plt.show(block=True)

if __name__=='__main__':
    main()