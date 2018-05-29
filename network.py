# Install all dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import random
import math as math

# --------------- FUNCTIONS TO GENERATE NODE LISTS ON DEMAND ------------------

# Generate a node list (default utility function, default type distribution, default tolerance distribution)
def generate_node_list(node_number, uf="norm", type_dist="uniform", tol_dist="uniform"):

    # Initialise count and node_list to store results
    count = 0
    node_list = []

    # Create the node_list
    while (count < node_number):

        temp = []

        # Add value for the node's type
        if (type_dist == "uniform"):
            temp.append(uniform_dist_generator())

        elif (type_dist == "normal"):
            temp.append(normal_dist_generator())

        elif (type_dist == "beta"):
            temp.append(beta_dist_generator())

        elif (type_dist == "exp"):
            temp.append(exp_dist_generator())


        # Add value for the node's tolerance
        if (tol_dist == "uniform"):
            temp.append(uniform_dist_generator())

        elif (tol_dist == "normal"):
            temp.append(normal_dist_generator())

        elif (tol_dist == "beta"):
            temp.append(beta_dist_generator())

        elif (tol_dist == "exp"):
            temp.append(exp_dist_generator())

        # Add the utility function type
        temp.append(uf)

        # Append the temp to the node_list
        node_list.append(temp)

        # Increment the count
        count += 1

    return node_list

# Helper function to return a value from the Uniform distribution
def uniform_dist_generator():
    return np.random.uniform()

# Helper function to return a value from the Normal distribution
def normal_dist_generator():
    return np.random.normal()

# Helper function to return a value from the Beta distribution (both shape parameters under 1)
def beta_dist_generator():
    return np.random.beta(0.5, 0.5)

# Helper function to return a value from the Exponential distribution
def exp_dist_generator():
    return np.random.exponential()

# Helper function to create a bifurcated two party system
def generate_two_party_list(nodes_per_group, tol_dist="uniform", uf="norm"):

    # Initialise variables
    party1_list = []
    party2_list = []
    count = 0

    # Loop
    while (count < nodes_per_group):

        # Temp variables
        temp1 = []
        temp2 = []

        # Add the types
        temp1.append(0)
        temp2.append(1)

        # Add the tolerance
        if (tol_dist == "uniform"):
            temp1.append(uniform_dist_generator())
            temp2.append(uniform_dist_generator())

        elif (tol_dist == "normal"):
            temp1.append(normal_dist_generator())
            temp2.append(normal_dist_generator())

        elif (tol_dist == "beta"):
            temp1.append(beta_dist_generator())
            temp2.append(beta_dist_generator())

        elif (tol_dist == "exp"):
            temp1.append(exp_dist_generator())
            temp2.append(exp_dist_generator())

        # Append to temp
        temp1.append(uf)
        temp2.append(uf)

        # Add to total list
        party1_list.append(temp1)
        party2_list.append(temp2)

        # Increment
        count += 1

    # Add all to node_list
    node_list = party1_list + party2_list

    return node_list


# Helper function to generate a three party system
def generate_three_party_list(nodes_per_group, tol_dist="uniform", uf="norm"):

    # Initialise variables
    party1_list = []
    party2_list = []
    party3_list = []
    count = 0

    # Loop
    while (count < nodes_per_group):

        # Temp variables
        temp1 = []
        temp2 = []
        temp3 = []

        # Add the types
        temp1.append(0)
        temp2.append(1)
        temp3.append(0.5)

        # Add the tolerance
        if (tol_dist == "uniform"):
            temp1.append(uniform_dist_generator())
            temp2.append(uniform_dist_generator())
            temp3.append(uniform_dist_generator())

        elif (tol_dist == "normal"):
            temp1.append(normal_dist_generator())
            temp2.append(normal_dist_generator())
            temp3.append(normal_dist_generator())

        elif (tol_dist == "beta"):
            temp1.append(beta_dist_generator())
            temp2.append(beta_dist_generator())
            temp3.append(beta_dist_generator())

        elif (tol_dist == "exp"):
            temp1.append(exp_dist_generator())
            temp2.append(exp_dist_generator())
            temp3.append(exp_dist_generator())

        # Append to temp
        temp1.append(uf)
        temp2.append(uf)
        temp3.append(uf)

        # Add to total list
        party1_list.append(temp1)
        party2_list.append(temp2)
        party3_list.append(temp3)

        # Increment
        count += 1

    # Add all to node_list
    node_list = party1_list + party2_list + party3_list

    return node_list

# --------------- CLASS FUNCTIONS FOR THE NETWORK FORMATION ------------------
class PSNG(object):

    # Initialisation
    def __init__(self, node_attr_list, cost_shift):

        # Initialise the variables
        self.node_attr_list = node_attr_list
        self.number_of_nodes = len(node_attr_list)
        self.social_network = nx.Graph()
        self.g_tree = nx.DiGraph()
        self.iter_count = 0
        self.current_gtree_node = 0
        self.decay = 0.9
        self.uf_at_index = 2
        self.type_at_index = 0
        self.tol_at_index = 1
        self.cost_shift = cost_shift

        # Utility function constant variable names
        self.uf_decl_norm = "norm"

        # Fill the social_network with the nodes plus their attributes
        self.add_nodes_to()

        # Add the current state of the network to the g_tree
        self.adj_matrix = self.update_adj_matrix()
        self.g_tree.add_node(self.current_gtree_node, adj_matrix=self.adj_matrix)

        # Need to store an array of all the edges that need to be visited
        self.to_visit = self.reinitialise_to_visit_array()


    # Function to dynamically create a pairwise stable function, or identify a closed cycle if one does not exist
    def dynamic_network_formation(self):

        '''
        # Current counter to break loop during development (DEVELOPMENT)
        counter = 0
        '''

        # Loop until either find a cycle in the g_tree, or we have visited all potential edges (i.e. pairwise stable network)
        while ((len(list(nx.simple_cycles(self.g_tree))) == 0) and (len(self.to_visit) != 0)):

            # Choose an element randomly from the 'to_visit' array
            edge_to_assess = random.choice(self.to_visit)

            # Remove the element from the 'to_visit' array
            self.to_visit.remove(edge_to_assess)

            # Get the two current nodes under consideration
            first_node = edge_to_assess[0]
            second_node = edge_to_assess[1]

            # Check if the edge under consideration is already linked or not
            is_edge_filled = self.adj_matrix.item((first_node, second_node))

            # print(first_node, second_node)

            # Branching decision 1: if edge does not exist
            if (is_edge_filled == 0):

                # Call method to check whether edge should be added or not (if true, then edge is added already)
                if (self.check_to_add_edge(first_node, second_node) == False):

                    # Remove the link if the addition is not beneficial to both parties - i.e. status quo
                    self.social_network.remove_edge(first_node, second_node)

                # Otherwise, update the adjacency matrix, and g_tree
                else:

                    # Update the adjacency matrix
                    self.adj_matrix = self.update_adj_matrix()

                    # Check whether g_tree needs to be updated
                    self.update_g_tree(self.adj_matrix)

                    # Reinitialise the to_visit array
                    self.to_visit = self.reinitialise_to_visit_array()

            # If the edge does already exist
            elif (is_edge_filled == 1):

                # Call method to check whether edge should be removed or not (if true, then edge is removed already)
                if (self.check_to_remove_edge(first_node, second_node) == False):

                    # Re-add the link if the removal is not beneficial for either party - i.e. status quo
                    self.social_network.add_edge(first_node, second_node)

                else:

                    # Update the adjacency matrix
                    self.adj_matrix = self.update_adj_matrix()

                    # Check whether g_tree needs to be updated
                    self.update_g_tree(self.adj_matrix)

                    # Reinitialise the to_visit array
                    self.to_visit = self.reinitialise_to_visit_array()

            '''
            # Increment the counter at each loop and break at max (DEVELOPMENT)
            counter = counter + 1

            if (counter == 100):
                print(first_node)
                print(second_node)
                break
            '''

        if (len(list(nx.simple_cycles(self.g_tree))) != 0):
            print("Cycle found")


    # --------------- SOCIAL NETWORK HELPER FUNCTIONS ----------------

    # Function to take a node_attr_list and add the nodes with their attributes to the social_network
    def add_nodes_to(self):
        for node in range(0, len(self.node_attr_list)):
            self.social_network.add_node(node, attr=self.node_attr_list[node])

    # Function to reinstantiate the 'to_visit' array to include the max of all edges that could exist in a complete graph
    def reinitialise_to_visit_array(self):

        # Retrieve all current edges and non-existent edges
        edges = nx.edges(self.social_network)
        non_edges = nx.non_edges(self.social_network)
        return list(edges) + list(non_edges)

    # Helper function to add an edge to the social network graph (and test the adjacency matrix)
    def add_edge_to_social_network(self, node1, node2):
        self.social_network.add_edge(node1, node2)
        self.adj_matrix = self.update_adj_matrix()

    # Helper function to update the state of the adjacency matrix
    def update_adj_matrix(self):
        return nx.adjacency_matrix(self.social_network, nodelist=range(self.number_of_nodes)).todense()

    # Helper function to print the current state of the adjacency matrix
    def print_adj_matrix(self):
        print(self.adj_matrix)

    # Helper function to check if edge should be added to graph
    def check_to_add_edge(self, node1, node2):

        # Calculate the current utilities of the two nodes
        node1_utility_prior = self.calculate_total_utility(node1)
        node2_utility_prior = self.calculate_total_utility(node2)

        # Add the edge to the graph
        self.social_network.add_edge(node1, node2)

        # Recalculate the utilities
        node1_utility_after = self.calculate_total_utility(node1)
        node2_utility_after = self.calculate_total_utility(node2)

        # print(node1_utility_prior, node1_utility_after, "and", node2_utility_prior, node2_utility_after)

        # If both of these utilities post the additional edge have increased, return True; otherwise False
        if ((node1_utility_after >= node1_utility_prior) and (node2_utility_after >= node2_utility_prior)):
            return True

        return False

    # Helper function to check if edge should be removed to graph
    def check_to_remove_edge(self, node1, node2):

        # Calculate the current utilities of the two nodes
        node1_utility_prior = self.calculate_total_utility(node1)
        node2_utility_prior = self.calculate_total_utility(node2)

        # Remove the edge from the graph
        self.social_network.remove_edge(node1, node2)

        # Recalculate the utilities
        node1_utility_after = self.calculate_total_utility(node1)
        node2_utility_after = self.calculate_total_utility(node2)

        # print(node1_utility_prior, node1_utility_after, "and", node2_utility_prior, node2_utility_after)

        # If either of these utilities is strictly higher after the removal than before, then return True
        if ((node1_utility_after > node1_utility_prior) or (node2_utility_after > node2_utility_prior)):
            return True

        return False

    # Helper function to calculate utility for a node from the graph structure
    def calculate_total_utility(self, node):

        # Get a dict of all reachable nodes from source, by shortest path length
        reachable_nodes = nx.single_source_shortest_path_length(self.social_network, node)

        # Store value of total utility
        total_utility = 0

        # Loop through the dict, calculate sub utilities with other reachable nodes, and decay at path length
        for dest in reachable_nodes.keys():
            total_utility += (self.decay ** reachable_nodes[dest]) * self.calculate_node_to_node_utility(node, dest)

            '''
            if (reachable_nodes[dest] < 5):
                total_utility += (self.decay ** reachable_nodes[dest]) * self.calculate_node_to_node_utility(node, dest)
            else:
                total_utility += 0
            '''

        # Return the total_utility
        return total_utility


    # Helper function to calculate the node-to-node utility, as a switchboard to the preferred utility function type
    def calculate_node_to_node_utility(self, src, dest):

        # Extract node attribute from source node
        uf_type = self.social_network.node[src]['attr'][self.uf_at_index]

        # Switchboard for the relevant utility function
        if (uf_type == self.uf_decl_norm):
            return self.uf_norm(src, dest)


    # --------------- UTILITY FUNCTIONS ----------------

    # Utility function implementation for the adjusted Normal distribution function
    def uf_norm(self, src, dest):

        # Get the attributes from src and dest
        src_type = self.social_network.node[src]['attr'][self.type_at_index]
        src_tol = self.social_network.node[src]['attr'][self.tol_at_index]
        dest_type = self.social_network.node[dest]['attr'][self.type_at_index]

        # Create the function
        constant = 1 / math.sqrt(2 * math.pi)

        try:
            exponent = math.exp(-0.5*(dest_type - src_type)*(dest_type - src_type)/(src_tol*src_tol))
        except:
            exponent = 0

        return 1*exponent - self.cost_shift


    # --------------- G-TREE HELPER FUNCTIONS ----------------

    # Helper function to update the g_tree
    def update_g_tree(self, adj_matrix):

        # Search for whether the g_tree already has recorded this state of the adjacency matrix
        found, node_already_found = self.does_node_already_exist(adj_matrix)

        # Temporarily store a reference to the current node in the g_tree
        src_node = self.current_gtree_node

        # If not found, then add a new node to the g_tree with the current updated adj_matrix
        if (found == False):
            self.iter_count += 1
            self.g_tree.add_node(self.iter_count, adj_matrix=self.adj_matrix)
            self.current_gtree_node = self.iter_count
            self.g_tree.add_edge(src_node, self.current_gtree_node)

        # Otherwise the node exists, so add an edge to the network, and reset the current_gtree_node
        elif (found == True):
            self.g_tree.add_edge(src_node, node_already_found)
            self.current_gtree_node = node_already_found

    # Helper function to find whether a node in the g_tree already has the same adjacency matrix as the current iteration
    def does_node_already_exist(self, adj_matrix):

        # Search for nodes in g_tree that have the same attributes as current state of the adjacency matrix
        shared_nodes = [node for node in self.g_tree.nodes(data=True) if (node[1]["adj_matrix"] == adj_matrix).all()]

        # test = [node for node in self.g_tree.nodes(data=True)]

        # Generate a boolean based on the above
        found = any(shared_nodes)

        # If found is False, then we can return None for node_already_found
        if (found == False):
            node_already_found = None

            # node_already_found = test[0][0]

            return found, node_already_found

        # Otherwise collect the node (which must be unique) number -> either way does not matter as algorithm will soon terminate
        node_already_found = shared_nodes[0][0]

        # Return the finding
        return found, node_already_found

    # Helper function to print g_tree
    def print_g_tree_path(self):
        print(self.g_tree.nodes())

    # Helper function to print current node details for latest node in the gtree
    def print_latest_g_tree_node(self):
        print(self.g_tree.node[self.current_gtree_node])

    # Helper function to add a new node to the g_tree, with an input adjacency matrix
    def add_new_node_g_tree(self, node_number, adj_matrix):
        self.g_tree.add_node(node_number, adj_matrix=adj_matrix)

    # Helper function to print all the edges from the g_tree
    def print_g_tree_all_edges(self):
        print(self.g_tree.edges())

    # Helper function to add an edge to a g_tree
    def g_tree_add_edge(self, src, dest):
        self.g_tree.add_edge(src, dest)
