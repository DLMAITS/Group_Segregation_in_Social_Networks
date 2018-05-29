# Install all dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import random
import math as math

# Install network.py
from network import *

# Main function
def main():

    test = PSNG(generate_three_party_list(7, "normal"), 0.25)

    test.dynamic_network_formation()

    print(nx.degree_centrality(test.social_network))

    print(nx.average_clustering(test.social_network))

    print(nx.graph_number_of_cliques(test.social_network))

    colour_map = []
    for i in test.social_network:
        if (test.social_network.node[i]['attr'][0] < 0.5):
            colour_map.append('blue')
        elif (test.social_network.node[i]['attr'][0] > 0.5):
            colour_map.append('green')
        else:
            colour_map.append('orange')

    nx.draw(test.social_network, node_color=colour_map, alpha=0.9, node_size=500, with_labels=True)
    plt.show()

# Test
if __name__ == "__main__":
    main()
