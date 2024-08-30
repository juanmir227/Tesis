import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.watts_strogatz_graph(13500, 5, 0.5)
print(G)
temp = list(G.degree())

degree_list = []
for element in temp:
    degree_list.append(element[1])
print(degree_list)
plt.hist(degree_list)
plt.show()