import networkx as nx
import pandas as pd
import numpy as np
import math
import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
import json

class MultiGraph(object):
	"""
	Gives functionality for time slices of a graph, mainly consisting of a dict of graphs with same nodes and edges, but different weights from slice to slice.
	df a pandas dataframe where cols = indices of time slice eg. dates and index = nodes of graph
	edges a list of tuples of nodes to be connected
	"""
	def __init__(self, df, edges):
		all_graphs = []
		for i, date in enumerate(df.columns):
			G = nx.Graph(date = date)
			G.add_nodes_from(list(df.index))
			G.add_edges_from(edges)
			G.remove_node("ærø")
			entries = {mun: df.loc[mun, date] for mun in df.index}
			nx.set_node_attributes(G, entries, name = "weight")


			all_graphs.append(G)
		self.graphs = all_graphs

	def get_graph_list(self):
		return self.graphs

	def compute_gradients(self):

		gradient_graphs = []
		for G in self.graphs:
			cases = nx.get_node_attributes(G, "weight") #Dict of {nodes: new cases}
			DG = nx.DiGraph(date = G.graph["date"])
			DG.add_nodes_from(G.nodes)
			for edge in G.edges: #Tuple of (edge from, edge to)
				diff = cases[edge[0]] - cases[edge[1]] #calculate difference in new cases
				if diff < 0:

					DG.add_edge(edge[0], edge[1], weight=diff)
				else:
					DG.add_edge(edge[1], edge[0], weight=diff)
			gradient_graphs.append(DG)

		return gradient_graphs



def get_neighbor_municipalties():
	"""
	Created in desperation of not being able to find data on which municipalties were neighbors
	Analyzes geographic data (XML/GML) of the danish mun. borders
	Returns dict with {mun.0: [(x0, y0), ( x1, y1), ...] ...} 
	(Actual graph statistics uses json file which includes manually inserted ferries and bridge borders)
	Don't look at it...
	"""
	tree = ET.parse("data/DK_AdministrativeUnit.gml")
	root = tree.getroot()
	mun_pos = {}
	for child1 in root.iter("{http://www.opengis.net/gml/3.2}featureMember"):
		flag = False
		for child in child1.iter("{http://inspire.ec.europa.eu/schemas/gn/4.0}text"):
		 	if "Region" not in child.text:
		 		if child.text != "Christiansø":
		 			#mun_pos.append(child.text)
		 			flag = True
		if flag:
			positions = []
			for pos in child1.iter("{http://www.opengis.net/gml/3.2}posList"):
				p = pos.text.split(" ")
				p = [float(e) for e in p]
				p = [(p[2*i], p[2*i+1]) for i, e in enumerate(p[::2])]

				positions += p
			
			mun_pos[child.text] = positions

	return mun_pos

def get_neighbors(mun_pos):
	"""Returns: a dict of {mun1 : [neigh1, neigh2, ...], ...}"""
	all_muns = {}
	for mun_name in mun_pos:
		neighbors = []
		for mun in mun_pos:
			if mun_name != mun:
				if set(mun_pos[mun_name]) & set(mun_pos[mun]):
					neighbors.append(mun)
		all_muns[mun_name] = neighbors
		print(mun_name)
	return all_muns

def get_mun_centers(mun_pos):
	"""Similar to get_neighbors, but for every dict key, returns a tuple of avg. pos. instead of list of all positions"""
	out_dict = {}
	for mun in mun_pos:
		position_pairs = mun_pos[mun]
		x, y = 0, 0
		for position_tuple in position_pairs:
			x += position_tuple[1]
			y+= position_tuple[0]
		x /= len(position_pairs)
		y /= len(position_pairs)
		out_dict[mun] = (x,y)
	return out_dict


def read_neighbor_data():
	"""Returns: dict where key=municipalty and item=neighbors"""	
	with open('data/nabokommuner_data.json', 'r') as outfile:
		data = json.load(outfile)
		data.pop("ærø", None)
		return data

def read_covid_data():
	"""Returns: Pandas Dataframe where columns=date and index=municipalties, inputs are daily covid infections"""
	df = pd.read_csv("data/Municipality_cases_time_series.csv", sep = ";").T
	df = df.drop("NA") # Clean
	df = df.rename(index={"Copenhagen": "København"})
	df.columns = list(df.loc["date_sample"])
	df = df.drop("date_sample")
	return df

def show_graph(G, labels = True):
	"""Shows matplotlib plot of network"""
	plt.figure(figsize = (16,9), dpi=120)
	c_dict = nx.get_node_attributes(G, "weight")
	c_list = [c_dict[k] for k in c_dict]

	arrow_dict = nx.get_edge_attributes(G, "weight")
	arrow_list = [arrow_dict[k] + 0.1 for k in arrow_dict]
	if labels:
		label_pos = mun_cs
		for p in label_pos:  # raise text positions
		    label_pos[p] = (mun_cs[p][0], mun_cs[p][1] + 7000)
		nx.draw_networkx_labels(G, label_pos)
	nc = nx.draw_networkx_nodes(G, pos = mun_cs, node_color = c_list, vmin=0,
		vmax=50, cmap=plt.cm.winter, label=G.graph["date"])
	nc = nx.draw_networkx_edges(G, pos = mun_cs, vmin=0,
		vmax=50, cmap=plt.cm.winter, label=G.graph["date"], width=arrow_list)
	plt.axis("off")
	plt.title(f"{G.graph['date']}")
	plt.savefig(f"out/{G.graph['date']}.png")
	plt.close()
	#plt.show()


df = read_covid_data() # Data frame
muns = df.index # List of municipalties

print(df)

#G = nx.Graph()
#G.add_nodes_from(list(muns))


n_dict = read_neighbor_data()

edge_tuples = [] # Add edges to graph
for mun in n_dict:
	for e in n_dict[mun]:
		edge_tuples.append((mun, e)) 
#G.add_edges_from(edge_tuples)
#G.remove_node("ærø") # Clean
mun_cs = get_mun_centers(get_neighbor_municipalties())





M = MultiGraph(df, edge_tuples)
GM = M.compute_gradients()
#for G in M.get_graph_list():
#	show_graph(G, labels=False)
for G in GM:
	show_graph(G, labels=False)	