import networkx as nx
import numpy as np
import tempfile
import os
import subprocess


ORCA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "orca.exe")


def get_degrees(graphs):
	"""
	Computes the degrees of nodes in graphs.
	Arguments:
		`graphs`: a list of N NetworkX undirected graphs
	Returns a list of N NumPy arrays (each of size V) of degrees, where each
	array is ordered by the nodes in the graph. Note that V can be different for
	each graph.
	"""
	result = []
	for g in graphs:
		degrees = nx.degree(g)
		result.append(np.array([deg for _, deg in degrees]))
	return result


def get_clustering_coefficients(graphs):
	"""
	Computes the clustering coefficients of nodes in graphs.
	Arguments:
		`graphs`: a list of N NetworkX undirected graphs
	Returns a list of N NumPy arrays (each of size V) of clustering
	coefficients, where each array is ordered by the nodes in the graph. Note
	that V can be different for each graph.
	"""
	result = []
	for g in graphs:
		coefs = nx.clustering(g)
		result.append(np.array([clust for _, clust in coefs.items()]))
	return result


def get_spectra(graphs):
	"""
	Computes the spectrum of graphs as the eigenvalues of the normalized
	Laplacian.
	Arguments:
		`graphs`: a list of N NetworkX undirected graphs
	Returns a list of N NumPy arrays (each of size V) of eigenvalues. Note that
	V can be different for each graph.
	"""
	return [nx.normalized_laplacian_spectrum(g) for g in graphs]


def run_orca(graph, max_graphlet_size=4):
	"""
	Runs Orca on a given graph to count the number of orbits of each type each
	node belongs in.
	Arguments:
		`graph`: a NetworkX undirected graph
		`max_graphlet_size`: maximum size of graphlets whose orbits to count;
			must be either 4 or 5
	Returns a V x O NumPy array of orbit counts for each of the V nodes.
	"""
	# Create temporary directory to do work in
	temp_dir_obj = tempfile.TemporaryDirectory()
	temp_dir = temp_dir_obj.name
	in_path = os.path.join(temp_dir, "graph.in")
	out_path = os.path.join(temp_dir, "orca.out")

	# Create input file
	with open(in_path, "w") as f:
		edges = graph.edges
		f.write("%d %d\n" % (len(graph), len(edges)))
		for edge in edges:
			f.write("%d %d\n" % edge)

	# Run Orca
	with open(os.devnull, "w") as f:
		subprocess.check_call([
			ORCA_PATH, "node", str(max_graphlet_size), in_path, out_path
		], stdout=f)

	# Read in result
	with open(out_path, "r") as f:
		result = [
			np.array(list(map(int, line.strip().split()))) for line in f
		]
		result = np.stack(result)

	temp_dir_obj.cleanup()

	return result


def get_orbit_counts(graphs, max_graphlet_size=4):
	"""
	Computes the orbit counts of nodes in graphs. The orbit counts of a node are
	the number of times it appears in each orbit of the possible graphlets of
	size up to `max_graphlet_size`. For example, for `max_graphlet_size` of 4,
	there are 15 orbits possible for a node.
	Orbits are computed using Orca:
	https://academic.oup.com/bioinformatics/article/30/4/559/205331
	Arguments:
		`graphs`: a list of N NetworkX undirected graphs
		`max_graphlet_size`: maximum size of graphlets whose orbits to count;
			must be either 4 or 5
	Returns a list of N NumPy arrays (of size V x O) of orbit counts, where each
	array is ordered by the nodes in the graph. Note that V can be different for
	each graph, but O is the same for the same `max_graphlet_size`).
	"""
	assert max_graphlet_size in (4, 5)
	return [run_orca(graph, max_graphlet_size) for graph in graphs]	


if __name__ == "__main__":
	graphs = [
		nx.erdos_renyi_graph(
			np.random.choice(np.arange(10, 20)),
			np.random.choice(np.linspace(0, 1, 10))
		)
		for _ in range(100)
	]

	degrees = get_degrees(graphs)
	cluster_coefs = get_clustering_coefficients(graphs)
	spectra = get_spectra(graphs)
	orbit_counts = get_orbit_counts(graphs)
	print(orbit_counts)