import matplotlib.pyplot as plt
import networkx as nx
import collections

label_mapping = {0: 's', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 't'}


class GraphVisualization:
    def __init__(self, flow_network):
        self.flow_network = flow_network
        self.G = nx.DiGraph()
        self.create_graph()  # Create the graph structure during initialization

    def create_graph(self):
        # Add nodes
        for node in label_mapping.values():
            self.G.add_node(node)

        # Add edges with the user-defined capacities
        for (u, v), capacity in self.flow_network.edges.items():
            from_node = label_mapping[u]
            to_node = label_mapping[v]
            if (v, u) not in self.flow_network.edges:
                self.G.add_edge(from_node, to_node, capacity=capacity, flow=0)

    def visualize(self):
        # Manually set positions for the vertical layout
        pos = {'s': (0, 1), 'a': (1, 2), 'b': (2, 2),
               'c': (1, 0), 'd': (2, 0), 't': (3, 1)}

        # Draw the network with specified positions
        nx.draw(self.G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrowsize=20)
        edge_labels = {(u, v): f"C: {self.G[u][v]['capacity']}\nF: {self.G[u][v]['flow']}" for u, v in self.G.edges()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color='red')

        # Display the plot
        plt.title('Network Flow Visualization')
        plt.axis('off')  # Turn off the axis
        plt.show()

    def update_flows(self, flows):
        # Clear any previous flows for accuracy
        for u, v in self.G.edges():
            self.G[u][v]['flow'] = 0

        # Update the flow on the edges with new values
        for (u, v), flow in flows.items():
            if self.G.has_edge(u, v):
                self.G[u][v]['flow'] = flow
        # Call visualize to redraw the graph with updated flows
        self.visualize()


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0] * vertices for _ in range(vertices)]
        self.edges = {}

    def add_edge(self, u, v, w):
        if v not in self.edges.get(u, []):
            self.graph[u][v] = w
            self.edges[(u, v)] = w

    def bfs(self, source, sink, parent):
        visited = [False] * self.V
        queue = collections.deque()
        queue.append(source)
        visited[source] = True

        while queue:
            u = queue.popleft()

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == sink:
                        return True
        return False

    def ford_fulkerson(self, source, sink):
        parent = [-1] * self.V
        max_flow = 0
        flows = {}  # Use a dictionary to keep track of the flow for each edge

        while self.bfs(source, sink, parent):
            path_flow = float('Inf')
            # Find minimum residual capacity of the edges along the path filled by BFS
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add flow to overall flow
            max_flow += path_flow
            # Update residual capacities of the edges and reverse edges along the path
            s = sink
            while s != source:
                u = parent[s]
                self.graph[u][s] -= path_flow
                self.graph[s][u] += path_flow
                # Update flow for edge in flows dictionary
                if (u, s) in flows:
                    flows[(u, s)] += path_flow
                else:
                    flows[(u, s)] = path_flow
                s = parent[s]

        # Convert flows to the format with node labels instead of numbers
        labeled_flows = {(label_mapping[u], label_mapping[v]): flow for (u, v), flow in flows.items()}
        return max_flow, labeled_flows


def get_edge_capacity(edge_description):
    while True:
        try:
            capacity = int(input(f"Enter the capacity for {edge_description}: "))
            if capacity < 0:
                print("Capacity cannot be negative. Please enter a valid positive integer.")
                continue
            return capacity
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

graph = Graph(6)  # Includes s, a, b, c, d, t

# User-defined capacities
edges = {
    ('s', 'a'): get_edge_capacity("edge s->a"),
    ('s', 'c'): get_edge_capacity("edge s->c"),
    ('a', 'b'): get_edge_capacity("edge a->b"),
    ('c', 'a'): get_edge_capacity("edge c->a"),
    ('b', 't'): get_edge_capacity("edge b->t"),
    ('c', 'd'): get_edge_capacity("edge c->d"),
    ('d', 'b'): get_edge_capacity("edge d->b"),
    ('d', 't'): get_edge_capacity("edge d->t")
}

# Edges are mapped to integers for the Graph class
nodes = {'s': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 't': 5}

# Add user-defined edges to the graph
for (u, v), capacity in edges.items():
    graph.add_edge(nodes[u], nodes[v], capacity)

# Set source and sink
source, sink = nodes['s'], nodes['t']

# Run Ford-Fulkerson algorithm
max_flow, flows = graph.ford_fulkerson(source, sink)
print(f"The maximum possible flow is {max_flow}")

# Print flows for debugging
print("Flows after running Ford-Fulkerson:")
for (u, v), flow in flows.items():
    print(f"Flow from {u} to {v}: {flow}")

# Visualization
graph_visualization = GraphVisualization(graph)
graph_visualization.update_flows(flows)