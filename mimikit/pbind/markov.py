import numpy as np
from .pbind import patify, patvalue, Pattern, embedInStream, EOP, inf


class MSM:

    def __init__(self, definition=None):
        self.nodes = dict()
        # edges are of form: [targets, weights]
        self.edges = dict()
        if definition:
            self.add_nodes_with_edges(definition)

    def add_node(self, node, node_data):
        self.nodes[node] = node_data

    def add_edge(self, origin, target, weight=1.0):
        edges = self.edges.get(origin, None) or [[], np.array([])]
        edges[0].append(target)
        edges[1] = np.append(edges[1], [weight])
        self.edges[origin] = edges

    def add_nodes_with_edges(self, nodes):
        for n in nodes:
            if len(n) < 3:
                self.add_node(n[0], n[1])
            elif len(n) < 4:
                self.add_node(n[0], n[1])
                for edge in n[2]:
                    self.add_edge(n[0], edge)
            else:
                self.add_node(n[0], n[1])
                for target, weight in zip(n[2], n[3]):
                    self.add_edge(n[0], target, weight)

    def next_state(self, current_state):
        edges = self.edges.get(current_state, None)
        if not edges:
            return None
        return np.random.choice(edges[0], p=edges[1])

    def normalize_weights(self):
        for k, v in self.edges.items():
            v[1] = np.asarray(v[1]) / np.sum(v[1])
        return self


class Pmsm(Pattern):

    def __init__(self, msm, initial_state=0, steps=inf):
        self.initial_state = initial_state
        self.msm = msm.normalize_weights()
        self.steps = steps

    def embedInStream(self, rout):
        steps = patvalue(self.steps)
        counter = 0
        state = patvalue(self.initial_state)

        while counter < steps:
            pat = self.msm.nodes.get(state, None)
            print(pat)
            if pat is None:
                yield EOP
            yield embedInStream(rout, pat)
            state = self.msm.next_state(state)
            print('state', state)
            if state is None:
                yield EOP
            counter += 1
        yield EOP
