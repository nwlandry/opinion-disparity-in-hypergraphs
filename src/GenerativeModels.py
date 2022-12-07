import itertools
import operator
from functools import reduce

import numpy as np


def uniform_SBM_hypergraph(n, m, p, sizes):
    """Create a uniform SBM hypergraph."""
    # Check if dimensions match
    if len(sizes) != len(p):
        raise Exception("'sizes' and 'p' do not match.")
    if len(np.shape(p)) != m:
        raise Exception("The dimension of p does not match m")
    # Check that p has the same length over every dimension.
    if len(set(np.shape(p))) != 1:
        raise Exception("'p' must be a square tensor.")
    if np.max(p) > 1 or np.min(p) < 0:
        raise Exception("Entries of 'p' not in [0,1].")
    if np.sum(sizes) != n:
        raise Exception("Sum of sizes does not match n")

    node_labels = range(n)

    block_range = range(len(sizes))
    block_iter = itertools.product(block_range, repeat=m)
    # Split node labels in a partition (list of sets).
    size_cumsum = [sum(sizes[0:x]) for x in range(0, len(sizes) + 1)]
    partition = [
        list(node_labels[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(0, len(size_cumsum) - 1)
    ]

    edgelist = list()
    for block in block_iter:
        if p[block] == 1:  # Test edges cases p_ij = 0 or 1
            edges = itertools.product((partition[i] for i in block_range))
            for e in edges:
                edgelist.append(list(e))
        elif p[block] > 0:
            partition_sizes = [len(partition[i]) for i in block]

            max_index = reduce(operator.mul, partition_sizes, 1)
            if max_index < 0:
                raise Exception("Index overflow error!")
            index = np.random.geometric(p[block]) - 1

            while index < max_index:
                indices = index_to_edge_partition(index, partition_sizes, m)
                e = [partition[block[i]][indices[i]] for i in range(m)]
                if len(e) == len(set(e)):
                    edgelist.append(e)
                index += np.random.geometric(p[block])
    return edgelist


def uniform_planted_partition_hypergraph(n, m, k, epsilon):
    sizes = [int(n / 2), n - int(n / 2)]

    p = k / (m * n ** (m - 1))
    delta = epsilon * k / (m * n ** (m - 1))
    p_in = p + (2 ** (m - 1) - 1) * delta
    p_out = p - delta

    p = p_out * np.ones([2] * m)
    p[tuple([0] * m)] = p_in
    p[tuple([1] * m)] = p_in

    return uniform_SBM_hypergraph(n, m, p, sizes)


def uniform_random_partition_hypergraph(n, m, g, k_in, k_out):
    sizes = [int(n / g) for i in range(g - 1)]
    sizes.append(n - (g - 1) * int(n / g))

    # N/m<k_in> = m|E|= m p(N/m)^m
    p_in = k_in / (m * (n / g) ** (m - 1))
    # N<k_out> = m|E| = m p (N^m - g(N/m)^m)
    p_out = n * k_out / (m * (n**m - (n / g) ** m))

    p = p_out * np.ones([g] * m)
    for i in range(g):
        p[tuple([i] * m)] = p_in
    return uniform_SBM_hypergraph(n, m, p, sizes)


def uniform_erdos_renyi_hypergraph(n, m, k):
    p = k / (m * n ** (m - 1))
    edgelist = list()
    index = np.random.geometric(p) - 1  # -1 b/c zero indexing
    max_index = n**m
    while index < max_index:
        e = index_to_edge(index, n, m)
        if len(e) == len(set(e)):
            edgelist.append(e)
        index += np.random.geometric(p)
    return edgelist


# https://stackoverflow.com/questions/53834707/element-at-index-in-itertools-product
def index_to_edge(index, n, m):
    return [(index // (n**r) % n) for r in range(m - 1, -1, -1)]


def index_to_edge_partition(index, partition_sizes, m):
    try:
        return [
            int(index // np.prod(partition_sizes[r + 1 :]) % partition_sizes[r])
            for r in range(m)
        ]
    except:
        raise Exception("Invalid parameters")


def get_omega(edgelist, m, communities):
    omega = np.zeros(tuple([len(np.unique(communities))] * m))
    for edge in edgelist:
        omega[tuple(communities[list(edge)])] += 1
    return omega
