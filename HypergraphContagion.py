import multiprocessing as mp
import random
from collections import Counter, defaultdict

import numpy as np
import xgi


def get_polarization_in_parallel(arglist, numProcesses):
    with mp.Pool(processes=numProcesses) as pool:
        polarization = pool.starmap(get_polarization, arglist)
    return polarization


def get_polarization(
    filename,
    gamma,
    beta,
    community1,
    community2,
    tmax,
    fraction_to_average,
    num_sims,
    isVerbose,
):
    H = xgi.read_hypergraph_json(filename)

    polarization = 0
    for sim in range(num_sims):
        t, _, I1, _, I2 = Gillespie_SIS_two_communities(
            H,
            beta,
            gamma,
            community1,
            community2,
            transmission_function=collective_contagion,
            tmin=0,
            tmax=tmax,
        )
        polarization += abs(
            get_fixed_point(
                t,
                I1 / len(community1) - I2 / len(community2),
                time_to_average=fraction_to_average * (np.max(t) - np.min(t)),
            )
            / num_sims
        )

    if isVerbose:
        print(
            polarization,
            flush=True,
        )
    return polarization


def get_fixed_point(time, data, time_to_average=None):
    n = np.size(time, axis=0)
    if time_to_average is not None:
        cutoff = time[-1] - time_to_average
        i = n - 1
        while cutoff < time[i] and i > 0:
            i -= 1
        # interpolate
        if i > -np.size(time, axis=0):
            first_data_point = np.interp(
                cutoff, [time[i], time[i + 1]], [data[i], data[i + 1]]
            )
            time = np.concatenate([np.array([cutoff]), time[i + 1 :]])
            data = np.concatenate([np.array([first_data_point]), data[i + 1 :]])
    try:
        return np.average(
            data,
            weights=[(np.max(time) - np.min(time)) / np.size(time, axis=0)]
            + list(time[1:] - time[:-1]),
        )
    except:
        return np.mean(data)


#######################
#                     #
#   Auxiliary stuff   #
#                     #
#######################

# built-in functions
def collective_contagion(node, status, edge):
    """Collective contagion function.

    Parameters
    ----------
    node : hashable
        node ID
    status : dict
        keys are node IDs and values are their statuses.
    edge : iterable
        hyperedge

    Returns
    -------
    int
        0 if no transmission can occur, 1 if it can.
    """
    for i in set(edge).difference({node}):
        if status[i] != "I":
            return 0
    return 1


class SamplingDict(object):
    def __init__(self, weighted=False):
        self.item_to_position = {}
        self.items = []

        self.weighted = weighted
        if self.weighted:
            self.weight = defaultdict(int)  # presume all weights positive
            self.max_weight = 0
            self._total_weight = 0
            self.max_weight_count = 0

    def __len__(self):
        return len(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def _update_max_weight(self):
        C = Counter(
            self.weight.values()
        )  # may be a faster way to do this, we only need to count the max.
        self.max_weight = max(C.keys())
        self.max_weight_count = C[self.max_weight]

    def insert(self, item, weight=None):
        r"""
        If not present, then inserts the thing (with weight if appropriate)
        if already there, replaces the weight unless weight is 0

        If weight is 0, then it removes the item and doesn't replace.

        WARNING:
            replaces weight if already present, does not increment weight.


        """
        if self.__contains__(item):
            self.remove(item)
        if weight != 0:
            self.update(item, weight_increment=weight)

    def update(self, item, weight_increment=None):
        r"""
        If not present, then inserts the thing (with weight if appropriate)
        if already there, increments weight

        WARNING:
            increments weight if already present, cannot overwrite weight.
        """
        if (
            weight_increment is not None
        ):  # will break if passing a weight to unweighted case
            if weight_increment > 0 or self.weight[item] != self.max_weight:
                self.weight[item] = self.weight[item] + weight_increment
                self._total_weight += weight_increment
                if self.weight[item] > self.max_weight:
                    self.max_weight_count = 1
                    self.max_weight = self.weight[item]
                elif self.weight[item] == self.max_weight:
                    self.max_weight_count += 1
            else:  # it's a negative increment and was at max
                self.max_weight_count -= 1
                self.weight[item] = self.weight[item] + weight_increment
                self._total_weight += weight_increment
                self.max_weight_count -= 1
                if self.max_weight_count == 0:
                    self._update_max_weight
        elif self.weighted:
            raise Exception("if weighted, must assign weight_increment")

        if item in self:  # we've already got it, do nothing else
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove(self, choice):
        position = self.item_to_position.pop(
            choice
        )  # why don't we pop off the last item and put it in the choice index?
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

        if self.weighted:
            weight = self.weight.pop(choice)
            self._total_weight -= weight
            if weight == self.max_weight:
                self.max_weight_count -= 1
                if self.max_weight_count == 0 and len(self) > 0:
                    self._update_max_weight()

    def choose_random(self):
        if self.weighted:
            while True:
                choice = random.choice(self.items)
                if random.random() < self.weight[choice] / self.max_weight:
                    break
            return choice

        else:
            return random.choice(self.items)

    def random_removal(self):
        r"""uses other class methods to choose and then remove a random node"""
        choice = self.choose_random()
        self.remove(choice)
        return choice

    def total_weight(self):
        if self.weighted:
            return self._total_weight
        else:
            return len(self)

    def update_total_weight(self):
        self._total_weight = sum(self.weight[item] for item in self.items)


##########################
#                        #
#    SIMULATION CODE     #
#                        #
##########################

"""
    The code in the region below is used for stochastic simulation of
    epidemics on networks
"""


def Gillespie_SIS_two_communities(
    H,
    tau,
    gamma,
    community1,
    community2,
    transmission_function=collective_contagion,
    initial_infecteds=None,
    rho=None,
    tmin=0,
    tmax=100,
    recovery_weight=None,
    transmission_weight=None,
    **args
):
    """Simulates the SIS model for hypergraphs with the Gillespie algorithm.

    Parameters
    ----------
    H : xgi.Hypergraph
        The hypergraph on which to simulate the SIR contagion process
    tau : dict
        Keys are edge sizes and values are transmission rates
    gamma : float
        Healing rate
    transmission_function : lambda function, default: threshold
        The contagion function that determines whether transmission is possible.
    initial_infecteds : iterable, default: None
        Initially infected node IDs.
    initial_recovereds : iterable, default: None
        Initially recovered node IDs.
    recovery_weight : hashable, default: None
        Hypergraph node attribute that weights the healing rate.
    transmission_weight : hashable, default: None
        Hypergraph edge attribute that weights the transmission rate.
    rho : float, default: None
        Fraction initially infected. Cannot be specified if
        `initial_infecteds` is defined.
    tmin : float, default: 0
        Time at which the simulation starts.
    tmax : float, default: float("Inf")
        Time at which the simulation terminates if there are still
        infected nodes.

    Returns
    -------
    tuple of np.arrays
        t, S, I

    Raises
    ------
    HyperContagionError
        If the user specifies both rho and initial_infecteds.
    """

    if rho is not None and initial_infecteds is not None:
        raise Exception("cannot define both initial_infecteds and rho")

    if transmission_weight is not None:

        def edgeweight(item):
            return item[transmission_weight]

    else:

        def edgeweight(item):
            return None

    if recovery_weight is not None:

        def nodeweight(u):
            return H.nodes[u][recovery_weight]

    else:

        def nodeweight(u):
            return None

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.num_nodes * rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)

    initial_infecteds = list(community1)
    I1 = [len(initial_infecteds)]
    S1 = [len(community1) - I1[0]]
    I2 = [0]
    S2 = [len(community2) - I2[0]]
    times = [tmin]

    t = tmin

    members = H.edges.members(dtype=dict)
    memberships = H.nodes.memberships()

    status = defaultdict(lambda: "S")
    for node in initial_infecteds:
        status[node] = "I"

    if recovery_weight is None:
        infecteds = SamplingDict()
    else:
        infecteds = SamplingDict(weighted=True)

    unique_edge_sizes = xgi.unique_edge_sizes(H)

    IS_links = dict()
    for size in unique_edge_sizes:
        if transmission_weight is None:
            IS_links[size] = SamplingDict()
        else:
            IS_links[size] = SamplingDict(weighted=True)

    for node in initial_infecteds:
        infecteds.update(node, weight_increment=nodeweight(node))
        for edge_id in memberships[
            node
        ]:  # must have this in a separate loop after assigning status of node
            # handle weighted vs. unweighted?
            edge = members[edge_id]
            for nbr in edge:  # there may be self-loops so account for this later
                if status[nbr] == "S":
                    contagion = transmission_function(nbr, status, edge, **args)
                    if contagion != 0:
                        IS_links[len(edge)].update(
                            (edge_id, nbr),
                            weight_increment=edgeweight(edge_id),
                        )  # need to be able to multiply by the contagion?

    total_rates = dict()
    total_rates[0] = gamma * infecteds.total_weight()  # I_weight_sum
    for size in unique_edge_sizes:
        total_rates[size] = tau[size] * IS_links[size].total_weight()  # IS_weight_sum

    total_rate = sum(total_rates.values())
    if total_rate > 0:
        delay = random.expovariate(total_rate)
    else:
        print("Total rate is zero and no events will happen!")
        delay = float("Inf")

    t += delay

    while infecteds and t < tmax:
        # rejection sampling
        while True:
            choice = random.choice(list(total_rates.keys()))
            if random.random() < total_rates[choice] / total_rate:
                break

        if choice == 0:  # recover
            recovering_node = (
                infecteds.random_removal()
            )  # chooses a node at random and removes it
            status[recovering_node] = "S"

            # Find the SI links for the recovered node to get reinfected
            for edge_id in memberships[recovering_node]:
                edge = members[edge_id]
                contagion = transmission_function(recovering_node, status, edge, **args)
                if contagion != 0:
                    IS_links[len(edge)].update(
                        (edge_id, recovering_node),
                        weight_increment=edgeweight(edge_id),
                    )

            # reduce the number of infected links because of the healing
            for edge_id in memberships[recovering_node]:
                edge = members[edge_id]
                for nbr in edge:
                    # if the key doesn't exist, don't attempt to remove it
                    if status[nbr] == "S" and (edge_id, nbr) in IS_links[len(edge)]:
                        contagion = transmission_function(nbr, status, edge, **args)
                        if contagion == 0:
                            try:
                                IS_links[len(edge)].remove((edge_id, nbr))
                            except:
                                pass

            times.append(t)
            if recovering_node in community1:
                S1.append(S1[-1] + 1)
                I1.append(I1[-1] - 1)
                S2.append(S2[-1])
                I2.append(I2[-1])
            elif recovering_node in community2:
                S2.append(S2[-1] + 1)
                I2.append(I2[-1] - 1)
                S1.append(S1[-1])
                I1.append(I1[-1])
        else:
            _, recipient = IS_links[choice].choose_random()
            status[recipient] = "I"

            infecteds.update(recipient, weight_increment=nodeweight(recipient))

            for edge_id in memberships[recipient]:
                try:
                    IS_links[len(members[edge_id])].remove((edge_id, recipient))
                except:
                    pass

            for edge_id in memberships[recipient]:
                edge = members[edge_id]
                for nbr in edge:
                    if status[nbr] == "S":
                        contagion = transmission_function(nbr, status, edge, **args)
                        if contagion != 0:
                            IS_links[len(edge)].update(
                                (edge_id, nbr),
                                weight_increment=edgeweight(edge_id),
                            )
            times.append(t)
            if recipient in community1:
                S1.append(S1[-1] - 1)
                I1.append(I1[-1] + 1)
                S2.append(S2[-1])
                I2.append(I2[-1])
            elif recipient in community2:
                S2.append(S2[-1] - 1)
                I2.append(I2[-1] + 1)
                S1.append(S1[-1])
                I1.append(I1[-1])

        total_rates[0] = gamma * infecteds.total_weight()
        for size in unique_edge_sizes:
            total_rates[size] = tau[size] * IS_links[size].total_weight()
        total_rate = sum(total_rates.values())
        if total_rate > 0:
            delay = random.expovariate(total_rate)
        else:
            delay = float("Inf")
        t += delay

    return np.array(times), np.array(S1), np.array(I1), np.array(S2), np.array(I2)
