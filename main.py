import matplotlib.pyplot as plt
import numpy as np
import random as r


class Node:
    def __init__(self, idn, x, y):
        self.idn = idn
        self.pos = np.array((float(x), float(y)))


class Bee:
    def __init__(self):
        self.choosen_nodes = []
        self.recuiter = True
        self.distance = 0.0

    def choose_rand_move(self, move, nods):
        # choosen node must be unique
        for i in range(move):
            if self.is_complete():
                break
            else:
                sel = nods[r.randint(0, len(nodes) - 1)]
                while sel in self.choosen_nodes:
                    sel = nods[r.randint(0, len(nodes) - 1)]
                self.choosen_nodes.append(sel)

            self.total_distance()

    def change_role(self, role):
        self.recuiter = role

    def replace_nodes(self, nods):
        self.choosen_nodes = nods
        self.total_distance()

    def total_distance(self):
        distance = 0.0
        for i in range(len(self.choosen_nodes) - 1):
            node1 = self.choosen_nodes[i]
            node2 = self.choosen_nodes[i + 1]
            distance += np.linalg.norm(node1.pos - node2.pos)

        distance += np.linalg.norm(self.choosen_nodes[-1].pos - self.choosen_nodes[0].pos)
        self.distance = distance

    def is_complete(self):
        if len(self.choosen_nodes) >= len(nodes):
            return True
        else:
            return False


def load_nodes(filename):
    ret = []
    with open(filename) as f:
        nodes_s = f.readlines()
    nodes_s = [x.strip() for x in nodes_s]
    for n in nodes_s:
        node = n.split(' ')
        ret.append(Node(node[0], node[1], node[2]))
    return ret


nodes = load_nodes("data1")


# Bee Colony Optimization Algorithm

def main():
    epoch = 10
    n_bee = 500
    n_move = 3

    bees = []
    best_bee = Bee()

    e = 0

    # init bees
    for i in range(n_bee):
        bees.append(Bee())

    while not best_bee.is_complete():

        print "\nEpoch", e + 1

        print "forward pass"
        # forward pass
        for bee in bees:
            bee.choose_rand_move(n_move, nodes)

        # backward pass
        print "evaluating"
        bees = sorted(bees, key=lambda be: be.distance, reverse=False)
        best_bee = bees[0]

        print "Best distance so far", best_bee.distance
        print "Best route so far", [n.idn for n in best_bee.choosen_nodes]

        print "Bees are making decision to be recruiter or follower"
        Cmax = max(bees, key=lambda b: b.distance).distance
        Cmin = min(bees, key=lambda b: b.distance).distance

        recruiters = []
        for bee in bees:
            Ob = (Cmax - bee.distance) / (Cmax - Cmin)  # range [0,1]
            probs = np.e ** (-(1 - Ob) / (len(bee.choosen_nodes) * 0.01))
            rndm = r.uniform(0, 1)
            # print "ob and probs", Ob, probs
            if rndm < probs:
                bee.change_role(True)
                recruiters.append(bee)
            else:
                bee.change_role(False)

        print "number of recruiter", len(recruiters)
        print "Bees are choosing their recruiter"
        # creating a roulette wheel
        divider = sum([(Cmax - bee.distance) / (Cmax - Cmin) for bee in recruiters])
        probs = [((Cmax - bee.distance) / (Cmax - Cmin)) / divider for bee in recruiters]
        cumulative_probs = [sum(probs[:x + 1]) for x in range(len(probs))]

        for bee in bees:
            if not bee.recuiter:
                rndm = r.uniform(0, 1)
                selected_bee = Bee()
                for i, cp in enumerate(cumulative_probs):
                    if rndm < cp:
                        selected_bee = recruiters[i]
                        break
                bee.replace_nodes(selected_bee.choosen_nodes[:])
        e += 1


def sandbox():
    x = [node.pos[0] for node in nodes]
    y = [node.pos[1] for node in nodes]
    l = [node.idn for node in nodes]

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, lbl in enumerate(l):
        ax.annotate(lbl, (x[i], y[i]))

    plt.show()


# sandbox()
main()
