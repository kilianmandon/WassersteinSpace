import time

from scipy.optimize import linprog
import scipy
from anytree import Node, PreOrderIter
from anytree.exporter import DotExporter
import numpy as np
from matplotlib import pyplot as plt

from generate_stochastic_experiment import coin_toss_tree, standard_filtration, filtration_pullup_children, flatten, small_experiment, possible_limit


class Process:
    def __init__(self, distribution: np.ndarray, filtration: Node):
        self.distribution = distribution
        self.filtration = filtration


def adapted_wasserstein(p1, p2, distance):
    distribution1 = p1['distribution']
    n1 = distribution1.size
    distribution2 = p2['distribution']
    n2 = distribution2.size
    filtration1 = p1['filtration']
    filtration2 = p2['filtration']
    conditions = []
    condition_equalities = []
    for x in range(n1):
        condition = np.zeros((n1, n2))
        condition[x, :] = 1
        conditions.append(condition)
        condition_equalities.append(distribution1[x])

    for y in range(n2):
        condition = np.zeros((n1, n2))
        condition[:, y] = 1
        # conditions.append(condition)
        # condition_equalities.append(distribution2[y])
        conditions.append(-condition)
        condition_equalities.append(-distribution2[y])

    for t in range(filtration1.height):
        nodes_x = [n for n in PreOrderIter(filtration1) if n.depth == t]
        nodes_y = [n for n in PreOrderIter(filtration2) if n.depth == t]
        for node1 in nodes_x:
            for node2 in nodes_y:
                for node3 in node1.children:
                    A = node3.subset
                    B = node2.subset
                    C = node1.subset
                    p_a = sum([distribution1[x] for x in A])
                    p_c = sum(distribution1[x] for x in C)
                    condition = np.zeros((n1, n2))
                    condition[A.reshape(-1, 1), B.reshape(1, -1)] = p_c
                    condition[C.reshape(-1, 1), B.reshape(1, -1)] -= p_a
                    conditions.append(condition)
                    condition_equalities.append(0)

                for node3 in node2.children:
                    A = node3.subset
                    B = node1.subset
                    C = node2.subset
                    p_a = sum([distribution1[x] for x in A])
                    p_c = sum(distribution1[x] for x in C)
                    condition = np.zeros((n1, n2))
                    condition[A.reshape(-1, 1), B.reshape(1, -1)] = p_c
                    condition[C.reshape(-1, 1), B.reshape(1, -1)] -= p_a
                    conditions.append(condition)
                    condition_equalities.append(0)

    condition_array = np.zeros((len(conditions), n1 * n2))
    condition_equality_array = np.zeros(len(conditions))
    print(f"Number of conditions: {len(conditions)}")
    print(f"Number of variables: {n1 * n2}")
    for i in range(len(conditions)):
        condition_array[i, :] = conditions[i].flatten()
        condition_equality_array[i] = condition_equalities[i]

    print(f"Non zero proportion: {np.count_nonzero(condition_array) / condition_array.size}")
    # print(np.concatenate((condition_array, condition_equality_array.reshape(-1, 1)), axis=1))
    condition_array = scipy.sparse.csr_matrix(condition_array)
    cost = distance.flatten()

    res = linprog(cost, A_ub=condition_array, b_ub=condition_equality_array, options={'sparse': True})
    x = res['x'].reshape(n1, n2)
    d = res['fun']
    con = res['con']
    print("Result:")
    print(x)
    # print(f"Resulting distance: {d}")
    # print(f"Condition error: {np.linalg.norm(con)}")
    return x,d,con


def paper_experiment():
    d1s = []
    d2s = []
    for n in range(10):
        exp1 = small_experiment(2 ** (-n))
        exp2 = possible_limit()
        f1 = standard_filtration(exp1)
        f2 = standard_filtration(exp2)
        f3 = standard_filtration(exp2)
        f3 = filtration_pullup_children(f3, f3.children[0])
        DotExporter(f3).to_picture('implementation/vis0.png')
        DotExporter(f2).to_picture("implementation/vis2.png")
        DotExporter(f1).to_picture("implementation/vis1.png")

        p1 = flatten(exp1, f1)
        p2 = flatten(exp2, f2)
        p3 = flatten(exp2, f3)

        def distance_function(i, j):
            d = 0
            n1 = p1['original_nodes'][i]
            n2 = p2['original_nodes'][j]

            while n1.parent is not None:
                d += np.abs(n1.s-n2.s)
                n1 = n1.parent
                n2 = n2.parent
            d += np.abs(n1.s - n2.s)
            return d
                

        distance = np.array([
            [distance_function(i, j) for j in range(p2['distribution'].size)]
            for i in range(p1['distribution'].size)])
        print(distance)

        _, d1, _ = adapted_wasserstein(p1, p2, distance)
        _, d2, _ = adapted_wasserstein(p1, p3, distance)
        d1s.append(d1)
        d2s.append(d2)

    plt.plot(d1s, label=r"$\mathcal{AW}_p(X^s, Y)$")
    # plt.plot(d2s, label="true limit")
    plt.ylim(bottom=0)
    plt.legend()
    # plt.show()
    plt.savefig('implementation/figure.png')

def old_experiment():
    # Classic coin toss: p1 is a fair toss with natural sigma algebra,
    # p2 is unfair and with previsible sigma algebra at some node
    exp1 = coin_toss_tree(depth=5, p=0.5)
    exp2 = coin_toss_tree(depth=5, p=0.5)
    f1 = standard_filtration(exp1)
    f2 = standard_filtration(exp2)
    filtration_pullup_children(f2, f2.children[0])

    p1 = flatten(exp1, f1)
    p2 = flatten(exp2, f2)
    distance_fctn = lambda i, j: np.abs(p1['original_nodes'][i].s - p2['original_nodes'][j].s)
    distance = np.array([
        [distance_fctn(i, j) for j in range(p2['distribution'].size)]
        for i in range(p1['distribution'].size)])

    t1 = time.time()
    adapted_wasserstein(p1, p2, distance)
    print(f"Took: {time.time() - t1}")

def main():
    paper_experiment()


if __name__ == '__main__':
    main()
