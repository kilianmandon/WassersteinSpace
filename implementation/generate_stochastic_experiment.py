import numpy as np
from anytree import Node, RenderTree, PreOrderIter
from anytree.exporter import DotExporter


def small_experiment(d):
    root = Node("s", p=1, s=0)
    step1 = Node("up", p=0.5, s=d, parent=root)
    step2 = Node("down", p=0.5, s=-d, parent=root)
    next_step1 = Node("up2", p=step1.p, parent=step1, s=1)
    next_step2 = Node("down2", p=step2.p, parent=step2, s=-1)

    return root

def possible_limit():
    root = Node("s", p=1, s=0)
    step = Node("center", p=1, s=0, parent=root)
    next_step1 = Node("up", p=0.5, s=1, parent=step)
    next_step2 = Node("down", p=0.5, s=-1, parent=step)
    return root


def coin_toss_tree(depth, p=0.5):
    root = Node("s", p=1, s=0)
    last_gen = [root]
    for i in range(depth):
        new_gen = []
        for node in last_gen:
            new_gen.append(Node(f"{node.name}0", parent=node, p=(1 - p) * node.p, s=node.s))
            new_gen.append(Node(f"{node.name}1", parent=node, p=p * node.p, s=node.s+1))
        last_gen = new_gen
    return root

def flatten(experiment, filtration):
    results = experiment.leaves
    dist = np.array([n.p for n in results])
    for node in PreOrderIter(filtration):
        node.subset = np.array([results.index(n) for n in node.subset])
    return {
        'distribution': dist,
        'filtration': filtration,
        'original_nodes': results
    }



def copy(tree, data_callback=None):
    if data_callback is not None:
        new_root = Node(tree.name, **data_callback(tree))
    else:
        new_root = Node(tree.name)
    lookup = {tree: new_root}
    for node in PreOrderIter(tree):
        if node.is_root:
            continue
        if data_callback is not None:
            new_node = Node(node.name, parent=lookup[node.parent], **data_callback(node))
        else:
            new_node = Node(node.name, parent=lookup[node.parent])
        lookup[node] = new_node
    return new_root


def standard_filtration(experiment_tree):
    def data_callback(node):
        subset = list(node.leaves)
        return {'subset': subset}

    return copy(experiment_tree, data_callback)


def filtration_pullup_children(filtration, node):
    old_parent = node.parent
    node.parent = None
    for n in node.children:
        new_node = Node(f"{node.name}#{n.name}", parent=old_parent, subset=n.subset)
        n.parent = new_node
    return filtration


def main():
    experiment1 = coin_toss_tree(depth=3, p=0.5)
    experiment2 = coin_toss_tree(depth=3, p=0.8)
    filtration1 = standard_filtration(experiment1)
    filtration2 = standard_filtration(experiment2)
    filtration_pullup_children(filtration2, filtration2.children[0])



if __name__ == '__main__':
    main()
