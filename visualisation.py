import matplotlib.pyplot as plt
from tree import DecisionTree


def _plot_tree_recursive(node=None, coordinates = (0,0), depth=0, vertical_distance=0.8, horizontal_distance=50):
    x, y = coordinates
    #plt.xlim(-90, 110)
    #plt.ylim(-60, 1)
    
    if node.is_terminal_node():
        #print(node.depth)
        node_label = f"leaf: {node.value}"
        plt.text(x, y, node_label, va="center", ha="center", color='green', bbox=dict(facecolor='lightblue', edgecolor='blue', boxstyle='round'), fontsize=5)
    else:
        node_label = f"[X{node.feature}≤{node.threshold}]"
        plt.text(x, y, node_label, va="center", ha="center", bbox=dict(facecolor='lightgreen', edgecolor='green', boxstyle='round'), fontsize=5)

    if node.left is not None:
        x_left = x - max(horizontal_distance / (2**depth+1), 2)
        y_left = y - vertical_distance
        plt.plot([x, x_left], [y, y_left], "-", linewidth=0.7, color="orange")
        _plot_tree_recursive(node.left, coordinates=(x_left,y_left), depth=depth+1)
    
    if node.right is not None:
        x_right = x + max(horizontal_distance / (2**depth+1), 2)
        y_right = y - vertical_distance
        plt.plot([x, x_right], [y, y_right], "-", linewidth=0.7, color="black")
        _plot_tree_recursive(node.right, coordinates=(x_right,y_right), depth=depth+1)
    
    #plt.tight_layout()
    #plt.axis('equal')
    #plt.axis('off')
    #plt.autoscale()


def plot_tree(tree: DecisionTree, figsize=(16, 10), vertical_distance=0.8, horizontal_distance=50):
    # Check if the root contains anything
    if tree.root is None:
        print("Tree is empty.")
        return
    plt.figure(figsize=figsize)
    plt.title("Decision tree visualisation")

    # Call the recursive plotting logic
    _plot_tree_recursive(tree.root, (0,0), 0, vertical_distance, horizontal_distance)
    plt.show()