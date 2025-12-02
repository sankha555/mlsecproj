import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text


def nn_predict(model, X_np):
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32)
        logits = model(X_t)
        return logits.argmax(dim=1).cpu().numpy()


# 2. Train global surrogate decision tree
def train_global_surrogate_tree(model,
                                X,
                                max_depth=6,
                                min_samples_leaf=20,
                                criterion="gini"):
    y_nn = nn_predict(model, X)
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=0
    )
    tree.fit(X, y_nn)

    return tree

# 3. Compute validation accuracy of NN and tree
def compute_accuracies(model, tree, X_test, y_test):
    y_nn_pred = nn_predict(model, X_test)
    y_tree_pred = tree.predict(X_test)

    nn_val_acc = (y_nn_pred == y_test).mean()
    tree_val_acc = (y_tree_pred == y_test).mean()
    fidelity = (y_nn_pred == y_tree_pred).mean()

    return nn_val_acc, tree_val_acc, fidelity


# 4. Human-readable global explanation (rules)
def print_global_rules(tree, input_dim):
    feature_names = [f"x{i}" for i in range(input_dim)]
    rules = export_text(tree, feature_names=feature_names)
    print("\n=== Global Decision Tree Rules ===")
    print(rules)
    return rules

# 5. Tree to JSON
def export_tree_json(tree, feature_names):
    tree_ = tree.tree_
    def recurse(node):
        node_info = {}
        # If leaf node
        if tree_.feature[node] == -2:
            # class is argmax of value array
            cls = int(np.argmax(tree_.value[node][0]))
            node_info["leaf"] = True
            node_info["class"] = cls
            node_info["samples"] = int(tree_.n_node_samples[node])
            return node_info

        # Otherwise: internal node
        feature = tree_.feature[node]
        threshold = float(tree_.threshold[node])

        node_info["leaf"] = False
        node_info["feature"] = feature_names[feature]
        node_info["threshold"] = threshold
        node_info["samples"] = int(tree_.n_node_samples[node])

        left = recurse(tree_.children_left[node])
        right = recurse(tree_.children_right[node])

        node_info["left"] = left
        node_info["right"] = right

        return node_info

    return recurse(0)

def export_witness_json(tree, x):
    node = tree.tree_
    feature = node.feature
    threshold = node.threshold

    witness = {
        "input": x.tolist(),
        "path": [],
        "prediction": int(tree.predict(x.reshape(1, -1))[0])
    }

    idx = 0  # start at root

    while feature[idx] != -2:  # while not leaf
        feat = feature[idx]
        thr = threshold[idx]
        val = float(x[feat])

        go_left = val <= thr
        decision = {
            "feature_index": int(feat),
            "threshold": float(thr),
            "value": float(val),
            "comparison": "<=",
            "result": bool(go_left)
        }

        witness["path"].append(decision)

        # follow the path
        idx = node.children_left[idx] if go_left else node.children_right[idx]

    return witness