import os
import torch
import torch.optim as optim
import json
from dynamic_net import DynamicNet
from model import load_credit_and_train, load_adult_and_train
from global_tree_surrogate import train_global_surrogate_tree, compute_accuracies, export_tree_json, export_witness_json
from lime_shap import lime_fidelity, dtree_fidelity, shap_fidelity

# Config
NN_LAYERS = [2, 4, 8, 16]
TREE_DEPTHS = [3, 4, 5, 6, 7, 8]
SAVE_ROOT = "../experiments"
MODEL_ROOT = "../stored_models"
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

def nn_path(dataset, L):
    return os.path.join(MODEL_ROOT, f"{dataset}_L{L}.pt")


def save_model(model, dataset, L):
    torch.save(model.state_dict(), nn_path(dataset, L))


def load_model_if_exists(model, dataset, L):
    path = nn_path(dataset, L)
    if os.path.exists(path):
        print(f"✓ Loading cached model: {path}")
        model.load_state_dict(torch.load(path))
        return True
    return False

def run_experiments_for_dataset(dataset_name, loader_fn):

    print(f"\n========================================")
    print(f" RUNNING EXPERIMENTS FOR: {dataset_name}")
    print(f"========================================\n")

    dataset_dir = os.path.join(SAVE_ROOT, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    # Load dataset only once
    _, X_train, y_train, X_test, y_test, input_dim, scaler, features = loader_fn(epochs=0)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    # results[L][depth] = metrics
    results = {}

    for L in NN_LAYERS:

        print(f"\n==============================")
        print(f" Neural Network depth = {L} ")
        print(f"==============================")

        # Folder for storing tree/witness
        depth_dir = os.path.join(dataset_dir, f"L{L}")
        os.makedirs(depth_dir, exist_ok=True)
        results[L] = {}

        # Build dynamic NN
        model = DynamicNet(
            input_dim=input_dim,
            hidden_size=16,
            output_dim=2,
            total_layers=L
        )

        # Load pretrained model if available
        if not load_model_if_exists(model, dataset_name, L):
            print(f"Training neural network L={L} from scratch...")

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(800):
                model.train()
                optimizer.zero_grad()
                logits = model(X_train_t)
                loss = criterion(logits, y_train_t)
                loss.backward()
                optimizer.step()

            save_model(model, dataset_name, L)
            print(f"✓ Saved model {dataset_name}_L{L}.pt")

        # Compute NN accuracy ONCE per L
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32))
            nn_pred = logits.argmax(dim=1).cpu().numpy()

        nn_acc = (nn_pred == y_test).mean()
        results[L]["NN_accuracy"] = nn_acc

        for depth in TREE_DEPTHS:

            print(f"\n--- Training tree depth={depth} ---")

            tree = train_global_surrogate_tree(
                model, X_train,
                max_depth=depth,
                min_samples_leaf=200
            )

            tree_json = export_tree_json(tree, feature_names=features)
            with open(os.path.join(depth_dir, f"{depth}_tree.json"), "w") as f:
                json.dump(tree_json, f, indent=4)
            print(f"✓ Saved {dataset_name}/L{L}/{depth}_tree.json")

            x = X_test[0]
            witness = export_witness_json(tree, x)

            with open(os.path.join(depth_dir, f"{depth}_witness.json"), "w") as f:
                json.dump(witness, f, indent=4)
            print(f"✓ Saved {dataset_name}/L{L}/{depth}_witness.json")

            # Accuracies
            _, tree_acc, _ = compute_accuracies(model, tree, X_test, y_test)

            # Fidelity metrics
            lime_g, _, _ = lime_fidelity(model, X_test, scaler, K=50, n_neigh=300, mode="gaussian")
            lime_u, _, _ = lime_fidelity(model, X_test, scaler, K=50, n_neigh=300, mode="uniform")
            dt_fid, _, _ = dtree_fidelity(model, tree, X_test, K=50)
            sh_fid, _, _ = shap_fidelity(model, X_train, X_test, K=50, nsamples=300)

            # Store metrics
            results[L][depth] = {
                "Tree_accuracy": tree_acc,
                "Tree_fidelity": dt_fid,
                "LIME_G": lime_g,
                "LIME_U": lime_u,
                "SHAP": sh_fid
            }

            print(f"Depth {depth}: Tree Fidelity = {dt_fid:.4f}")

    print(f"\n============== SUMMARY FOR {dataset_name.upper()} ==============\n")

    for L in NN_LAYERS:
        print(f"\n--------------------------------------------------------")
        print(f"              RESULTS FOR NN LAYERS = {L}")
        print(f"--------------------------------------------------------")
        print(f"Neural Network Accuracy = {results[L]['NN_accuracy']:.4f}\n")

        for depth in TREE_DEPTHS:
            m = results[L][depth]
            print(f"Tree Depth = {depth}")
            print(f"  Tree Accuracy:      {m['Tree_accuracy']:.4f}")
            print(f"  Tree Fidelity:      {m['Tree_fidelity']:.4f}")
            print(f"  LIME Gaussian:      {m['LIME_G']:.4f}")
            print(f"  LIME Uniform:       {m['LIME_U']:.4f}")
            print(f"  SHAP Fidelity:      {m['SHAP']:.4f}")
            print("")

    return results

adult_results = run_experiments_for_dataset("adult", load_adult_and_train)
credit_results = run_experiments_for_dataset("credit", load_credit_and_train)