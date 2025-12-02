import numpy as np
import torch
import time
from sklearn.linear_model import Ridge
import shap

shap.utils._tqdm = lambda *args, **kwargs: args[0]

def nn_predict(model, X_np):
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32)
        logits = model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

def nn_predict_proba(model, X_np):
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32)
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()



# LIME neighborhood sampling
def lime_neighborhood(x, n_samples, d, scaler=None,
                      mode="gaussian", sigma=0.2, half_edge=0.2):

    if mode == "gaussian":
        Z = np.random.normal(loc=x, scale=sigma, size=(n_samples, d))
        w = np.ones(n_samples)

    elif mode == "uniform":
        Z = x + np.random.uniform(-half_edge, half_edge, size=(n_samples, d))
        dist = np.linalg.norm(Z - x, axis=1)
        kernel_width = np.sqrt(d) * 0.75
        w = np.exp(-(dist ** 2) / (kernel_width ** 2))

    else:
        raise ValueError("mode must be gaussian or uniform")

    return Z, w


# LIME Fidelity (Gaussian or Uniform)
def lime_fidelity(model, X_test, scaler, K=50, n_neigh=300, mode="gaussian"):

    start = time.time()
    d = X_test.shape[1]
    indices = np.random.choice(len(X_test), K, replace=False)
    fids = []

    for idx in indices:
        x = X_test[idx]

        # 1) Sample neighbors
        Z, w = lime_neighborhood(x, n_neigh, d, scaler, mode=mode)

        # 2) NN predictions
        nn_prob = nn_predict_proba(model, Z)[:, 1]

        # 3) Fit weighted linear model
        reg = Ridge(alpha=1e-3)
        reg.fit(Z, nn_prob, sample_weight=w)

        # 5) Fidelity: compare model and LIME across the entire sampled neighborhood
        nn_lbls_Z = (nn_prob >= 0.5).astype(int)
        sur_lbls_Z = (reg.predict(Z) >= 0.5).astype(int)

        fids.append((sur_lbls_Z == nn_lbls_Z).mean())

    return np.mean(fids), np.std(fids), time.time() - start


# DECISION TREE Fidelity
def dtree_fidelity(model, tree, X_test, K=50):

    start = time.time()
    indices = np.random.choice(len(X_test), K, replace=False)
    fids = []

    for idx in indices:
        x = X_test[idx:idx+1]
        nn_lbl = nn_predict(model, x)[0]
        tree_lbl = tree.predict(x)[0]
        fids.append(int(tree_lbl == nn_lbl))

    return np.mean(fids), np.std(fids), time.time() - start

# SHAP Fidelity
def shap_fidelity(model, X_train, X_test, K=50, nsamples=200, bg_size=200):

    start = time.time()

    # background
    if len(X_train) > bg_size:
        idx = np.random.choice(len(X_train), bg_size, replace=False)
        bg = X_train[idx]
    else:
        bg = X_train

    # wrapper ensures SHAP sees a (n,2) output
    def shap_wrapper(Z):
        return nn_predict_proba(model, Z)

    explainer = shap.KernelExplainer(shap_wrapper, bg)
    indices = np.random.choice(len(X_test), K, replace=False)
    fids = []

    for idx in indices:
        x = X_test[idx:idx+1]
        nn_lbl = nn_predict(model, x)[0]

        shap_vals = explainer.shap_values(x, nsamples=nsamples)

        # Multi-class (binary) SHAP
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            phi = shap_vals[1][0]
            base = explainer.expected_value[1]
        else:
            phi = shap_vals[0][0]
            base = explainer.expected_value[0]

        surrogate_prob = base + phi.sum()
        sur_lbl = int(surrogate_prob >= 0.5)

        fids.append(int(nn_lbl == sur_lbl))

    return np.mean(fids), np.std(fids), time.time() - start