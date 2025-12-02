<h1 align="center">
ExpProof-Dtree
</h1>

This code supports training of global surrogate tree for neural networks which can be used for model explainability. 
We also compare our experiment method against local explanation methods LIME and SHAP.

To run the code, use conda environment.

```conda create -f environment.yml```

Run
```python model/main.py```

This experiment runs
1. Loads Adult and credit dataset
2. Trains neural networks of layers [2,4,8,16] 
3. Creates surrogate decsion trees on those neural networks of depth [3,4,5,6,7,8]
4. Compares the fidelity over 50 test points for surrogate tree, LIME and SHAP

# Tree JSON

You can find the tree JSON for different layers of neural network and different tree depths 
in the [Adult Surrogate trees](experiments/adult) and [Credit Surrogate trees](experiments/credit) directories.

# Heatmap Plot

The heatmap code and plot is also availble in the [Experiments](experiments) directory. 
