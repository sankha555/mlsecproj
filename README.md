
## Cleartext Decision Tree Semantics
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

### Tree JSON

You can find the tree JSON for different layers of neural network and different tree depths 
in the [Adult Surrogate trees](dtree/experiments/adult) and [Credit Surrogate trees](dtree/experiments/credit) directories.

### Heatmap Plot

The heatmap code and plot is also availble in the [Experiments](dtree/experiments) directory. 

<hr>

## Steps to Replicate ZKP Costs in Table 6

1. Install dependencies
```
sudo apt update
sudo apt install -y build-essential libgmp-dev libsodium-dev nasm

curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

git clone https://github.com/iden3/circom.git
cd circom
cargo build --release
sudo cargo install --path circom

npm install -g snarkjs
snarkjs --version
```

2. Enter the `zk` directory
```
cd zk
```

3. Run the profiler (`profiler.sh`) script with a single CLI argument indicating the dataset (`adult` `credit`), tree depth (`5`) and number of layers in the neural network (`2 4 8 16`). Eg. to replicate proof costs for adult dataset with tree depth 5 and 2 layers in the neural network, run 
```
chmod +x profiler.sh
./profiler.sh adult52
```

4. Let the setup, key generation, proof generation and verification run. The prover times and verifier times are printed at the end (look at the row indicating `user` time). The proof size is printed at the very end.
