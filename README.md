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
