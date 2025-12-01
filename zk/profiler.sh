#!/bin/bash

exp="$1"

echo "Explanation ZKP Costs"
echo ""

mkdir proof_${exp}
cd proof_${exp}

set -e
echo "Proof Setup"
echo ""
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First" -v -e="random"
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v

echo ""
echo "Generate Prover key"
echo ""
snarkjs groth16 setup ../r1cs/${exp}.r1cs pot12_final.ptau credit_0000.zkey
snarkjs zkey contribute credit_0000.zkey credit_0001.zkey --name="First" -v -e="random"

echo ""
echo "Export Verifier key"
echo ""
snarkjs zkey export verificationkey credit_0001.zkey vkey.json

echo "Generate Witness"
echo ""
pwd
node ../generate_witness.js ../wasm/${exp}.wasm ../input/input_${exp}.json witness.wtns

# Time proof generation
echo "Generate Proof"
echo ""
echo "Prover Time:"
time snarkjs groth16 prove credit_0001.zkey witness.wtns proof.json public.json
echo ""

# Time verification
echo "Verify Proof"
echo ""
echo "Verification Time:"
time snarkjs groth16 verify vkey.json public.json proof.json
echo ""


# Show Proof Size
kb_divide=1024
PROOF_SIZE=$(wc -c < proof.json)
PROOF_KB=$(echo "scale=2; $PROOF_SIZE / $kb_divide" | bc)
echo "Proof Size: " $PROOF_KB "KB"
echo ""

cd ..