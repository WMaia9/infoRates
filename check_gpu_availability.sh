#!/bin/bash
# Script para verificar disponibilidade de GPUs nos nós

echo "=== VERIFICAÇÃO DE GPUS DISPONÍVEIS ==="
echo

echo "Nós GPU disponíveis:"
sinfo -p gpu --format="%.10N %.8T %.10m %.6c" | grep -E "(idle|mix)"

echo
echo "Detalhes dos nós GPU:"
for node in gnode001 gnode008 gnode017 gnode018 gnode019 gnode020 gnode021 gnode024; do
    echo "Nó $node:"
    srun --partition=gpu --nodelist=$node --gres=gpu:a100:1 --time=00:00:10 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "  Indisponível"
done

echo
echo "Para usar multi-node, execute:"
echo "./run_multi_node_eval.sh"