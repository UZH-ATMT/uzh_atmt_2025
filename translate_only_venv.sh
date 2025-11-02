#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=1:0:0
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_translate.out

source ../atmt_venv/bin/activate

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE
python translate.py \
    --input cz-en/data/prepared/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/output.txt \
    --max-len 300