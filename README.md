# ProtMonomer
ProtMonomer is an end-to-end deep-learning framework for predicting protein three-dimensional structures from 
multiple sequence alignments (MSAs). This repository contains the inference implementation.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yunda-si/ProtMonomer.git
cd ProtMonomer
chmod 777 ./bin/maxhamming
```

### 2. Create a Python environment

```bash
conda create -n protmonomer python=3.12 -y
conda activate protmonomer
```

### 3. Install Python dependencies
Install a PyTorch build compatible with your CUDA environment:

```bash
# CUDA 12.6
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
# CUDA 13.0
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
# CUDA 13.2
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132
```

Then install the remaining dependencies
```bash
pip install biopython numpy ml-collections PeptideBuilder deepspeed scikit-learn matplotlib
pip install openmm # pip install openmm[cuda12] | pip install openmm[cuda13]
pip install flash-attn --no-build-isolation # Installation may take a while.
pip install nvidia-cutlass
```

Test the attention environment:
```bash
python utils/test_attn.py
```

### 4. Download model weights
Download the [ProtMonomer checkpoints](https://drive.google.com/drive/folders/1_d663gCdwh3wGDHHmMXftAm8-hKRx7mV?usp=sharing) and place them in `./weights`:

```text
weights/
├── stage2.pt
└── stage3.pt
```

`stage2.pt` provides better confidence estimation and is recommended for multi-conformation prediction and screening. `stage3.pt` generally provides higher structure accuracy and is recommended for single-conformation prediction and challenging targets. 
Predicted structures can be refined using `utils/refine.py`. On [45 CASP15 targets](https://drive.google.com/drive/folders/1_d663gCdwh3wGDHHmMXftAm8-hKRx7mV), `stage2.pt` achieves a mean TM-score of approximately 0.816, while `stage3.pt` achieves a mean TM-score of approximately 0.828.

## Quick start

### Predict one target

```bash
python predict_from_msa.py \
  --msa_file ./example/test.a3m \
  --save_path ./results \
  --weight_file ./weights/stage3.pt \
  --device cuda:0
```

### Predict all targets in a directory

```bash
python predict_from_msa.py \
  --msa_path ./example \
  --save_path ./results \
  --weight_file ./weights/stage3.pt \
  --device cuda:0
```

`--msa_file` and `--msa_path` are mutually exclusive. Input files should be in `.a3m` format,
and the first sequence is treated as the query sequence.


## Command-line arguments


| Argument |  Default | Required | Description                                                                    |
|---|---------:|:--------:|--------------------------------------------------------------------------------|
| `--msa_file` |   `None` |   No*    | Path to one input `.a3m` MSA file.                                             |
| `--msa_path` |   `None` |   No*    | Directory containing input `.a3m` files.                                       |
| `--save_path` |   `None` |   Yes    | Output directory.                                |
| `--weight_file` |   `None` |   Yes    | Model checkpoint path.                                          |
| `--file_type` |    `pdb` |    No    | Output format: `cif` or `pdb`.                                          |
| `--device` | `cuda:0` |    No    | Inference device.                                              |
| `--random_seed` |     `42` |    No    | Random seed.                                               |
| `--num_cpu` |      `8` |    No    | Number of CPU threads.                           |
| `--split_seq` |      `0` |    No    | Chunk size along the MSA sequence dimension. `0` disables chunking.            |
| `--split_res` |      `0` |    No    | Chunk size along residue dimensions. `0` disables chunking.                    |
| `--split_atom` |      `0` |    No    | Chunk size along atom dimensions. `0` disables chunking.                       |
| `--num_iter` |      `8` |    No    | Number of structure-recycling iterations.                                      |
| `--filter_msa` | disabled |    No    | Enable MSA filtering. Requires `./bin/maxhamming`.           |
| `--max_homolog` |   `8192` |    No    | Maximum MSA depth.                                         |
| `--last` | disabled |    No    | Evaluate confidence only at the final recycling iteration. |


Chunked computation can reduce GPU memory usage for long sequences or deep MSAs: 

```bash
--split_res <positive_integer>
--split_seq <positive_integer>
--split_atom <positive_integer>
```

All three default to `0`, which disables chunking. Increasing the degree of chunking can reduce memory use but may increase inference time.

Example:

```bash
python predict_from_msa.py \
  --msa_file ./example/test.a3m \
  --save_path ./results \
  --weight_file ./weights/stage3.pt \
  --device cuda:0 \
  --split_res 128 \
  --split_seq 128 \
  --split_atom 2 \
  --num_iter 8
```

## Predicting multiple conformations
Generate sub-MSAs using [AF-Cluster](https://github.com/HWaymentSteele/AF_Cluster), 
[CF-Random](https://github.com/ncbi/CF-random_software), or random downsampling, and then predict structures from each sampled MSA. 
Generated structures can be clustered and visualized with `utils/sample.py`.


## Citation

```bibtex
@article{ProtMonomer2026,
  title   = {Accurate and efficient prediction of protein conformations with ProtMonomer},
  author  = {Yunda Si, Suqi Zhang, Luonan Chen},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {https://doi.org/10.64898/2026.08.28.747824}
}
```

## Contact

For bug reports, feature requests, and usage questions, please open a GitHub issue or contact [yunda_si@ucas.edu.cn](mailto:yunda_si@ucas.edu.cn) or [lnchen@sjtu.edu.cn](mailto:lnchen@sjtu.edu.cn).
