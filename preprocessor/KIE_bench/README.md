# KIE-bench DP inference script

This folder contains scripts for preprocessing KIE_bench dataset.

## Installation

To install the required dependencies, run:
```bash
conda create -n dp_pipeline python=3.10
conda activate dp_pipeline
pip install -r requirements.txt

# For handling PDFs, make sure poppler is installed
apt-get update && apt-get install -y poppler-utils
```

## How to use

```bash
# 1. change input_dir and output_dir in "KIE_bench_dp.sh" 
# 2. run
bash KIE_bench_dp.sh
```


- For more details, See "https://github.com/UpstageAI/docev-data-engine/blob/sehwan/feat/dp-pipeline/document-parse/README.md"