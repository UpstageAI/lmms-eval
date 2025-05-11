# KIE-bench DP inference script

This folder contains scripts for preprocessing KIE_bench dataset.

## 1. DP pipeline

### Installation
To install the required dependencies, run:
```bash
conda create -n dp_pipeline python=3.10
conda activate dp_pipeline
pip install -r requirements.txt

# For handling PDFs, make sure poppler is installed
apt-get update && apt-get install -y poppler-utils
```

### How to use
```bash
# 1. change input_dir and output_dir in "KIE_bench_dp.sh" 
# 2. run
bash KIE_bench_dp.sh
```

- json 파일들이 `{output_dir}/dp_result/raw/` 경로에 저장됩니다.
- For more details, See "https://github.com/UpstageAI/docev-data-engine/blob/sehwan/feat/dp-pipeline/document-parse/README.md"

## 2. DP to html

### How to use
DP(Document Parser)에서 생성된 JSON 파일을 HTML로 변환합니다.

```bash
python KIE_bench_dp2html.py <dp json files dir> <dp html files dir>
```

### Parameters
- `<dp json files dir>`: DP 결과 JSON 파일이 저장된 디렉토리 경로
- `<dp html files dir>`: 변환된 HTML 파일이 저장될 디렉토리 경로
- `--num_cpus`: 병렬 처리에 사용할 CPU 코어 수 (기본값: 50)

```bash
# 예시
python KIE_bench_dp2html.py ./dp_result/raw ./dp_result/html --num_cpus 8
```

## 3. to HF dataset

- KIE-bench -> KIE-bench huggingface version
    - 주의: 이를 통해 생성된 KIE-bench huggingface version에는 일부 정보가 실제 Upstage huggingface uploaded version과 다를 수 있습니다. (다른 부분은 코드 참고) 하지만 사용되지 않는 부분이기에 최신 버전의 KIE bench를 빠르게 적용하기 위해 전처리 코드를 작성하였습니다. 최신 버전의 KIE-bench가 huggingface에 업로드 되어 있는 경우, 해당 데이터를 다운로드 받아 사용하시기 바랍니다.
    ```
    # 1. 아래 경로의 코드 내 base_path 변수 수정
    # 2. 아래 코드 실행
    python preprocessor/KIE_bench_to_HF_dataset.py
    ```
