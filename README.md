# DocVision 모델 평가용 lmms-eval 패키지

## 설명

`lmms-eval` 프로그램을 포크하여, DocVision 모델을 추가한 레포지토리입니다.
모델 평가를 수행하기 위해서는 다음 설명에 따라 설정을 진행합니다.

### 단계 1 - 평가용 Conda 환경 설정

아래와 같이 lmms-eval 사용을 위한 Conda 환경을 생성합니다.
이 때, MoRA adapter 사용을 위해 `mora` 패키지를 추가로 설치합니다.

```bash
(base) hancheol@upstage-node-001:~/projects $

# lmms-eval 환경 생성
conda create -n lmms-eval python=3.11
conda activate lmms-eval

# DocVision용 lmms-eval 설치
git clone git@github.com:UpstageAI/lmms-eval.git
cd lmms-eval
pip install -e .

# 추가 필수 라이프러리 설치
pip install -r requirements.txt
```

## 단계 2 - 모델 준비 및 평가 실행

학습된 DocVision 모델을 준비합니다.
그리고 아래와 같이 `scripts/run_eval.sh` 파일을 실행하여 평가를 수행합니다.
`--model_path` 와 `--tasks` 는 필수 인자입니다.
lmms-eval이 지원하는 모든 태스크는 [current_tasks.md](https://github.com/UpstageAI/lmms-eval/blob/main/docs/current_tasks.md)에서 확인 가능합니다.

```bash
(lmms-eval) hancheol@upstage-node-001:~/projects/lmms-eval $

scripts/run_eval.sh \
    --model_path /app/docfm/checkpoints/training/DocVision/SFT-SyntheticData/20250208_solar-exp-2_with-figureqa_900kX3_multipage-base-model/steps_5240 \
    --tasks docvqa \
    --gpu_ids 0,1,2,3 \
    --port 35001
```


### (Optional) KIE_bench 사용을 위한 추가 설정
- `pdf2image` 파이썬 라이브러리 사용을 위해 `poppler-utils` 설치가 필요합니다.
    ```
    apt-get install poppler-utils
    ```

- Dataset 전처리 (KIE-bench -> KIE-bench huggingface version)
    - 주의: 이를 통해 생성된 KIE-bench huggingface version에는 일부 정보가 실제 Upstage huggingface uploaded version과 다를 수 있습니다. (다른 부분은 코드 참고) 하지만 사용되지 않는 부분이기에 최신 버전의 KIE bench를 빠르게 적용하기 위해 전처리 코드를 작성하였습니다. 최신 버전의 KIE-bench가 huggingface에 업로드 되어 있는 경우, 해당 데이터를 다운로드 받아 사용하시기 바랍니다.
    ```
    # 1. 코드 내 base_path 변수 수정
    # 2. 아래 코드 실행
    python preprocessor/KIE_bench_to_HF_dataset.py
    ```