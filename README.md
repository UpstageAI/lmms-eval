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

1. Prerequisite
- `pdf2image` 파이썬 라이브러리 사용을 위해 `poppler-utils` 설치가 필요합니다.
    ```
    apt-get install poppler-utils
    ```

2. 데이터셋 전처리
    - `preprocessor/KIE_bench/README.md` 참고


3. VLM_LLM_IE 모델 실행
    - `VLM_LLM_IE` 는 table caption 평가를 위해 VLM inference -> LLM inference 형태의 두번의 인퍼런스를 통해 information extraction을 수행하는 모듈입니다. `vllm` API inference 형식으로 구현하였습니다.  
    - 실행 방법
        1. `UpstageAI/docev-data-engine` repo를 참고하여 필요한 모델의 vllm server를 실행합니다.
        2. 아래 예시 코드를 변형하여 실행합니다. vllm server 실행시 사용한 configuration과 동일해야합니다.
        ```
        bash scripts/run_eval_VLM_LLM_IE.sh \
            --vlm_model_name "Qwen/Qwen2.5-VL-32B-Instruct" \
            --llm_model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
            --vlm_port 8002 \
            --llm_port 8005 \
            --vlm_host "192.168.1.5" \
            --llm_host "192.168.1.5" \
            --vlm_max_tokens 32768 \
            --llm_max_tokens 32768 \
            --port 35000
        ```

4. DP 결과 HTML로 inference하는 방법
    ```
    bash scripts/run_eval_DP_LLM_IE.sh \
        --vlm_model_name "_" \
        --llm_model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
        --vlm_port -1 \
        --llm_port 8005 \
        --vlm_host "_" \
        --llm_host "192.168.1.5" \
        --vlm_max_tokens 0 \
        --llm_max_tokens 32768 \
        --port 35000
    ```