import base64
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional, Tuple, Union
import os
import asyncio
import numpy as np
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from lmms_eval.models.vllm_client_utils.BaseClient import BaseClient

@register_model("VLM_LLM_IE")
class VLM_LLM_IE(lmms):
    """
    VLM_LLM_IE 모델은 핵심 정보 추출(KIE) 벤치마크를 위한 모델입니다.
    VLM 모델 인퍼런스, LLM 모델 인퍼런스를 순차적으로 진행합니다.

    vllm 백엔드를 사용하여 VLM, LLM 모델을 구현합니다.
    - vllm 백엔드 사용 시 vllm에서 구현된 모델들은 새롭게 구현하지 않아도 되는 장점이 있습니다.

    reference: lmms_eval/models/vllm.py
    """


    def __init__(
        self,
        vlm_model_name: str,
        llm_model_name: str,
        vlm_host: str = "localhost",
        llm_host: str = "localhost",
        vlm_port: int = 35001,
        llm_port: int = 35002,
        vlm_max_completion_tokens: int = 4096,
        llm_max_completion_tokens: int = 4096,
        threads: int = 16,  # Threads to use for decoding visuals
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert batch_size == 1, "batch_size must be 1"
        use_docev_dp = "docev_dp" in vlm_model_name.lower() or "docevdp" in vlm_model_name.lower() or "docev-dp" in vlm_model_name.lower()
        self.vlm_prompt_key = "DocEV_DP_user_prompt" if use_docev_dp else "vlm_user_prompt"
        eval_logger.info(f"VLM prompt key: {self.vlm_prompt_key}")

        self.vlm_prompt_key = "DocEV_DP_user_prompt" if "docev_dp" in vlm_model_name.lower() else "vlm_user_prompt"
        eval_logger.info(f"VLM prompt key: {self.vlm_prompt_key}")

        # 2. 모델 초기화 (DP 사용시 각 모델 포트 번호 증가)
        self.vlm_client = BaseClient(vlm_model_name, vlm_host, vlm_port, max_completion_tokens=vlm_max_completion_tokens)
        self.llm_client = BaseClient(llm_model_name, llm_host, llm_port, max_completion_tokens=llm_max_completion_tokens)
        

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Process requests through VLM model and then LLM model asynchronously.
        VLM 인퍼런스 완료된 요청은 즉시 LLM 인퍼런스로 넘어갑니다.
        
        Args:
            requests: List of Instance objects for the first model
            
        Returns:
            List of final text outputs from the second model
        """
        # 비동기 이벤트 루프 실행
        results = asyncio.run(self._async_generate_until(requests))
        return results
    
    async def _async_generate_until(self, requests: List[Instance]) -> List[str]:
        """비동기 처리를 위한 내부 메소드"""
        total = len(requests)
        
        # 프로그레스바 (메인 프로세스에서만 표시)
        pbar = tqdm(total=total, disable=(self._rank != 0), desc="Processing")
        
        # 동시 실행 제한을 위한 세마포어 (최대 4개의 async 요청 동시 실행. vllm server에 동시에 수백~수천개의 인퍼런스 요청이 가지 않도록 제한하는 역할)
        semaphore = asyncio.Semaphore(4)
        
        async def _process_single_request_limited(request, idx, total):
            async with semaphore:
                return await self._process_single_request(request, idx, total, pbar)
        
        # 모든 요청에 대해 비동기 처리 시작
        tasks = []
        for idx, request in enumerate(requests):
            task = asyncio.create_task(
                _process_single_request_limited(request, idx, total)
            )
            tasks.append(task)
        
        # 완료된 태스크 결과 수집
        completed_results = await asyncio.gather(*tasks)
        pbar.close()
        
        results = [result["llm_response"] for result in completed_results]
        return results

    async def _process_single_request(self, request, idx, total, progress_bar=None):
        """비동기적으로 단일 요청 처리: VLM -> LLM 순서로 인퍼런스 진행"""
        # 1. 데이터 처리
        context_dict, gen_kwargs, doc_to_visual, doc_id, task, split = request.arguments
        context_dict = json.loads(context_dict)
        image_url_list = doc_to_visual(self.task_dict[task][split][doc_id])
        vlm_user_prompt = context_dict[self.vlm_prompt_key]

        # 2. VLM 모델 인퍼런스
        vlm_response = self.vlm_client.run(system_prompt=None, user_prompt=vlm_user_prompt, image_url_list=image_url_list)
        
        if progress_bar:
            progress_bar.set_description(f"VLM [{idx+1}/{total}] completed & LLM [{idx}/{total}] completed")

        # 3. VLM 출력을 기반으로 LLM 입력 프롬프트 생성
        llm_pre_prompt = context_dict["llm_pre_prompt"]
        llm_post_prompt = context_dict["llm_post_prompt"]
        schema = json.loads(context_dict["schema"])
        prompt = f"{llm_pre_prompt}{vlm_response}{llm_post_prompt}"
        
        # 4. LLM 모델 인퍼런스
        llm_response = self.llm_client.run(system_prompt=None, user_prompt=prompt, image_url_list=[], guided_json=schema)
        
        if progress_bar:
            progress_bar.set_description(f"VLM [{idx+1}/{total}] completed & LLM [{idx+1}/{total}] completed")
            progress_bar.update(1)
        
        # 5. 결과 및 샘플 저장
        # os.makedirs("sample_outputs", exist_ok=True)
        # save_dict = {
        #     "llm_output": llm_response,
        #     "vlm_output": vlm_response,
        #     "context_dict": context_dict,
        #     "doc_id": doc_id,
        #     "doc": self.task_dict[task][split][doc_id],
        #     "vlm_user_prompt": context_dict["vlm_user_prompt"],
        #     "llm_user_prompt": f"{context_dict['llm_pre_prompt']}{vlm_response}{context_dict['llm_post_prompt']}"
        # }
        # with open(f"sample_outputs/{idx}.json", "w") as f:
        #     json.dump(save_dict, f)
            
        return {
            "idx": idx,
            "vlm_response": vlm_response,
            "llm_response": llm_response,
        }

    

    def _set_gen_kwargs(self, gen_kwargs):
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if gen_kwargs["max_new_tokens"] > 4096:
            gen_kwargs["max_new_tokens"] = 4096
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = 0.95
        return gen_kwargs

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _preprocess_vlm(self, visuals):
        if 'docev_preview' in self.vlm_pretrained:
            return visuals
        else:
            raise ValueError(f"Unsupported VLM model: {self.vlm_pretrained}")
        
    def _preprocess_llm(self, texts):
        if 'deepseek' in self.llm_pretrained.lower():
            return texts
        else:
            raise ValueError(f"Unsupported LLM model: {self.llm_pretrained}")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood() is not applicable for VLM-LLM-IE")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("generate_until_multi_round() is not applicable for VLM-LLM-IE")

    def _encode_video(self, video: str):
        raise NotImplementedError("_encode_video() is not applicable for VLM-LLM-IE")

    def _encode_image(self, image: Union[Image.Image, str]):
        """
        Encode the image to base64 string
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str