import base64
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional, Tuple, Union

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
        self.threads = threads

        # 1. accelerator 초기화
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        # 2. 모델 초기화 (DP 사용시 각 모델 포트 번호 증가)
        self.vlm_client = BaseClient(vlm_model_name, vlm_host, vlm_port+self._rank, vlm_max_completion_tokens)
        self.llm_client = BaseClient(llm_model_name, llm_host, llm_port+self._rank, llm_max_completion_tokens)
        
    def generate_until(self, requests: List[Instance]) -> List[str]:
        # TODO: async inferece 구현. VLM 인퍼런스를 모두 기다릴 필요 없이, VLM 인퍼런스 완료 한 객체는 바로 LLM 인퍼런스 시작

        """
        Process requests through this model, then feed the outputs to a second model.
        
        Args:
            requests: List of Instance objects for the first model
            
        Returns:
            List of final text outputs from the second model
        """


        # === 1. loop 돌면서 VLM 모델 inference ===
        vlm_res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="VLM Responding")
        for request in requests:
            # 1.1 데이터 처리
            context_dict, gen_kwargs, doc_to_visual, doc_id, task, split = request.arguments
            context_dict = json.loads(context_dict)
            image_url_list = doc_to_visual(self.task_dict[task][split][doc_id])

            vlm_user_prompt = context_dict["vlm_user_prompt"]

            # 2.2 VLM 모델 inference
            response_vlm = self.vlm_client.run(system_prompt=None, user_prompt=vlm_user_prompt, image_url_list=image_url_list)

            # 2.3 데이터 저장
            vlm_res.append(response_vlm)
            pbar.update(1)
        pbar.close()

        # === 3. LLM 모델에 넣기 위한 데이터 처리 ===
        llm_res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="LLM Responding")
        for idx, request in enumerate(requests):
            # 3.1 데이터 처리
            context_dict, gen_kwargs, doc_to_visual, doc_id, task, split = request.arguments
            context_dict = json.loads(context_dict)

            # 3.2 vlm output을 이용해 LLM 입력 프롬프트 생성
            llm_pre_prompt = context_dict["llm_pre_prompt"]
            llm_post_prompt = context_dict["llm_post_prompt"]
            schema = context_dict["schema"]
            prompt = f"{vlm_res[idx]}{llm_pre_prompt}{schema}{llm_post_prompt}"
            
            # 3.3 LLM 모델 inference
            # TODO: structured output 사용 (https://docs.vllm.ai/en/latest/features/structured_outputs.html) -> pydantic json_schema
            llm_response = self.llm_client.run(system_prompt=None, user_prompt=prompt, image_url_list=[])
            llm_res.append(llm_response)

            pbar.update(1)
        pbar.close()

        # === 4. save sample outputs ===
        # os.makedirs("sample_outputs", exist_ok=True)
        # for idx, request in enumerate(requests):
        #     context_dict, gen_kwargs, doc_to_visual, doc_id, task, split = request.arguments
        #     context_dict = json.loads(context_dict)
        #     save_dict = {
        #         "llm_output": llm_res[idx],
        #         "vlm_output": vlm_res[idx],
        #         "context_dict": context_dict,
        #         "doc_id": doc_id,
        #         "doc": self.task_dict[task][split][doc_id]
        #     }
        #     with open(f"sample_outputs/{idx}.json", "w") as f:
        #         json.dump(save_dict, f)
            
        return llm_res

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