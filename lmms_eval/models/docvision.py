# InstructBLIP (instructblip.py)를 참조하여 작성

import os
import warnings
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.docvision_utils.pipeline_initialize import init_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


@register_model("docvision")
class DocVision(lmms):
    """
    DocVision Model
    """

    def __init__(
        self,
        pretrained: str,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        # 1. config 로딩
        def get_config(pretrained):
            # 저장된 모델 경로의 training_config.yaml 파일을 수정 없이 그대로 사용할 수 있도록, 환경 변수를 변경
            config = OmegaConf.load(os.path.join(pretrained, "training_config.yaml"))
            config.components.architecture.class_name = config.components.architecture.class_name.replace("mllm_engine.models", "lmms_eval.models.docvision_utils")
            config.components.vision_encoder.class_name = config.components.vision_encoder.class_name.replace("mllm_engine.models", "lmms_eval.models.docvision_utils")
            config.components.lm.class_name = config.components.lm.class_name.replace("mllm_engine.models", "lmms_eval.models.docvision_utils")
            config.test.checkpoint_path = pretrained
            return config
        self._config = get_config(pretrained)

        self._batch_size_per_gpu = int(batch_size)
        self._image_token = self.config.components.lm.image_token
        self._eos_token = self.config.test.get("stop_token", None)
        assert self._eos_token is not None, "test.stop_token (e.g., <|im_end|>) in training_config.yaml is not set!"

        # 2. accelerator 초기화
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if self.accelerator.is_local_main_process:
            eval_logger.info(f"Using {self.accelerator.num_processes} devices with data parallelism")
        self._world_size = self.accelerator.num_processes
        self._rank = self.accelerator.local_process_index
        self._device = torch.device(f"cuda:{self.rank}")

        # 3. DocVision model, tokenizer, image_processor 초기화
        self._model, self._tokenizer, self._image_processor, _ = init_model(
            cfg=self.config,
            resume_from_checkpoint=True,
            checkpoint_path=pretrained,
            test=True,
        )
        self._model.eval()
        self._eos_token_id = self.tokenizer.encode(self._eos_token)[1]

        # 4. DeepSpeed 설정 초기화 및 accelerator 적용
        kwargs = {
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
            "train_batch_size": self.batch_size_per_gpu * self.world_size,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1.0,
        }
        AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
        self._model = self.accelerator.prepare(self.model)
        eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and **set zero stage to 0**")

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self._eos_token_id

    @property
    def image_token(self):
        return self._image_token

    @property
    def image_processor(self):
        return self._image_processor

    @property
    def batch_size_per_gpu(self):
        return self._batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _load_image_tensor(self, image_list: List[Image.Image]) -> torch.Tensor:
        tensor = self.image_processor(image_list, return_tensors="pt")
        image_tensors, image_sizes = tensor["pixel_values"], tensor["image_sizes"]
        images_used = [True] * len(image_list)
        return image_tensors, image_sizes, images_used

    def process_sample(self, images: List[torch.Tensor], question: str) -> Dict[str, Any]:
        if len(images) == 0:
            raise ValueError("Input image_list is empty")

        # 1. Process input image (single or multipage)
        #   - When input is multipage, each page will be processed on their best grid shape and padded to the largest grid shape,
        #     which will be processed properly during model inference.
        #   - pixel_values tensor has the shape:
        #       [num_of_pages, num_of_largest_grid_patches, 3, height, width]
        pixel_values, image_sizes, image_used = self._load_image_tensor(images)

        # 2. Process input text
        #   - mllm_engine/data/utils.py:tokenizing_for_instruction()
        #   - lmms_eval/models/llava.py:generate_until()
        if self.image_token not in question:
            image_tokens = [self.image_token] * len(images)
            question = " ".join(image_tokens) + "\n" + question
        conv = [{"role": "user", "content": question}]
        role_text = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        input_tokens = self.tokenizer(role_text, add_special_tokens=False, return_tensors="pt")

        return {
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "input_ids": input_tokens["input_ids"][0],
            "attention_mask": input_tokens["attention_mask"][0],
        }

    def pad_sequence(self, input_ids, batch_first, padding_value):
        # left-padding
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        input_ids = torch.flip(input_ids, [1])
        return input_ids

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented loglikelihood() method yet."

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size_per_gpu, batch_fn=None)
        num_iters = len(requests) // self.batch_size_per_gpu if len(requests) % self.batch_size_per_gpu == 0 else len(requests) // self.batch_size_per_gpu + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"

            # 1. 입력 데이터 전처리
            #   - 원래 DocVision 의 Dataset 구현에 있던 기능들을 가져옴
            processed_batch = []
            for images, question in zip(visuals, contexts):
                processed_batch.append(self.process_sample(images, question))

            # 2. generate 에 필요한 파라미터 설정
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if ("eos_token_id" not in gen_kwargs) and (self.eot_token_id is not None):
                gen_kwargs["eos_token_id"] = self.eot_token_id

            # 3. 배치 형태로 변형
            input_ids_list = [s["input_ids"] for s in processed_batch]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eot_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            pixel_values = torch.cat([s["pixel_values"] for s in processed_batch], dim=0).to(dtype=torch.bfloat16, device=self.device)
            image_sizes = torch.cat([s["image_sizes"] for s in processed_batch], dim=0).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...

            # 4. 결과 생성 고고!
            try:
                lm_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    pad_token_id=pad_token_ids,
                    eos_token_id=gen_kwargs["eos_token_id"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    use_cache=True,
                )
                text_outputs = self.tokenizer.batch_decode(
                    lm_output[:, input_ids.shape[-1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (question, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for InstructBlip")
