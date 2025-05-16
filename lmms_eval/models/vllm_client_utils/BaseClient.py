"""
BaseClient는 OpenAI API를 사용하여 모델을 호출하는 기본 클래스입니다.
본 코드는 VLM_LLM_IE 모델에서 인퍼런스하기 위한 모듈로 사용됩니다.

ref: https://github.com/UpstageAI/docev-data-engine/blob/main/layout_viewer/viewer_inference_util/BaseClient.py
- structured output 처리를 위한 기능 추가 및 불필요한 함수 제거 적용
"""

from typing import List, Union
from openai import OpenAI
import base64

class BaseClient:
    def __init__(self, model_name: str, host: str, port: int, use_system_prompt: bool = False, prompt_image_type: int = 1, max_completion_tokens: int = 4096):
        self.MODEL = model_name
        self.use_system_prompt = use_system_prompt
        self.prompt_image_type = prompt_image_type
        self.client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="token-abc123",
        )
        self.max_completion_tokens = max_completion_tokens
    
    def _convert_image_url_to_chat_completion_format(self, base64_image:str) -> str:
        """
        Convert image url to chat completion format
        """
        if self.prompt_image_type == 1:
            return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
            }
        else:
            raise ValueError(f"Invalid prompt image type: {self.prompt_image_type}")

    def _make_prompt(self, system_prompt:str, user_prompt:str, image_url_list:List[str], guided_json:Union[dict, None]=None):
        messages = []
        if system_prompt and self.use_system_prompt:
            messages.append({
                "role": "system", 
                "content": system_prompt
            })
        
        user_content = []
        if len(image_url_list) > 0 and self.prompt_image_type != -1:
            for image_url in image_url_list:
                base64_image = encode_base64_content_from_image(image_url)
                user_content.append(self._convert_image_url_to_chat_completion_format(base64_image))

        user_content.append({
            "type": "text",
            "text": user_prompt
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        if guided_json is not None:
            extra_body = {
                "guided_json": guided_json
            }
        else:
            extra_body = None   
        return messages, extra_body

    def _chat_completion(self, system_prompt:str, user_prompt:str, image_url_list:List[str], guided_json:Union[dict, None]=None):
        messages, extra_body = self._make_prompt(system_prompt, user_prompt, image_url_list, guided_json)

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.MODEL,
            max_completion_tokens=self.max_completion_tokens,
            temperature=0,
            extra_body=extra_body
        )

        return chat_completion.choices[0].message.content


    def run(self, system_prompt:str, user_prompt:str, image_url_list:List[str], guided_json:Union[dict, None]=None) -> None:
        result = self._chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, image_url_list=image_url_list, guided_json=guided_json)
        return result
    
def encode_base64_content_from_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')