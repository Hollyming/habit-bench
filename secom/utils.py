# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
import os.path as osp
import re
from datetime import datetime
from time import sleep

from dotenv import load_dotenv
from vllm import LLM, SamplingParams


class OpenAILLM:
    def __init__(self, model_name="gpt-4o-mini-2024-07-18"):
        from openai import OpenAI

        self.model_name = model_name

        self.client = OpenAI()
        self.client.base_url = os.getenv("OPENAI_API_BASE")
        keys = os.getenv("OPENAI_API_KEY")
        self.client.api_key = keys.split(",")[0]

    def __call__(
        self,
        prompt,
        system_prompt=None,
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        seed=42,
        max_num_retries=2,
        return_full=False,
    ) -> str:
        if system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        retry = 0
        last_error = None
        token_limit = max_tokens
        external_api = os.environ.get("HABITBENCH_INFERENCE_BACKEND") == "external-openai-compatible"
        if external_api:
            max_num_retries = max(max_num_retries, 5)
        while retry < max_num_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=token_limit,
                    seed=seed,
                )
                content = completion.choices[0].message.content
                finish_reason = completion.choices[0].finish_reason
                if not isinstance(content, str) or not content.strip() or finish_reason == "length":
                    # Reasoning is received by the SDK but is never substituted
                    # for the formal segmentation. Retry with completion room.
                    if external_api:
                        token_limit = min(max(token_limit * 2, 4096), 16384)
                    raise ValueError(f"Unusable formal SeCom response: finish_reason={finish_reason}, "
                                     f"content_chars={len(content) if isinstance(content, str) else 0}")
                if not return_full:
                    return content

                ret_dict = {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "model_name": self.model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response": content,
                    "response_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    # "completion_obj": completion,
                }
                return ret_dict

            except Exception as e:
                last_error = e
                retry += 1
                if retry < max_num_retries:
                    sleep(5)
                print(f"Error: {e}", flush=True)

        raise RuntimeError(
            "Calling OpenAI failed after retrying for " f"{retry} times."
        ) from last_error


class LocalLLM:
    def __init__(self, model_name_or_path):
        self.model = LLM(model=model_name_or_path)

    def __call__(
        self,
        prompt: str,
        temperature: float = 0.9,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        seed: int = 42,
    ):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
        )
        outputs = self.model.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text


def extract_result(text, tag="tag"):
    if not isinstance(text, str):
        return "", False
    pattern = rf"<{tag}>([\s\S]*?)<\/{tag}>"
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        return "", False
    else:
        return matches[0], True


def extract_yes_no(text):
    is_yes = False
    match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if match:
        is_yes = match.group(0).lower() == "yes"
    return is_yes
