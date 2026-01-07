from typing import Dict, List, Optional

import numpy as np

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

import sglang as sgl

DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True)


class SGLangModel(BaseModel):
    """Model Wrapper for SGLang (Offline Engine API)."""

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        model_kwargs: dict = None,
        generation_kwargs: dict = dict(temperature=0.0, max_new_tokens=1024),  # defalut generation params
        dllm_algorithm: str = "LowConfidence",   # default dllm algorithm
        dllm_algorithm_config: dict = None,  # not used
        meta_template: Optional[Dict] = None,
        use_fastchat_template: bool = False,  # keep the same signature; unused here
        lora_path: str = None,
        stop_words: List[str] = [],
        tokenizer_path: Optional[str] = None,
    ):
        super().__init__(path=path, max_seq_len=max_seq_len, meta_template=meta_template)

        assert sgl is not None, (
            "Please follow sglang official docs to install sglang"
        )

        self.logger = get_logger()
        self.dllm_algorithm = dllm_algorithm
        self.dllm_algorithm_config = dllm_algorithm_config or {}
        self.model_kwargs = model_kwargs
        self.model_kwargs.update({
            "dllm_algorithm": self.dllm_algorithm,
        })
        
        self._load_model(path, self.model_kwargs)

        self.tokenizer = None
        try:
            from transformers import AutoTokenizer

            tok_path = tokenizer_path or path
            trc = (model_kwargs or {}).get("trust_remote_code", True)
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=trc)
        except Exception as e:
            self.logger.warning(
                f"[SGLang] Failed to init tokenizer automatically. "
                f"Some features (mode=mid, get_token_len) may not work. err={e}"
            )
            self.tokenizer = None

        self.generation_kwargs = generation_kwargs or {}
        for k in ["do_sample", "top_k", "repetition_penalty"]:
            self.generation_kwargs.pop(k, None)    # align with vllm wrapper

        self.use_fastchat_template = use_fastchat_template
        self.stop_words = stop_words or []
        self.lora_path = lora_path

    def _load_model(self, path: str, add_model_kwargs: dict = None, num_retry: int = 3):
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        if add_model_kwargs is not None:
            model_kwargs.update(add_model_kwargs)
        # SGLang Offline Engine:
        # Some models may need impl="transformers" and trust_remote_code=True, etc.
        self.model = sgl.Engine(
            model_path=path, 
            max_running_requests=1,          # OpenCompass single batch
            mem_fraction_static=0.8,          # avoid gpu oom                    
            **model_kwargs)

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        stopping_criteria: List[str] = [],
        **kwargs,
    ) -> List[str]:
        """Generate results given a list of inputs."""

        messages = _convert_chat_messages(inputs)
        prompts = [self.tokenizer.apply_chat_template(m_i, add_generation_prompt=True, tokenize=False) for m_i in messages]
        
        generation_kwargs = {}
        generation_kwargs.update(self.generation_kwargs)
        generation_kwargs.update(kwargs)
        print(f"[SGLang] generation_kwargs: {generation_kwargs}")
        print(f"[SGLang] dllm kwargs: Algorothm: {self.dllm_algorithm}")

        # SGLang sampling params use `max_new_tokens` (not max_tokens).
        sampling_params = dict(generation_kwargs)
        sampling_params["max_new_tokens"] = max_out_len

        _stop = list(set((self.stop_words or []) + (stopping_criteria or [])))
        if _stop:
            sampling_params["stop"] = _stop

        # Offline Engine API: outputs = llm.generate(prompts, sampling_params)
        # LoRA: server /generate supports lora_path; offline engine may accept lora_path kw.
        try:
            if self.lora_path:
                outputs = self.model.generate(prompts, sampling_params, lora_path=self.lora_path)
            else:
                outputs = self.model.generate(prompts, sampling_params)
        except TypeError:
            # fallback: if Engine.generate doesn't accept lora_path kw, ignore
            outputs = self.model.generate(prompts, sampling_params)

        return [o.get("text", "") for o in outputs]


    ### note that sglang official does not guarant the correctness of prompt logprobs, we just implement here but not recommend to use it right now.
    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """Compute (approx) cross-entropy loss from returned logprobs.
        Note: SGLang logprob return behavior has known caveats in some versions.
        We do best-effort parsing from meta_info.input_token_logprobs.
        """
        if self.tokenizer is None:
            raise RuntimeError("[SGLang] get_ppl needs tokenizer")

        # We want input token logprobs; SGLang /generate supports return_logprob and logprob_start_len. :contentReference[oaicite:7]{index=7}
        # Try to request logprobs for prompt tokens.
        sampling_params = dict(self.generation_kwargs)
        sampling_params["max_new_tokens"] = 1  # keep it minimal; focus on prompt
        sampling_params["temperature"] = 0.0  # deterministic; not required but stable

        outputs = self.model.generate(
            inputs,
            sampling_params,
            return_logprob=True,
            logprob_start_len=0,  # attempt to include prompt tokens
        )

        ce_loss = []
        for i, out in enumerate(outputs):
            meta = out.get("meta_info", {}) or {}
            input_token_logprobs = meta.get("input_token_logprobs", []) or []

            # Each entry often looks like [logprob, token_id, extra]
            # Some versions may insert None; we filter them.
            lp = []
            for triple in input_token_logprobs:
                if not isinstance(triple, (list, tuple)) or len(triple) < 1:
                    continue
                logp = triple[0]
                if logp is None:
                    continue
                lp.append(float(logp))

            if not lp:
                # If prompt logprobs are unavailable, we cannot compute ppl reliably.
                raise RuntimeError(
                    "[SGLang] input_token_logprobs is empty. "
                    "Your SGLang version/server may not return prompt logprobs reliably."
                )

            if mask_length is not None:
                lp = lp[-mask_length[i] :]

            loss = -float(np.sum(lp)) / max(len(lp), 1)
            ce_loss.append(loss)

        return np.array(ce_loss)

    def get_loglikelihood(self, inputs: List[str], conts: List[str]) -> List[float]:
        mask_length = [self.get_token_len(c, add_special_tokens=False) for c in conts]
        return -self.get_ppl(inputs, mask_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        """Get length of tokenized string."""
        if self.tokenizer is None:
            raise RuntimeError("[SGLang] tokenizer is not initialized.")
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return len(token_ids)


def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs

if __name__ == "__main__":
    import sglang as sgl
    ###### Basic sglang dllm inference test
    model_path = "####
    llm = sgl.Engine(
        model_path=model_path,
        dllm_algorithm="LowConfidence",
        max_running_requests=1,
        mem_fraction_static=0.8,    
        trust_remote_code=True,
        
    )
    dialogues = [
        [
            {"role": "user", "content": "Write a brief introduction of the great wall"},
        ],
        [
            {"role": "user", "content": "Write a brief introduction of the great wall"},
        ]
    ]
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    prompts = []
    for dialogue in dialogues:
        prompt = tokenizer.apply_chat_template(
            dialogue,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    print(prompts)
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 1024,
    }

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)
