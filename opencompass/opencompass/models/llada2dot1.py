from typing import List

from opencompass.registry import MODELS

from .llada2 import LLaDA2MoEModel, _convert_chat_messages


@MODELS.register_module()
class LLaDA2Dot1MoEModel(LLaDA2MoEModel):
    """Dedicated wrapper for LLaDA2.1 MoE models.

    LLaDA2.1 keeps the same high-level loading interface as LLaDA2.0, but its
    model-side ``generate`` supports extra editing/refinement controls.
    Keeping a separate wrapper lets configs target those semantics explicitly
    without branching inside the LLaDA2.0 wrapper.
    """

    def __init__(self,
                 editing_threshold: float = 0.9,
                 max_post_steps: int = 16,
                 num_to_transfer: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.editing_threshold = editing_threshold
        self.max_post_steps = max_post_steps
        self.num_to_transfer = num_to_transfer

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs."""
        messages = _convert_chat_messages(inputs)
        prompt = [
            self.tokenizer.apply_chat_template(
                m_i, add_generation_prompt=True, tokenize=False)
            for m_i in messages
        ]
        gen_length = min(max_out_len, self.gen_length)
        print('steps:', self.gen_steps, 'length:', gen_length,
              'blocksize:', self.gen_blocksize)
        print('temperature:', self.temperature, 'cfg:', self.cfg)
        print('mask_id:', self.mask_id, 'padding_id:', self.padding_id)
        print('editing_threshold:', self.editing_threshold,
              'max_post_steps:', self.max_post_steps,
              'num_to_transfer:', self.num_to_transfer)
        print('final prompt:', prompt)
        responses = []
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        try:
            for single_prompt in prompt:
                tokenized = self.tokenizer(single_prompt, return_tensors='pt')
                input_ids = tokenized['input_ids'].to(self.model.device)

                generated = self.model.generate(
                    inputs=input_ids,
                    temperature=self.temperature,
                    block_length=self.gen_blocksize,
                    steps=self.gen_steps,
                    gen_length=gen_length,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    eos_early_stop=self.eos_early_stop,
                    minimal_topk=self.minimal_topk,
                    threshold=self.threshold,
                    editing_threshold=self.editing_threshold,
                    max_post_steps=self.max_post_steps,
                    eos_id=self.padding_id,
                    mask_id=self.mask_id,
                    num_to_transfer=self.num_to_transfer,
                )
                responses.append(
                    self.tokenizer.decode(
                        generated[0], skip_special_tokens=True))
        finally:
            self.tokenizer.padding_side = original_padding_side

        print('--------------------')
        for i, response in enumerate(responses):
            print(f'Response {i}:', response)
            print('====================')
        print('--------------------')
        return responses
