from opencompass.models import SGLangModel

models = [
    dict(
        type=SGLangModel,
        abbr='llada2.0-mini-sglang',
        path='/TO/YOUR/PATH',
        dllm_algorithm='LowConfidence',
        generation_kwargs={
            "max_new_tokens": 1024,
            "temperature": 0.0,
        },
        max_seq_len=2048,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]