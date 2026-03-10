from opencompass.models import SGLangModel

models = [
    dict(
        type=SGLangModel,
        abbr='llada2.0-mini-sglang',
        path='/TO/YOUR/PATH',
        dllm_algorithm='LowConfidence',
        dllm_algorithm_config=dict(
            block_size=32,
            threshold=0.95,
        ),
        model_kwargs=dict(
            trust_remote_code=True,
            attention_backend='flashinfer',
        ),
        generation_kwargs=dict(
            temperature=0.0,
            top_p=0.95,
            top_k=50,
            max_new_tokens=1024,
        ),
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]
