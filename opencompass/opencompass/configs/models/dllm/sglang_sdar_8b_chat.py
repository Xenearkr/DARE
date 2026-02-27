from opencompass.models import SGLangModel

models = [
    dict(
        type=SGLangModel,
        abbr='sdar-8b-chat-sglang',
        path='/TO/YOUR/PATH',
        dllm_algorithm='LowConfidence',
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
        max_seq_len=2048,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]
