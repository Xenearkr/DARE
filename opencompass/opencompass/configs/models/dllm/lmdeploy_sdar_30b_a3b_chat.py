from opencompass.models import TurboMindModel

models = [
    dict(
        type=TurboMindModel,
        abbr='sdar-30b-a3b-chat-turbomind',
        path='/TO/YOUR/PATH',
        engine_config=dict(
            session_len=8192, max_batch_size=1, tp=8, dtype="float16",
            max_prefill_token_num=4096,
            cache_max_entry_count=0.8,
            dllm_block_length=4,
            dllm_denoising_steps=4,
            dllm_unmasking_strategy="low_confidence_dynamic",
            dllm_confidence_threshold=0.9,
        ),
        gen_config=dict(top_k=50, temperature=1.0, top_p=0.95, do_sample=False, max_new_tokens=4096),
        max_seq_len=8192,
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=8),
    )
]
