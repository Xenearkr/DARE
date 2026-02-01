from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import \
        gpqa_datasets
    from ..opencompass.configs.models.dllm.lmdeploy_sdar_8b_chat import \
        models as lmdeploy_sdar_8b_chat
datasets = gpqa_datasets
models = lmdeploy_sdar_8b_chat
confidence_threshold = 0.95
if 0 < confidence_threshold < 1.0:
    dllm_unmasking_strategy = "low_confidence_dynamic"
elif confidence_threshold == 1.0:
    dllm_unmasking_strategy = "low_confidence_static"
eval_cfg = {
    'engine_config': {
        'session_len': 8192, 
        'max_batch_size': 16, 
        'tp': 1,
        'dtype': "float16",
        'max_prefill_token_num': 4096,
        'cache_max_entry_count': 0.8,
        'dllm_block_length': 4,
        'dllm_denoising_steps': 4,
        'dllm_unmasking_strategy': dllm_unmasking_strategy,
        'dllm_confidence_threshold': confidence_threshold,
    },
    'gen_config': {
        'top_k': 50, 
        'temperature': 1.0, 
        'top_p': 0.95, 
        'do_sample': False, 
        'max_new_tokens': 4096,
    },
    'max_seq_len': 8192,
    'max_out_len': 4096,
    'batch_size': 16,
}

for model in models:
    model.update(eval_cfg)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,  
        num_split=None,  
        min_task_size=16,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)
