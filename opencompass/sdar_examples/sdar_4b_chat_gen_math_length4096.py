from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5 import \
        math_datasets
    from ..opencompass.configs.models.dllm.sdar_4b_chat import \
        models as sdar_4b_chat
datasets = math_datasets
models = sdar_4b_chat
summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
eval_cfg = {
    'gen_length': 4096,
    'block_length': 4,
    'denoising_steps': 4, 
    'batch_size': 1, 
    'batch_size_': 1,
    'model_kwargs': {
        'attn_implementation': 'flash_attention_2',  #'sdpa'
        'torch_dtype': 'bfloat16',
        'device_map': 'auto',
        'trust_remote_code': True,
    },
    'temperature': 1.0,
    'top_k': 1, 
    'top_p': 1.0,
    'confidence_threshold': 1.0,
    'remasking': 'low_confidence_static',
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

