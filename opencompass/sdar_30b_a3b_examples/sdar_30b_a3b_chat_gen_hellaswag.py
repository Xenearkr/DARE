from mmengine.config import read_base

with read_base():
    from ..opencompass.configs.datasets.hellaswag.hellaswag_gen import \
        hellaswag_datasets
    from ..opencompass.configs.models.dllm.sdar_30b_a3b_chat import \
        models as sdar_30b_a3b_chat

datasets = hellaswag_datasets
models = sdar_30b_a3b_chat

summarizer = dict(
    summary_groups=sum([v for k, v in locals().items()
                        if k.endswith('_summary_groups')], []),
)

confidence_threshold = 0.95
if 0 < confidence_threshold < 1.0:
    dllm_unmasking_strategy = 'low_confidence_dynamic'
elif confidence_threshold == 1.0:
    dllm_unmasking_strategy = 'low_confidence_static'

eval_cfg = {
    'gen_length': 4096,
    'block_length': 4,
    'denoising_steps': 4, 
    'batch_size': 1,
    'batch_size_': 1,
    'model_kwargs': {
        'attn_implementation': 'flash_attention_2',
        'torch_dtype': 'torch.bfloat16',
        'device_map': 'auto',
        'trust_remote_code': True,
    },
    'temperature': 1.0,
    'top_k': 50,
    'top_p': 0.95,
    'confidence_threshold': confidence_threshold,
    'remasking': dllm_unmasking_strategy,
}

for model in models:
    model.update(eval_cfg)

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        _scope_='opencompass',
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