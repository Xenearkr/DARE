from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import \
        humaneval_datasets
    from ..opencompass.configs.models.dllm.llada2_mini import \
        models as llada2_mini
datasets = humaneval_datasets
models = llada2_mini
eval_cfg = {
    'gen_blocksize': 32, 
    'gen_length': 512, 
    'gen_steps': 32, 
    'batch_size_': 1, 
    'batch_size': 1,
    'model_kwargs': {
        'attn_implementation': 'sdpa',  #'sdpa'
        'torch_dtype': 'bfloat16',
        'device_map': 'auto',
    },
    "temperature": 0.0,
    "threshold": 0.95,
    "minimal_topk": 1.0,
    "eos_early_stop": False
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