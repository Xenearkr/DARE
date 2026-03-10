from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.aime2025.aime2025_gen_repeat8_5e9f4f import \
        aime2025_datasets
    from ..opencompass.configs.models.dllm.llada2_mini import \
        models as llada2_mini
    from ..opencompass.configs.summarizers.groups.OlympiadBench import \
        OlympiadBenchMath_summary_groups
    
datasets = aime2025_datasets
models = llada2_mini
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)
summary_groups.extend([
    {
        'name': 'AIME2025-Aveage8',
        'subsets':[[f'aime2025-run{idx}', 'accuracy'] for idx in range(8)]
    }
])
summarizer = dict(
    dataset_abbrs=[
        ['AIME2025-Aveage8', 'naive_average'],
    ],
    summary_groups=summary_groups,
)

eval_cfg = {
    'gen_blocksize': 32, 
    'gen_length': 2048, 
    'gen_steps': 32, 
    'batch_size': 1, 
    'batch_size_': 1,
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

