from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.aime2024.aime2024_gen_repeat8_6e39a4 import \
        aime2024_datasets
    from ..opencompass.configs.models.dllm.sglang_sdar_1dot7b_chat import \
        models as sglang_sdar_1dot7b_chat
    from ..opencompass.configs.summarizers.groups.OlympiadBench import \
        OlympiadBenchMath_summary_groups
datasets = aime2024_datasets
models = sglang_sdar_1dot7b_chat
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)
summary_groups.extend([
    {
        'name': 'AIME2024-Aveage8',
        'subsets':[[f'aime2024-run{idx}', 'accuracy'] for idx in range(8)]
    }
])
summarizer = dict(
    dataset_abbrs=[
        ['AIME2024-Aveage8', 'naive_average'],
    ],
    summary_groups=summary_groups,
)
eval_cfg = {
    'model_kwargs': {
        'dllm_algorithm': 'LowConfidence',
        'mem_fraction_static': 0.6,
        'max_running_requests': 1,
        'attention_backend': 'flashinfer',
        'trust_remote_code': True,
    },
    'generation_kwargs': {
        'temperature': 1.0,
        'top_p': 1.0,
        'top_k': 50,
        'max_new_tokens': 4096,
    },
    'max_seq_len': 8192,
    'max_out_len': 4096,
    'batch_size': 4,
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
