from opencompass.models import LLaDA2MoEModel

models = [
    dict(
        type=LLaDA2MoEModel,
        abbr='llada2-mini',
        path='/TO/YOUR/PATH',
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
