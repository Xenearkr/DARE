from opencompass.models import LLaDA2Dot1MoEModel

models = [
    dict(
        type=LLaDA2Dot1MoEModel,
        abbr='llada2.1-mini',
        path='/TO/YOUR/PATH',
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
