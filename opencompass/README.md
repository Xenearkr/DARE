## Evaluation Guideline

### ⚙️ Evaluation Hyper-Parameters

You need modify the evaluation hyper-parameters, such as ```gen_length```, ```block_length``` , ```steps```, etc. in ```xxx_examples``` dir. Specifically, the ```eval_cfg``` of ```DARE/opencompass/dream_examples```, ```DARE/opencompass/dream_examples``` and ```DARE/opencompass/dream_examples``` ...


### 📂 Data Preparation

You can choose one for the following method to prepare datasets.

#### Offline Preparation

You can download and extract the datasets with the following commands:

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

#### Automatic Download from OpenCompass

We have supported download datasets automatic from the OpenCompass storage server. You can run the evaluation with extra `--dry-run` to download these datasets.
Currently, the supported datasets are listed in [here](https://github.com/open-compass/opencompass/blob/main/opencompass/utils/datasets_info.py#L259). More datasets will be uploaded recently.

#### (Optional) Automatic Download with ModelScope

Also you can use the [ModelScope](www.modelscope.cn) to load the datasets on demand.

Installation:

```bash
pip install modelscope[framework]
export DATASET_SOURCE=ModelScope
```

Then submit the evaluation task without downloading all the data to your local disk. Available datasets include:

```bash
humaneval, triviaqa, commonsenseqa, tydiqa, strategyqa, cmmlu, lambada, piqa, ceval, math, LCSTS, Xsum, winogrande, openbookqa, AGIEval, gsm8k, nq, race, siqa, mbpp, mmlu, hellaswag, ARC, BBH, xstory_cloze, summedits, GAOKAO-BENCH, OCNLI, cmnli
```

Some third-party features, like Humaneval and Llama, may require additional steps to work properly, for detailed steps please refer to the [Installation Guide](https://opencompass.readthedocs.io/en/latest/get_started/installation.html).

<p align="right"><a href="#top">🔝Back to top</a></p>


### 📍 Speciallized Installation for DLLMs Support

For LLaDA2.0 series inference, you need to ensure:

```bash
transformers>=4.54.0
```

For SGLang inference support of LLaDA2.0, you need to install sglang as follows:

```bash
# Use the at least v0.5.7 branch
git clone -b v0.5.7 https://github.com/sgl-project/sglang.git
cd sglang

# Install the python packages
pip install --upgrade pip
pip install -e "python"
```

For more installation, please refer to [SGLang Docs](https://docs.sglang.io/get_started/install.html)



### 🗂️ Evaluation on Local Benchmark

You can evaluate the local benchmark by running the script `scripts/eval_local_bench.sh`. Before that, you need to refer to `verl.utils.preprocess.preprocess` to organize the local benchmark into ```.parquet``` format. (In fact, the backend of the local evaluation essentially launches a training script, but with `val_only=True` specified)


### 🧰 Evaluation Extension

If you want to add more models:
- Define a new ```model.py``` in ```DARE/opencompass/opencompass/configs/models/dllm```
- Define a modelclass (derived from BaseModel) for new model ```DARE/opencompass/opencompass/models```
- Add ```from .model import ModelClass, BaseModelClass  # noqa: F401``` in ```DARE/opencompass/opencompass/models/__init__.py```
- *Optional, define a generate.py for new model generation utils in ```DARE/opencompass/opencompass/models```

If you want to add more evaluation datasets:
- Refer to [opencompass docs](https://github.com/open-compass/opencompass/blob/main/README.md)
- If you want to use local datasets (e.g., .parquet format), refer to [DARE/scripts/preprocess_dataset.sh](https://github.com/yjyddq/DARE/blob/main/scripts/preprocess_dataset.sh) and [DARE/scripts/eval_local_hf.sh](https://github.com/yjyddq/DARE/blob/main/scripts/eval_local_hf.sh)