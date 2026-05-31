import numpy as np

from verl.trainer.ppo.metric_utils import *


def configure_wandb_dllm_metrics(tracking) -> None:
    """Group related DLLM training metrics into shared W&B panels."""
    wandb = getattr(tracking, "logger", {}).get("wandb")
    if wandb is None:
        return
    step = "training/global_step"
    wandb.define_metric(step)
    wandb.define_metric("rollout/*", step_metric=step)
    wandb.define_metric("cfl/*", step_metric=step)
    wandb.define_metric("pass_reward/*", step_metric=step)
    wandb.define_metric("reward/*", step_metric=step)
    wandb.define_metric("tpf/*", step_metric=step)


def compute_reward_extra_metrics(batch) -> dict[str, float]:
    """Aggregate pass_reward, reward, TPF and rollout decode stats for W&B."""
    nt = getattr(batch, "non_tensor_batch", None) or {}
    metrics: dict[str, float] = {}

    def _collect(key: str, prefix: str):
        if key not in nt:
            return
        vals = np.asarray(nt[key], dtype=np.float64)
        if vals.size == 0:
            return
        metrics[f"{prefix}/mean"] = float(np.mean(vals))
        metrics[f"{prefix}/max"] = float(np.max(vals))
        metrics[f"{prefix}/min"] = float(np.min(vals))

    _collect("pass_reward", "pass_reward")
    _collect("reward", "reward")
    _collect("efficiency_reward", "efficiency_reward")
    _collect("tpf", "tpf")
    _collect("tpf_baseline", "tpf_baseline")

    if "tpf" in nt and "acc" in nt:
        tpf = np.asarray(nt["tpf"], dtype=np.float64)
        acc = np.asarray(nt["acc"], dtype=np.float64)
        passed = tpf[(acc > 0) & (tpf > 0)]
        if passed.size > 0:
            metrics["tpf_passed/mean"] = float(np.mean(passed))

    # Shared W&B panel "rollout": nfe and gen_tokens as two series (no rollout_tpf).
    if "rollout_nfe" in nt:
        nfe = np.asarray(nt["rollout_nfe"], dtype=np.float64)
        if nfe.size > 0:
            metrics["rollout/nfe"] = float(np.mean(nfe))
    if "rollout_gen_tokens" in nt:
        gen = np.asarray(nt["rollout_gen_tokens"], dtype=np.float64)
        if gen.size > 0:
            metrics["rollout/gen_tokens"] = float(np.mean(gen))

    return metrics


def calc_pass_at_k(var_vals, k, n_bootstrap=10, seed=42):
    """
    Calculate pass@k: Randomly sample k values from var_vals without replacement, count as correct if at least one is 1, repeat n_bootstrap times and take the mean.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_bootstrap):
        chosen_samples = rng.choice(var_vals, size=min(k, len(var_vals)), replace=False)
        results.append(int(np.any(np.array(chosen_samples) == 1)))
    return float(np.mean(results))


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.
    
    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.
    
    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }
        
        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)
        
    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    # Group all sampling results by [data source][prompt][variable name] in three nested levels
    # For example: 8 samples for the same prompt will be grouped under the same data_source and prompt, with variable names like "score", "pred", etc.
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():  # LNY: N samples for the same prompt
                if isinstance(var_vals[0], str):  # Skip string type (e.g. "pred"), only count numerical type
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                # ==== Add pass@1 and pass@N ====
                metric["pass@1"] = calc_pass_at_k(var_vals, 1, n_bootstrap=10, seed=seed)
                metric[f"pass@{n_resps}"] = calc_pass_at_k(var_vals, n_resps, n_bootstrap=10, seed=seed)
                # ==============================

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
