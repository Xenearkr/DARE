"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""

import ast
import json
import time
import multiprocessing
import re
from multiprocessing import Manager
from typing import Any
from dataclasses import dataclass

from .code_utils.humanevalplus import get_num_test_cases
from .code_utils.humanevalplus import run_test as humanevalplus_run_test
from .code_utils.livecodebench import run_test as lcb_run_test
from .code_utils.opencompass_evaluator import HumanEvalEvaluator, HumanEvalPlusEvaluator, MBPPEvaluator

_FENCED_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)
_OPENING_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n?(.*)$", re.DOTALL)
_PYTHON_DEF_RE = re.compile(r"def\s+\w+\s*\(")


def _pick_best_code_block(code_blocks: list[str]) -> str:
    blocks = [b.strip() for b in code_blocks if b and b.strip()]
    if not blocks:
        return ""
    if len(blocks) == 1:
        return blocks[0]
    for block in blocks:
        if _PYTHON_DEF_RE.search(block):
            return block
    return blocks[-1]


def _extract_evalplus_completion(text: str) -> str | None:
    """EvalPlus instruct: opening ```python is in the prompt; response is a continuation."""
    body = text.strip()
    if not body:
        return None

    if body.endswith("```"):
        body = body[: body.rfind("```")].rstrip()

    # Drop trailing markdown close when model repeats the fence without an opener.
    if _PYTHON_DEF_RE.search(body) or re.search(r"^(from|import)\s+\w+", body, re.MULTILINE):
        return body.strip()
    return None


def extract_code_from_model(model_response: str):
    """
    Extract Python code from an LLM response.

    Supports:
    - Full markdown fences in the response (```python ... ```)
    - EvalPlus/Dream-Coder instruct continuations (fence opened in prompt; response
      is raw code, optionally with a trailing ``` only)
    """
    if not model_response or not model_response.strip():
        return None

    text = model_response.strip()
    code_blocks = _FENCED_BLOCK_RE.findall(text)
    if code_blocks:
        picked = _pick_best_code_block(code_blocks)
        return picked or None

    opening = _OPENING_FENCE_RE.match(text)
    if opening:
        picked = _pick_best_code_block([opening.group(1)])
        return picked or None

    completion = _extract_evalplus_completion(text)
    if completion:
        return completion

    return None


def extract_code_from_model_mbpp(model_response: str):
    """
    Extracts the code from MBPP model response using patterns similar to OpenCompass MBPPEvaluator.
    This function handles various MBPP-specific response formats including [BEGIN]/[DONE] markers.

    Parameters:
        model_response (str): The text output from the LLM for MBPP dataset.

    Returns:
        str: The extracted code, or None if no code is found.
    """
    patterns = [
        r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
        r"BEGIN\s*'(.*)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)'\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
        r"BEGIN\s*'(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)\s*DONE",
        r"BEGIN\s*'(.*)\s*DONE",
        r'\[BEGIN\]\s*(.*)\s*\[DONE\]',
        r'BEGIN\s*(.*)\s*\[DONE\]',
        r'\[BEGIN\]\s*(.*)\s*DONE',
        r'BEGIN\s*(.*)\s*DONE',
        r'```python\s*(.*)\s*```',
        r'```\s*(.*)\s*```',
        r'```python\s*(.*)\s*$',
        r'```\s*(.*)\s*$',
        r'(.*)\s*```.*',
        r"\[BEGIN\]\s*'(.*)",
        r'\[BEGIN\](.*)',
        r"'(.*)'\s*\[DONE\]",
    ]
    
    text = model_response
    for p in patterns:
        try:
            match = re.search(p, text, re.DOTALL)
        except TimeoutError:
            match = None

        if match:
            text = match.group(1)
            break
    
    # Additional cleanup
    text = text.split('```')[0]
    text = re.split(r"'?\s*\[?DONE\]?", text)[0]
    text = text.replace('\\_', '_')
    text = text.strip()
    
    # Return None if no meaningful code is found
    if not text or len(text.strip()) < 10:  # Minimum length check
        return None
        
    return text


def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.

    Args:
        code (str): The input Python code.

    Returns:
        str: Cleaned code without the main execution block.
    """
    code_lines = code.split("\n")
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            # Check if we're out of the block (less indentation)
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def postprocess_lcb_sample(sample, metadata):
    if isinstance(sample, str):
        try:
            sample = json.loads(sample)
        except json.JSONDecodeError:
            try:
                sample = ast.literal_eval(sample)
            except (ValueError, SyntaxError) as e:
                print(f"Error: Could not parse sample as JSON or literal: {e}")
                print(f"Sample content: {sample[:200]}...")
                raise ValueError(f"Invalid sample format: {e}")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            try:
                metadata = ast.literal_eval(metadata)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse metadata as JSON or literal: {e}")
                metadata = {}
    
    sample_inputs = [sam["input"] for sam in sample]
    sample_outputs = [sam["output"] for sam in sample]

    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}\n"
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def lcb_check_correctness_v2(sample, generation, metadata, timeout=6, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample, metadata)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)

    detailed_results = {"all_passed": False, "test_results": [], "total_tests": 0, "passed_tests": 0}

    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result.extend([[-1 for i in range(len(in_outs["inputs"]))]])
        detailed_results["total_tests"] = len(in_outs["inputs"])
        detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": False, "error": "global timeout"} for inp, out in zip(in_outs["inputs"], in_outs["outputs"], strict=False)]
        if debug:
            print("global timeout")
        return False, detailed_results

    if not result:
        return False, detailed_results

    # Create detailed test results
    in_outs = json.loads(sample["input_output"])
    detailed_results["total_tests"] = len(result[0])
    detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": res == True, "error": metadata_list[0].get("error", None), "error_message": metadata_list[0].get("error_message", None), "output": metadata_list[0].get("output", None)} for inp, out, res in zip(in_outs["inputs"], in_outs["outputs"], result[0], strict=False)]
    detailed_results["passed_tests"] = sum(1 for r in result[0] if r == True)
    detailed_results["all_passed"] = all(r == True for r in result[0])

    return all(x == True for x in result[0]), detailed_results


def humanevalplus_check_correctness(test: str, code: str, timeout_per_test: int = 1) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all HumanEvalPlus test cases.

    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    code = clean_code_main_block(code)

    num_test_cases = get_num_test_cases(test)
    succ, output = humanevalplus_run_test(code, test, timeout_per_test * num_test_cases)

    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_test_cases, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def humaneval_check_correctness(test: str, code: str, entry_point: str = None, timeout_per_test: int = 1) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all HumanEval test cases.
    Uses the existing HumanEvalPlus testing infrastructure.

    Args:
        test: String of the test file content (assert statements)
        code: Generated code to test
        entry_point: Function name to test (from HumanEval entry_point field)
        timeout_per_test: Maximum execution time in seconds per test

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    code = clean_code_main_block(code)
    
    # Count test cases by counting assert statements in the test
    num_test_cases = 0
    try:
        parsed = ast.parse(test)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Assert):
                num_test_cases += 1
    except SyntaxError:
        num_test_cases = test.count("assert ")
    
    # For HumanEval, we need to append a call to the check function since the test code only defines check() but doesn't call it
    if entry_point:
        test += f"\n\n# Execute the tests\ncheck({entry_point})"
    else:
        func_match = re.search(r"def\s+(\w+)\s*\(", code)
        if func_match:
            func_name = func_match.group(1)
            test += f"\n\n# Execute the tests\ncheck({func_name})"
    
    # For HumanEval, we need to append a call to the check function since the test code only defines check() but doesn't call it
    func_match = re.search(r"def\s+(\w+)\s*\(", code)  # extract function name from the code
    if func_match:
        func_name = func_match.group(1)
        test += f"\n\n# Execute the tests\ncheck({func_name})"
        
    succ, output = humanevalplus_run_test(code, test, timeout_per_test * num_test_cases)
    
    detailed_results = {
        "all_passed": succ, 
        "output": output, 
        "total_tests": num_test_cases,
        "test_results": [{"passed": succ, "output": output}]
    }
    
    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def mbpp_check_correctness(tests: str, code: str, timeout_per_test: int = 1) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all MBPP test cases.
    Uses the existing HumanEvalPlus testing infrastructure.

    Args:
        tests: List of test assertions (String Type)
        code: Generated code to test
        timeout_per_test: Maximum execution time in seconds per test

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    tests = json.loads(tests)
    code = clean_code_main_block(code)

    combined_test = "\n".join(tests)
    succ, output = humanevalplus_run_test(code, combined_test, timeout_per_test * len(tests))
    
    detailed_results = {
        "all_passed": succ,
        "total_tests": len(tests),
        "test_results": [{
            "passed": succ,
            "output": output
        }]
    }
    return succ, detailed_results


def taco_to_lcb_format(tests):
    """
    Given a dictionary with keys "inputs" and "outputs", returns a list of test cases and metadata.
    Each test case is a dictionary with keys "input" and "output". If the lists are unequal,
    missing entries are filled by reusing the first element of the shorter list.

    Args:
        data (dict): A dictionary with keys "inputs" and "outputs", each mapped to a list of strings.

    Returns:
        list of dict: A list where each element is a dict with keys "input" and "output".
        dict: Metadata containing function name if available
    """
    if isinstance(tests, str):
        tests = json.loads(tests)
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    # Determine the number of test cases to create.
    n = max(len(inputs), len(outputs))

    test_cases = []
    metadata = None
    for i in range(n):
        # Use the first element as a fallback if the list is shorter than n.
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        if not inp or not out:  # Filter out empty test cases
            continue
        out = out[0] if isinstance(out, list) else out
        test_case: dict[str, Any] = {"input": inp, "output": out, "metadata": {}}
        if "fn_name" in tests:
            test_case["testtype"] = "functional"
            metadata = {"func_name": tests["fn_name"]}
        test_cases.append(test_case)

    return test_cases, metadata


@dataclass
class RewardConfig:
    # General reward constants
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_error_reward: float = 0.0


class RewardCodeFn:
    """
    Reward function for evaluating code dataset answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the unit tests provided
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str):
        """
        Calculate the reward for a code task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution (code)

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        # total_start_time = time.time()

        model_response = action
        dataset_name = task_info.get("data_source", "")
        tests = task_info.get("ground_truth", None)
        metadata = task_info.get("metadata", None)

        if tests is None:
            print("No tests found in task_info")
            return dict(
                reward=self.config.format_error_reward,
                pass_reward=self.config.format_error_reward,
                is_correct=False,
                metadata={"error": "No tests found in task_info"},
                pred="",
            )

        if dataset_name == "mbpp":
            model_code = extract_code_from_model_mbpp(model_response)
        else:
            model_code = extract_code_from_model(model_response)
        # print("extract code:", model_code)
            
        if model_code is None:
            # print("No code found in model response")
            return dict(
                reward=self.config.format_error_reward,
                pass_reward=self.config.format_error_reward,
                is_correct=False,
                metadata={"error": "No code found in model response"},
                pred="",
            )

        # Tests: List[Dictionary] - Codeforces, LiveCodeBench
        # Tests: Dictionary[Lists] - CodeContests, Taco/Apps
        is_correct = False
        test_details: dict[str, Any] = {}

        if dataset_name in ["taco", "apps", "code_contests"]:
            tests, metadata = taco_to_lcb_format(tests)
            is_correct, test_details = lcb_check_correctness_v2(tests, model_code, metadata, debug=False)
        elif dataset_name in ["lcbv5", "codeforces", "primeintellect"]:
            is_correct, test_details = lcb_check_correctness_v2(tests, model_code, metadata, debug=False)
        elif dataset_name in ["humanevalplus"]:
            is_correct, test_details = humanevalplus_check_correctness(tests, model_code)
        elif dataset_name == "mbpp" or dataset_name == "kodcode":  # MBPP format: tests is a list of assertion strings 
            is_correct, test_details = mbpp_check_correctness(tests, model_code)
            # print("mbpp test_details:", test_details)
        elif dataset_name == "humaneval":
            is_correct, test_details = humaneval_check_correctness(tests, model_code)
            # # print("humaneval test_details:", test_details)
            
            # # Use OpenCompass HumanEval evaluator
            # func_match = re.search(r"def\s+(\w+)\s*\(", model_code)
            # entry_point = func_match.group(1) if func_match else None

            # # Build a minimal problem dict required by the evaluator
            # problem = {
            #     "task_id": "humaneval_task_0",
            #     "prompt": "",
            #     "test": tests,
            #     "entry_point": entry_point if entry_point else "solution",
            # }

            # evaluator = HumanEvalEvaluator()
            # result = evaluator.check_correctness(problem=problem, completion=model_code, timeout=3.0)

            # # Count test cases (approximate by number of assert statements)
            # num_test_cases = 0
            # try:
            #     parsed = ast.parse(tests)
            #     for node in ast.walk(parsed):
            #         if isinstance(node, ast.Assert):
            #             num_test_cases += 1
            # except SyntaxError:
            #     num_test_cases = tests.count("assert ")

            # is_correct = bool(result.get("passed", False))
            # test_details = {
            #     "all_passed": is_correct,
            #     "total_tests": num_test_cases,
            #     "test_results": [{"passed": is_correct, "result": result.get("result")}],
            #     "result": result.get("result"),
            # }
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented")

        # total_time = time.time() - total_start_time
        # print(f"Total reward function execution time: {total_time:.2f} seconds")

        if is_correct:
            pass_reward = self.config.correct_reward
            return dict(
                reward=pass_reward,
                pass_reward=pass_reward,
                is_correct=True,
                metadata=test_details,
                pred=model_code,
            )
        pass_reward = self.config.incorrect_reward
        return dict(
            reward=pass_reward,
            pass_reward=pass_reward,
            is_correct=False,
            metadata=test_details,
            pred=model_code,
        )


def rllm_reward_fn_code(data_source: str, llm_solution: str, ground_truth: dict, extra_info: dict):
    """Evaluate code solutions against ground truth answers

        This function creates a reward function to evaluate code solutions by pass the test_case from groun_truth. It can optionally use a language model
        for more sophisticated answer validation.

        Args:
            data_source: The source/dataset the problem comes from
            llm_solution: The solution string provided by the language model to evaluate
            ground_truth: some tests for this llm_solution
            enable_llm: Whether to enable language model validation for complex cases (default: False)

        Returns:
            tuple: (bool, dict) where:
                - bool: True if the solution passes all the test_case, False otherwise
                - dict: Detailed test results with test cases and pass/fail status

        Example:
                model_response = '''
    import sys
    from itertools import permutations
    def main():
        n,m=map(int, input().split())
        a=sum(list(map(int, input().split())))
        if a+(n-1)*10<=m:
            print(5)
        else:
            print(5)
    if __name__ == "__main__":
        main()
    '''

        print(f"test the code_forces")
        # tests = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ]
        metadata = {
             "tests": tests,
        }
        True, {"all_passed": True, "test_results": [...]}
    """
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)

    # Convert to new format
    task_info = {"problem": None, "problem_type": "code", "data_source": data_source, "ground_truth": ground_truth, **extra_info}

    reward_response = reward_fn(task_info, llm_solution)
    return reward_response
