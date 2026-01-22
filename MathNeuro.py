import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="Huggingface model to train, entered as string", type = str)
parser.add_argument('--eval_datasets', nargs='+', help="dataset(s) to evaluate models on post pruning to evalaute catastrophic forgetting, entered as strings; should be task names from Eleuther AI LM Evaluation Harness", type = str)
parser.add_argument('--train_dataset', help="path to math train dataset; should be a path to a CSV file with question/solution pairs in a columns titled 'question' and 'solution' along with ground-truth answers in a column called 'answer'", type = str)
parser.add_argument('--calibration_datasets', nargs='+', help="path to calibration datasets; should be paths to CSV files with instruction/response pairs in a column titled 'qa'", type = str)
parser.add_argument('--save_path', help="save path for eval results after running Eleuther AI LM Evaluation Harness post pruning", type = str)
parser.add_argument('--text_file', help="name of text file for saving pruning results during training if evaluating math reasoning using a non-Eleuther AI LM Evaluation Harness task in a PoT format", type = str)
parser.add_argument('--num_repeats', help="number of repeats for pruning or scaling experiment", type = int, default = 5)
parser.add_argument('--pre_train_eval', help="bool to indicate if full evaluation on eval and train datasets should be conducted before training", action="store_true")
parser.add_argument('--random_state', help="random state for initial dataset shuffling and creating train/eval split for train dataset", type = int, default = 42)
parser.add_argument('--scalar', help="scale factor for top parameters; default is 0 to run pruning experiments", type = float, default = 0)
parser.add_argument('--eval_dataset_size', help="desired number of samples for task specific eval dataset", type = int, default = None)
parser.add_argument('--eval_dataset_subset', help="desired number of samples for task specific eval dataset if subsetting to reduce run time", type = int, default = 100)
parser.add_argument('--train_dataset_size', help="desired number of samples to use from the training dataset CSV", type = int, default = None)
parser.add_argument('--calibration_dataset_size', help="desired number of samples to use from each calibration dataset CSV", type = int, default = None)
parser.add_argument('--calibration_dataset_names', nargs='+', help="desired name of calibration datasets; should be strings entered in same order as calibration_datasets", type = str)
parser.add_argument('--num_samples', help="desired number of samples for calculating task specific parameters", type = int, default = 500)
parser.add_argument('--eval_batch_size', help="batch size for evaluation and activation collection", type = int, default = 4)
parser.add_argument('--train_lm_eval_task', nargs='?', help="if your training dataset is an Eleuther AI LM Evaluation Harness task, specify the associated task for the test set.", type = str, default = None)
parser.add_argument('--train_task_only_pre', help="run train_lm_eval_task only during pre-train eval (skip during pruning runs)", action="store_true")
parser.add_argument('--proportion', help="desired proportion of top parameters to calculate", type = float, default = None)
parser.add_argument('--streetmath_eval', help="run StreetMath evaluation using the current model weights", action="store_true")
parser.add_argument('--streetmath_jsonl', help="path to StreetMath JSONL file", type = str, default = None)
parser.add_argument('--streetmath_root', help="path to StreetMathDataset root (contains streetmath_benchmark/ and data/)", type = str, default = None)
parser.add_argument('--streetmath_limit', help="limit StreetMath samples", type = int, default = None)
parser.add_argument('--streetmath_eval_size', help="limit StreetMath samples (alias of --streetmath_limit)", type = int, default = None)
parser.add_argument('--streetmath_batch_size', help="batch size for StreetMath generation", type = int, default = 1)
parser.add_argument('--streetmath_max_tokens', help="max new tokens for StreetMath generation", type = int, default = 256)
parser.add_argument('--streetmath_temperature', help="temperature for StreetMath generation", type = float, default = 0.2)
parser.add_argument('--streetmath_top_p', help="top_p for StreetMath generation", type = float, default = None)
parser.add_argument('--streetmath_top_k', help="top_k for StreetMath generation", type = int, default = None)
parser.add_argument('--streetmath_no_system', help="disable StreetMath system prompt", action="store_true")
parser.add_argument('--streetmath_custom_system', help="custom StreetMath system prompt text", type = str, default = None)
parser.add_argument('--streetmath_no_tools', help="disallow tool calls in StreetMath prompts", action="store_true")
parser.add_argument('--streetmath_hint', help="add StreetMath hint block", action="store_true")
parser.add_argument('--streetmath_prompt_template_file', help="path to custom StreetMath prompt template", type = str, default = None)
args = parser.parse_args()
if not args.eval_datasets and not args.train_lm_eval_task and not args.streetmath_eval:
    auto_eval = None
    if args.calibration_dataset_names:
        auto_eval = args.calibration_dataset_names[0].strip().lower()
    if auto_eval:
        args.eval_datasets = [auto_eval]
        if args.eval_dataset_subset is None or args.eval_dataset_subset > 1:
            args.eval_dataset_subset = 1
        print(f"[MathNeuro] Auto-eval enabled: eval_datasets={args.eval_datasets} eval_dataset_subset={args.eval_dataset_subset}")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import lm_eval
import json 
from pathlib import Path
import sys
import time
import math

def read_csv_safe(path, random_state=None):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if random_state is None:
        return df
    return df.sample(frac=1, random_state=random_state)


def build_task_manager(task_names):
    tasks_root = Path(lm_eval.__file__).resolve().parent / "tasks"
    task_dirs = set()
    for name in task_names:
        if not name:
            continue
        if name == "mmlu" or name.startswith("mmlu_"):
            task_dirs.add(tasks_root / "mmlu")
        elif name.startswith("gsm8k"):
            task_dirs.add(tasks_root / "gsm8k")
        else:
            candidate = tasks_root / name
            if candidate.is_dir():
                task_dirs.add(candidate)
    if task_dirs:
        return lm_eval.tasks.TaskManager(
            include_path=[str(path) for path in sorted(task_dirs)],
            include_defaults=False,
        )
    return lm_eval.tasks.TaskManager()

def resolve_mmlu_eval_tasks(task_names, total_limit, seed):
    if not task_names or set(task_names) != {"mmlu"} or total_limit is None:
        return task_names, total_limit
    if total_limit <= 0:
        return task_names, total_limit
    try:
        from lm_eval.tasks.mmlu import _generate_configs as mmlu_gen
        subjects = sorted(mmlu_gen.SUBJECTS.keys())
    except Exception:
        return task_names, total_limit
    if total_limit <= len(subjects):
        rng = np.random.default_rng(seed)
        selected = rng.choice(subjects, size=total_limit, replace=False).tolist()
        return [f"mmlu_{s}" for s in selected], 1
    per_task = max(1, int(math.ceil(total_limit / len(subjects))))
    return [f"mmlu_{s}" for s in subjects], per_task


def _streetmath_full_prompt(tokenizer, user_prompt, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return (system_prompt + "\n\n" if system_prompt else "") + user_prompt


def run_streetmath_eval(model, tokenizer, output_path, args):
    if args.streetmath_root:
        streetmath_root = Path(args.streetmath_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[1]
        streetmath_root = repo_root / "StreetMathDataset"
    if str(streetmath_root) not in sys.path:
        sys.path.insert(0, str(streetmath_root))

    from streetmath_benchmark.loader import load_streetmath
    from streetmath_benchmark.prompt import build_prompt, DEFAULT_SYSTEM_PROMPT
    from streetmath_benchmark.eval import build_result_record, summarize

    if args.streetmath_prompt_template_file:
        try:
            custom_user = Path(args.streetmath_prompt_template_file).read_text(encoding="utf-8")
        except Exception:
            custom_user = None
    else:
        custom_user = None

    if args.streetmath_no_system:
        system_prompt = None
    else:
        system_prompt = args.streetmath_custom_system or DEFAULT_SYSTEM_PROMPT

    if args.streetmath_jsonl:
        streetmath_jsonl = args.streetmath_jsonl
    else:
        streetmath_jsonl = str(streetmath_root / "data" / "street_math_test.jsonl")

    samples = load_streetmath(
        local_jsonl=streetmath_jsonl,
        split="test",
        limit=None,
        shuffle=False,
        seed=args.random_state,
    )
    eval_limit = args.streetmath_limit
    if args.streetmath_eval_size is not None:
        eval_limit = args.streetmath_eval_size if eval_limit is None else min(eval_limit, args.streetmath_eval_size)
    if eval_limit is not None:
        limit = int(eval_limit)
        if limit > 0 and len(samples) > limit:
            stride = max(1, len(samples) // limit)
            samples = samples[::stride][:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")
    results = []
    total = len(samples)
    batch_size = max(1, int(args.streetmath_batch_size))
    log_print(f"StreetMath eval: {total} samples, batch_size={batch_size}")
    for start_idx in range(0, total, batch_size):
        batch = samples[start_idx:start_idx + batch_size]
        user_prompts = []
        for sample in batch:
            user_prompts.append(build_prompt(
                sample=sample,
                custom_instructions=custom_user,
                disallow_tools=args.streetmath_no_tools,
                hint=args.streetmath_hint,
            ))
        full_prompts = [_streetmath_full_prompt(tokenizer, p, system_prompt) for p in user_prompts]

        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass

        start = time.time()
        tok = tokenizer(full_prompts, return_tensors="pt", padding=True).to(model.device)
        gen_kwargs = {
            **tok,
            "temperature": args.streetmath_temperature,
            "max_new_tokens": args.streetmath_max_tokens,
            "top_p": args.streetmath_top_p,
            "top_k": args.streetmath_top_k,
            "do_sample": args.streetmath_temperature > 0,
        }
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        elapsed = time.time() - start
        for i, sample in enumerate(batch):
            prompt_tokens = int(tok["attention_mask"][i].sum().item())
            completion_tokens = int(out[i].shape[-1] - prompt_tokens)
            response_ids = out[i][prompt_tokens:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(out[i].shape[-1]),
                "token_count_source": "tokenizer",
            }
            rec = build_result_record(
                sample=sample,
                provider_name="transformers",
                model_name=args.model,
                response_text=response_text,
                usage=usage,
                elapsed=elapsed,
                prompt_text=full_prompts[i],
                eval_config={
                    "decoding": {
                        "temperature": args.streetmath_temperature,
                        "top_p": args.streetmath_top_p,
                        "top_k": args.streetmath_top_k,
                        "max_tokens": args.streetmath_max_tokens,
                        "seed": args.random_state,
                    },
                },
            )
            results.append(rec)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_print(f"StreetMath progress: {min(start_idx + batch_size, total)}/{total}")

    summary = summarize([r for r in results if "judgement" in r])
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

if 'sgsm' in args.train_dataset:
    df = read_csv_safe(args.train_dataset) # Load SGSM dataset for few-shot prompting
    df = df[df['subset']=="sgsm_train"] # Subset SGSM to verified training subset
    df = df.sample(frac = 1, random_state = args.random_state)
    for i in range(0, len(df)):
        try:
            answer = df.iloc[i]['answer']
            answer = float(answer)
            df.iloc[i]['answer'] = answer
        except:
            df = df.drop([i])
    
    train = df.iloc[0:1500]
    
    val = df.iloc[1500:]
    val = val.sample(frac = 1, random_state = args.random_state)

if 'sgsm' not in args.train_dataset:
    train = read_csv_safe(args.train_dataset, random_state=args.random_state) # Load SGSM dataset for few-shot prompting

if args.train_dataset_size is not None:
    train = train.sample(n=min(args.train_dataset_size, len(train)), random_state=args.random_state)
    

calibration_datasets = []
for dataset in args.calibration_datasets:
    if '/' in dataset:
        dataset_name = dataset.split('/')[-1]
        dataset_name = dataset_name.split('.csv')[0]
        calibration_datasets.append(dataset_name)
    else:
        dataset_name = dataset.split('.csv')[0]
        calibration_datasets.append(dataset_name)

dataset_list = []
for dataset, dataset_name, name in zip(args.calibration_datasets, calibration_datasets, args.calibration_dataset_names):
    # Load the dataset into a DataFrame
    globals()[dataset_name] = read_csv_safe(dataset, random_state=args.random_state)  # Shuffle the DataFrame

    if args.calibration_dataset_size is not None:
        globals()[dataset_name] = globals()[dataset_name].sample(
            n=min(args.calibration_dataset_size, len(globals()[dataset_name])),
            random_state=args.random_state
        )
    
    # Assign a name attribute to the DataFrame
    globals()[dataset_name].name = name
    
    # Append the actual DataFrame object to the list
    dataset_list.append(globals()[dataset_name])

task_manager_names = []
if args.eval_datasets:
    task_manager_names.extend(args.eval_datasets)
if args.train_lm_eval_task:
    task_manager_names.append(args.train_lm_eval_task)
    
output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"
results_path =  f"{args.save_path}/eval_results/{args.model}/"
os.makedirs(os.path.dirname(results_path), exist_ok=True)
timestamp_log = os.path.join(results_path, "run_timestamps.log")

def log_ts(message):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        with open(timestamp_log, "a", encoding="utf-8") as f:
            f.write(f"{ts} {message}\n")
    except Exception:
        pass

def log_print(message):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {message}")
    log_ts(message)

log_ts("run_start")
log_print("run_start")

try:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    log_print(f"Loaded tokenizer from local cache: {args.model}")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    log_print(f"Loaded tokenizer (remote fallback): {args.model}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    log_print(f"Loaded model from local cache: {args.model}")
except Exception:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    log_print(f"Loaded model (remote fallback): {args.model}")
model.eval()
torch.set_grad_enabled(False)
if hasattr(model, "config"):
    model.config.use_cache = True
if args.pre_train_eval:
    if 'sgsm' in args.train_dataset:
        prune_solve = []
        prune_code = []
        prune_solutions = []
        for i in range(0, min(args.eval_dataset_subset, len(val))):
            # Format the prompt
            prompts = []
            questions = []
            final_question = val.iloc[i]['question']
            final_answer = val.iloc[i]['answer']
            final_prompt = f"""Instruct: {final_question} Let's write a Python program.\nOutput:"""
    
            for j in range(0, 8):
                question = train['question'].iloc[j]
                questions.append(question)
                answer = train['solution'].iloc[j]
                prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"""
                if prompt not in prompts:
                    prompts.append(prompt)
    
            prompts.append(final_prompt)
            formatted_prompt = "\n\n".join(prompts)
            #Query the model 
            inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
            model_answer = None
            with torch.no_grad():
                output = model.generate(inputs, max_new_tokens = 150)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Split the generated text by the prompt to extract the newly generated part
            generated_text_parts = generated_text.split(final_prompt)
            solution_text = generated_text_parts[-1].strip()
            prune_solutions.append(solution_text)
            if "Instruct:" in solution_text:
                solution_text = solution_text.split("Instruct:")[0] # Split up a generation that contains more than one question
            if "print" in solution_text:
                solution_text = solution_text.split("print")[0] # Split up a generation that contains a print statement
            if "Student:" in solution_text:
                solution_text = solution_text.split("Student:")[0] # Split up a generation that contains more than one question
            if "Output:" in solution_text:
                solution_text = solution_text.split("Output:")[0] # Split up a generation that contains more than one question
            if "#TODO" in solution_text:
                solution_text = solution_text.split("#TODO")[0] # Split up a generation that contains more than one question
            #solutions.append(solution_text)
            if 'return result' in solution_text:
                # Split the string on 'return result' but keep 'return result' in the result
                parts = re.split(r'(return result)', solution_text)
    
                # Rejoin the parts correctly
                solution_text = parts[0] + parts[1]
            try:
                exec(solution_text)
                model_answer = solution()
                prune_code.append(1)
                model_answer = float(model_answer)
                if model_answer != final_answer:
                    prune_solve.append(0)
    
                if model_answer == final_answer:
                    prune_solve.append(1)
    
            except:
                prune_code.append(0)
                prune_solve.append(0)
    
        with open(output_file, "a") as f:  # Open the file in append mode ("a")
                f.write(f"Average eval accuracy on {min(args.eval_dataset_subset, len(val))} questions before training with greedy decoding (few-shot): {np.mean(prune_solve)}\n") 
        task_manager = build_task_manager(task_manager_names)
        #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
        eval_tasks, eval_limit = resolve_mmlu_eval_tasks(args.eval_datasets, args.eval_dataset_subset, args.random_state)
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=eval_tasks,
            task_manager=task_manager,
            log_samples = False, 
            batch_size = args.eval_batch_size,
            limit = eval_limit,
            random_seed = args.random_state
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
        if args.streetmath_eval:
            streetmath_out = f"{args.save_path}/eval_results/{args.model}/STREET_MATH_pre.jsonl"
            run_streetmath_eval(model, tokenizer, streetmath_out, args)
    
    if args.train_lm_eval_task is not None:
        task_manager = build_task_manager(task_manager_names)
        #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=[args.train_lm_eval_task],
            task_manager=task_manager,
            log_samples = False, 
            batch_size = args.eval_batch_size,
            limit = args.eval_dataset_subset, 
            random_seed = args.random_state
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results_train_task.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
        
        eval_tasks, eval_limit = resolve_mmlu_eval_tasks(args.eval_datasets, args.eval_dataset_subset, args.random_state)
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=eval_tasks,
            task_manager=task_manager,
            log_samples = False, 
            batch_size = args.eval_batch_size,
            limit = eval_limit
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
        if args.streetmath_eval:
            streetmath_out = f"{args.save_path}/eval_results/{args.model}/STREET_MATH_pre.jsonl"
            run_streetmath_eval(model, tokenizer, streetmath_out, args)
            
magnitude = {}
def getActivation(name):
    # The hook function
    def hook(module, input, output):
        activations = input[0]  # Get the input activations
        weights = module.weight.data  # Get the weights
        if activations.dim() == 3:
            activations_norm = activations.norm(p=2, dim=1).mean(dim=0).to(torch.bfloat16)
        elif activations.dim() == 2:
            activations_norm = activations.norm(p=2, dim=0).to(torch.bfloat16)
        else:
            activations_norm = activations.to(torch.bfloat16)
        weight_magnitudes = torch.abs(weights).mean(dim=0).to(torch.bfloat16)
        modified_output = activations_norm * weight_magnitudes
        magnitude[name] = modified_output.detach().cpu()  # Store on CPU to reduce VRAM
    # Return the hook function
    return hook

for name, module in model.named_modules():
    if (isinstance(module, (nn.Linear))):
        hook_fn = getActivation(name)  # Get the hook function
        module.register_forward_hook(hook_fn)  # Register the hook function

if 'bad_gens_full.csv' in args.calibration_datasets:
    def find_params(model, gens, keep_ratio, prune = True, largest = True, num_samples = len(bad_gens_full)):
        global chosen_params
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

        param_dict = {}
        for name, param in model.named_parameters():
            param_dict[name] = torch.zeros_like(param, device="cpu")
        seen_layers = set()
        
        limit = min(num_samples, len(gens))
        batch_size = max(1, int(args.eval_batch_size))
        for start_idx in range(0, limit, batch_size):
            batch = gens.iloc[start_idx:start_idx + batch_size]
            prompts = batch['0'].tolist()
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(model.device)
            with torch.no_grad():
                _ = model(**inputs, use_cache=False)
            for key, tensor in magnitude.items():
                try:
                    param_dict[f"{key}.weight"] += tensor.detach().cpu()
                except:
                    pass
                seen_layers.add(key)
            magnitude.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if seen_layers:
            keys_to_remove = [key for key in param_dict if key.split('.weight')[0] not in seen_layers]
        else:
            keys_to_remove = list(param_dict.keys())
    
        for key in keys_to_remove:
            del param_dict[key]
        
        mask_dict = {}
    
    
        for k, v in param_dict.items():
            if "embed" in k:
                if prune == False:
                    mask_dict[k] = torch.zeros_like(v)
                else:
                    mask_dict[k] = torch.ones_like(v)
    
            else:
                if prune == False:
                    sizes = v.shape
                    num_params = v.numel()
                    keep_num = int(num_params * keep_ratio)
                    tensor = v.view(-1)
                    top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]
                    mask_dict[k] = torch.zeros_like(tensor)
                    mask_dict[k][top_pos] = 1
                    mask_dict[k] = mask_dict[k].reshape(v.shape)
                else:
                    sizes = v.shape
                    num_params = v.numel()
                    keep_num = int(num_params * keep_ratio)
                    tensor = v.view(-1)
                    top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]
                    mask_dict[k] = torch.ones_like(tensor)
                    mask_dict[k][top_pos] = 0
                    mask_dict[k] = mask_dict[k].reshape(v.shape)
    
        return mask_dict
        
    
def find_good_params(model, train, keep_ratio, prune = True, largest = True, num_samples = len(train)):
    global chosen_params
    import random

    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = torch.zeros_like(param, device="cpu")
    seen_layers = set()
            
    limit = min(num_samples, len(train))
    batch_size = max(1, int(args.eval_batch_size))
    for start_idx in range(0, limit, batch_size):
        batch = train.iloc[start_idx:start_idx + batch_size]
        prompts = []
        if 'qa' in train.columns.to_list():
            prompts = batch['qa'].tolist()
        else:
            for _, row in batch.iterrows():
                question = row['question']
                answer = row['solution']
                prompts.append(f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}""")
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
        for key, tensor in magnitude.items():
            try:
                param_dict[f"{key}.weight"] += tensor.detach().cpu()
            except:
                print(f'passed at {key}')
            seen_layers.add(key)
        magnitude.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if seen_layers:
        keys_to_remove = [key for key in param_dict if key.split('.weight')[0] not in seen_layers]
    else:
        keys_to_remove = list(param_dict.keys())

    for key in keys_to_remove:
        del param_dict[key]

    # create dictionary to store mask 
    mask_dict = {}


    for k, v in param_dict.items():
        # don't count classifier layer
        if "embed" in k:
            if prune == False:
                mask_dict[k] = torch.zeros_like(v)
            else:
                mask_dict[k] = torch.ones_like(v)

        else:
            if prune == False:
                sizes = v.shape
                num_params = v.numel()
                keep_num = int(num_params * keep_ratio)
                tensor = v.view(-1)
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]
                mask_dict[k] = torch.zeros_like(tensor)
                mask_dict[k][top_pos] = 1
                mask_dict[k] = mask_dict[k].reshape(v.shape)
            else:
                sizes = v.shape
                num_params = v.numel()
                keep_num = int(num_params * keep_ratio)
                tensor = v.view(-1)
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]
                mask_dict[k] = torch.ones_like(tensor)
                mask_dict[k][top_pos] = 0
                mask_dict[k] = mask_dict[k].reshape(v.shape)

    return mask_dict
    
def prune(bad_params, good_params, factor, return_good = False):
    prune_params = {}
    if return_good ==False:
        for k, v in bad_params.items():
            prune_params[k] = bad_params[k] - good_params[k]
            indices = prune_params[k]!=-1
            bad_indices = prune_params[k]==-1
            prune_params[k] = indices + (bad_indices*factor)

    else:
        for k, v in bad_params.items():
            prune_params[k] = good_params[k] - bad_params[k]
            indices = prune_params[k]!=-1
            good_indices = prune_params[k]==-1
            prune_params[k] = indices + (good_indices*factor)
    return prune_params

def scale(good_params, factor):
    prune_params = {}
    for k, v in good_params.items():
        good_indices = good_params[k]!=1
        keep_indices = good_params[k]==1
        prune_params[k] = keep_indices + (good_indices*factor)
    return prune_params

num_samples = args.num_samples
num_repeats = args.num_repeats
if args.proportion is None:
    #good_percents = [.0001, .001, .005, .01, .025, .05, .1, .15]
    good_percents = [0.0001, .001, .01, 0.1]
if args.proportion is not None:
    good_percents = [args.proportion]
scalar = args.scalar
log_print(f"Reusing loaded model for all datasets: {args.model}")
base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
for dataset in dataset_list:
    for repeat in range(0, num_repeats):
        log_print(f"Starting repeat {repeat + 1}/{num_repeats} for dataset={dataset.name}")
        run_seed = args.random_state + repeat
        sampled_train = train.sample(n = num_samples, replace = True, random_state = run_seed)
        sampled_comparison = dataset.sample(n = num_samples, replace = True, random_state = run_seed)
        for good_percent in good_percents:
            log_print(f"Using proportion={good_percent} for dataset={dataset.name} repeat={repeat}")
            log_ts(f"prune_start proportion={good_percent} repeat={repeat}")
            model.load_state_dict(base_state, strict=True)
            torch.cuda.empty_cache()
            magnitude = {}
            def getActivation(name):
                # The hook function
                def hook(module, input, output):
                    activations = input[0]  # Get the input activations
                    weights = module.weight.data  # Get the weights
                    if activations.dim() == 3:
                        activations_norm = activations.norm(p=2, dim=1).mean(dim=0).to(torch.bfloat16)
                    elif activations.dim() == 2:
                        activations_norm = activations.norm(p=2, dim=0).to(torch.bfloat16)
                    else:
                        activations_norm = activations.to(torch.bfloat16)
                    weight_magnitudes = torch.abs(weights).mean(dim=0).to(torch.bfloat16)
                    modified_output = activations_norm * weight_magnitudes
                    magnitude[name] = modified_output.detach().cpu()  # Store on CPU to reduce VRAM
                # Return the hook function
                return hook
            
            for name, module in model.named_modules():
                if (isinstance(module, (nn.Linear))):
                    hook_fn = getActivation(name)  # Get the hook function
                    module.register_forward_hook(hook_fn)  # Register the hook function
            good_params = find_good_params(model, sampled_train, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)
            torch.cuda.empty_cache()
            if 'Bad' in dataset.name:    
                comparison_params = find_params(model, sampled_comparison, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)
            else:
                comparison_params = find_good_params(model, sampled_comparison, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)

            prune_params = prune(comparison_params, good_params, scalar, return_good = True)
            del good_params
            del comparison_params
            pruned_state = {}
            masked_count = 0
            for key, base_tensor in base_state.items():
                mask = prune_params.get(key)
                if mask is None:
                    pruned_state[key] = base_tensor
                else:
                    pruned_state[key] = base_tensor * mask
                    masked_count += 1
            log_print(f"Applied pruning masks to {masked_count} tensors")
            model.load_state_dict(pruned_state, strict=True)
            del pruned_state
            del prune_params
            torch.cuda.empty_cache()
            def remove_hooks(model):
                # Function to remove all hooks
                for name, module in model.named_modules():
                    # Check if the module has any forward hooks
                    if hasattr(module, "_forward_hooks") and len(module._forward_hooks) > 0:
                        # Remove all forward hooks
                        module._forward_hooks.clear()
        
            remove_hooks(model)
            if args.streetmath_eval and not args.eval_datasets and args.train_lm_eval_task is None:
                streetmath_out = f"{args.save_path}/eval_results/{args.model}/STREET_MATH_calculate{good_percent}_run{repeat}.jsonl"
                os.makedirs(os.path.dirname(streetmath_out), exist_ok=True)
                run_streetmath_eval(model, tokenizer, streetmath_out, args)
            if 'sgsm' in args.train_dataset:
                prune_solve = []
                prune_code = []
                prune_solutions = []
                for i in range(0, min(args.eval_dataset_subset, len(val))):
                    # Format the prompt
                    prompts = []
                    questions = []
                    final_question = val.iloc[i]['question']
                    final_answer = val.iloc[i]['answer']
                    final_prompt = f"""Instruct: {final_question} Let's write a Python program.\nOutput:"""
    
                    for j in range(0, 8):
                        question = train['question'].iloc[j]
                        questions.append(question)
                        answer = train['solution'].iloc[j]
                        prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"""
                        if prompt not in prompts:
                            prompts.append(prompt)
    
                    prompts.append(final_prompt)
                    formatted_prompt = "\n\n".join(prompts)
                    #Query the model 
                    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
                    model_answer = None
                    #output = model.generate(inputs, max_new_tokens = 150, temperature = .7, do_sample = True)
                    with torch.no_grad():
                        output = model.generate(inputs, max_new_tokens = 150)
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    # Split the generated text by the prompt to extract the newly generated part
                    generated_text_parts = generated_text.split(final_prompt)
                    solution_text = generated_text_parts[-1].strip()
                    prune_solutions.append(solution_text)
                    if "Instruct:" in solution_text:
                        solution_text = solution_text.split("Instruct:")[0] # Split up a generation that contains more than one question
                    if "print" in solution_text:
                        solution_text = solution_text.split("print")[0] # Split up a generation that contains a print statement
                    if "Student:" in solution_text:
                        solution_text = solution_text.split("Student:")[0] # Split up a generation that contains more than one question
                    if "Output:" in solution_text:
                        solution_text = solution_text.split("Output:")[0] # Split up a generation that contains more than one question
                    if "#TODO" in solution_text:
                        solution_text = solution_text.split("#TODO")[0] # Split up a generation that contains more than one question
                    #solutions.append(solution_text)
                    if 'return result' in solution_text:
                        # Split the string on 'return result' but keep 'return result' in the result
                        parts = re.split(r'(return result)', solution_text)
    
                        # Rejoin the parts correctly
                        solution_text = parts[0] + parts[1]
                    try:
                        exec(solution_text)
                        model_answer = solution()
                        prune_code.append(1)
                        model_answer = float(model_answer)
                        if model_answer != final_answer:
                            prune_solve.append(0)
    
                        if model_answer == final_answer:
                            prune_solve.append(1)
    
                    except:
                        prune_code.append(0)
                        prune_solve.append(0)
    
                with open(output_file, "a") as f:  # Open the file in append mode ("a")
                        f.write(f"Average eval accuracy on {min(args.eval_dataset_subset, len(val))} questions for pruning top {good_percent}% good parameters based on not being activated by {dataset.name} based on {num_samples} training samples and greedy decoding (few-shot): {np.mean(prune_solve)}\n")  
                torch.cuda.empty_cache()
            if args.eval_datasets:
                task_manager = build_task_manager(task_manager_names)
                #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
                # Setting `task_manager` to the one above is optional and should generally be done
                # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
                # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
                eval_tasks, eval_limit = resolve_mmlu_eval_tasks(args.eval_datasets, args.eval_dataset_subset, run_seed)
                results = lm_eval.simple_evaluate( # call simple_evaluate
                    model = 'hf',
                    model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=eval_tasks,
                    task_manager=task_manager,
                    log_samples = False, 
                    batch_size = args.eval_batch_size,
                    limit = eval_limit,
                    random_seed = run_seed
                )
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_run{repeat}.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)
                if args.streetmath_eval:
                    streetmath_out = f"{args.save_path}/eval_results/{args.model}/STREET_MATH_calculate{good_percent}_run{repeat}.jsonl"
                    run_streetmath_eval(model, tokenizer, streetmath_out, args)
            if args.train_lm_eval_task is not None and not args.train_task_only_pre:
                task_manager = build_task_manager(task_manager_names)
                #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
                # Setting `task_manager` to the one above is optional and should generally be done
                # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
                # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
                results = lm_eval.simple_evaluate( # call simple_evaluate
                    model = 'hf',
                    model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=[args.train_lm_eval_task],
                    task_manager=task_manager,
                    log_samples = False, 
                    batch_size = args.eval_batch_size,
                    limit = args.eval_dataset_subset, 
                    random_seed = run_seed
                )
                results_path = f"{args.save_path}/eval_results/{args.model}/{args.train_lm_eval_task}_calculate{good_percent}_run{repeat}_train_task.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)
                    
                    eval_tasks, eval_limit = resolve_mmlu_eval_tasks(args.eval_datasets, args.eval_dataset_subset, run_seed)
                    results = lm_eval.simple_evaluate( # call simple_evaluate
                        model = 'hf',
                        model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                        tasks=eval_tasks,
                        task_manager=task_manager,
                        log_samples = False, 
                        batch_size = args.eval_batch_size,
                        limit = eval_limit,
                        random_seed = run_seed
                    )
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_run{repeat}.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)
                if args.streetmath_eval:
                    streetmath_out = f"{args.save_path}/eval_results/{args.model}/STREET_MATH_calculate{good_percent}_run{repeat}.jsonl"
                    run_streetmath_eval(model, tokenizer, streetmath_out, args)
                        
del base_state
del model
log_ts("run_end")
log_print("run_end")
