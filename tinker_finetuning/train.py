import argparse
import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import dotenv
import tinker
from tinker import types

dotenv.load_dotenv()

import wandb 


@dataclass
class Config:
    model_name: str
    training_prompts_path: str
    validation_prompts_path: str
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 5e-4
    max_sequence_length: int = 32500
    lora_rank: int = 32
    checkpoint_dir: str = "./tinker_checkpoints"
    run_name: Optional[str] = None
    # Weights & Biases (optional)
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None  # "online" | "offline" | "disabled"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[str] = None  # comma-separated


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train LoRA model with Tinker on prompt/output pairs.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Base model name for LoRA training")
    parser.add_argument("--training_prompts_path", type=str, default="../data/arc-agi-2025/prompts/arc-agi_training_prompts.json", help="Path to training prompts JSON")
    parser.add_argument("--validation_prompts_path", type=str, default="../data/arc-agi-2025/prompts/arc-agi_validation_prompts.json", help="Path to validation prompts JSON")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_seq_length", type=int, default=32500)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="./tinker_checkpoints")
    parser.add_argument("--run_name", type=str, default=None, help="Optional name prefix for checkpoints, e.g., run42")
    # WandB options
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (defaults to ARC-AGI-2)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/org (optional)")
    parser.add_argument("--wandb_mode", type=str, default=None, help="W&B mode: online|offline|disabled")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (defaults to --run_name if provided)")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags")

    args = parser.parse_args()
    return Config(
        model_name=args.model_name,
        training_prompts_path=args.training_prompts_path,
        validation_prompts_path=args.validation_prompts_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_sequence_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )


def build_clients_and_tokenizer(config: Config) -> tuple[tinker.TrainingClient, any]:
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is not set in environment")
    service_client = tinker.ServiceClient(api_key=api_key)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )
    tokenizer = training_client.get_tokenizer()
    return training_client, tokenizer


def load_prompts(training_path: str, validation_path: str) -> tuple[list[dict], list[dict]]:
    with open(training_path, "r") as f:
        train = json.load(f)
    with open(validation_path, "r") as f:
        valid = json.load(f)
    return train, valid


def filter_prompts_by_length(prompts: list[dict], tokenizer, max_len: int) -> list[dict]:
    out: list[dict] = []
    for p in prompts:
        input_ids = tokenizer.encode(p["prompt"] + p["output"])
        if len(input_ids) <= max_len:
            out.append(p)
    return out

def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # (see later docs) but here, we'll keep it simple.
    prompt = example['prompt']
    
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(example['output'], add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)
 
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
 
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]
 
    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )


def batch_generator(examples: List[types.Datum], batch_size: int, shuffle: bool = True) -> Iterator[List[types.Datum]]:
    idxs = list(range(len(examples)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield [examples[j] for j in idxs[i:i+batch_size]]

def collate_batch(batch: List[types.Datum]) -> List[types.Datum]:
    # tinker typically accepts a list of Datum; pad here if your client requires tensors
    return batch

def compute_mean_nll(logprobs_list, weights_list):
    total = 0.0
    total_w = 0.0
    for lp_seq, w_seq in zip(logprobs_list, weights_list):
        # Handle TensorData or similar wrappers
        if hasattr(lp_seq, 'tolist'):
            lp_seq = lp_seq.tolist()
        elif hasattr(lp_seq, 'to_numpy'):
            lp_seq = lp_seq.to_numpy()

        if hasattr(w_seq, 'tolist'):
            w_seq = w_seq.tolist()
        elif hasattr(w_seq, 'to_numpy'):
            w_seq = w_seq.to_numpy()

        lp = np.array(lp_seq, dtype=float)
        w = np.array(w_seq, dtype=float)

        # ensure equal length
        if lp.shape != w.shape:
            min_len = min(lp.shape[0], w.shape[0])
            print(f'[{now()}] Warning: logprobs shape {lp.shape} != weights shape {w.shape}, truncating to {min_len}')
            lp = lp[:min_len]
            w = w[:min_len]

        total += (-lp * w).sum()
        total_w += w.sum()

    return float(total / total_w) if total_w > 0 else None


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def save_checkpoint(training_client: tinker.TrainingClient, config: Config, step_num: int, metadata: dict | None = None):
    train_remote_path = None
    sampler_remote_path = None
    try:
        # Prefer official API per docs: save_state(name=...) returning future with .result().path
        remote_name = f"{config.run_name}-{step_num:06d}" if config.run_name else f"{step_num:06d}"
        if hasattr(training_client, 'save_state'):
            fut = training_client.save_state(name=remote_name)
            # Handle future-like result objects
            result_obj = fut.result() if hasattr(fut, 'result') else fut
            train_remote_path = getattr(result_obj, 'path', None)
            if train_remote_path:
                print(f'[{now()}] Saved remote training state -> {train_remote_path}')

        if hasattr(training_client, 'save_weights_for_sampler'):
            # As an alternative, at least save weights for sampling
            fut = training_client.save_weights_for_sampler(name=remote_name)
            result_obj = fut.result() if hasattr(fut, 'result') else fut
            sampler_remote_path = getattr(result_obj, 'path', None)
            if sampler_remote_path:
                print(f'[{now()}] Saved remote weights for sampler -> {sampler_remote_path}')
    except Exception as e:
        print(f'[{now()}] Warning: remote checkpoint save failed: {e}')
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_meta = {'step': step_num, 'timestamp': now(), 'train_remote_path': train_remote_path, 'sampler_remote_path': sampler_remote_path}
    if metadata:
        checkpoint_meta.update(metadata)
    local_name = f'checkpoint_step_{step_num}.json' if not config.run_name else f'{config.run_name}_step_{step_num}.json'
    path = os.path.join(config.checkpoint_dir, local_name)
    with open(path, 'w') as f:
        json.dump(checkpoint_meta, f)
    print(f'[{now()}] Saved local checkpoint metadata -> {path}')
    return path, train_remote_path, sampler_remote_path

# Async wrappers that prefer the async client API and fall back to sync calls in executor
async def call_forward_backward(training_client: tinker.TrainingClient, batch):
    if hasattr(training_client, 'forward_backward_async'):
        maybe_coro = training_client.forward_backward_async(batch, loss_fn='cross_entropy')
        res = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
        # handle future-like result objects returned by Tinker async APIs
        if hasattr(res, 'result_async'):
            return await res.result_async()
        if hasattr(res, 'result'):
            return res.result()
        return res
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, training_client.forward_backward, batch)

async def call_optim_step(training_client: tinker.TrainingClient, adam_params):
    if hasattr(training_client, 'optim_step_async'):
        maybe_coro = training_client.optim_step_async(adam_params)
        res = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
        if hasattr(res, 'result_async'):
            return await res.result_async()
        if hasattr(res, 'result'):
            return res.result()
        return res
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, training_client.optim_step, adam_params)

async def call_forward(training_client: tinker.TrainingClient, batch):
    """Forward pass without gradient accumulation (for evaluation)."""
    if hasattr(training_client, 'forward_async'):
        maybe_coro = training_client.forward_async(batch, loss_fn='cross_entropy')
        res = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
        if hasattr(res, 'result_async'):
            return await res.result_async()
        if hasattr(res, 'result'):
            return res.result()
        return res
    elif hasattr(training_client, 'forward'):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, training_client.forward, batch, 'cross_entropy')
    else:
        raise RuntimeError('TrainingClient has neither forward_async nor forward; cannot run validation forward pass.')

def _accumulate_neglogprob_totals(logprobs_list, weights_list):
    """Utility for validation: returns (total_neglogprob, total_weight)."""
    total = 0.0
    total_w = 0.0
    for lp_seq, w_seq in zip(logprobs_list, weights_list):
        if hasattr(lp_seq, 'tolist'):
            lp_seq = lp_seq.tolist()
        elif hasattr(lp_seq, 'to_numpy'):
            lp_seq = lp_seq.to_numpy()

        if hasattr(w_seq, 'tolist'):
            w_seq = w_seq.tolist()
        elif hasattr(w_seq, 'to_numpy'):
            w_seq = w_seq.to_numpy()

        lp = np.array(lp_seq, dtype=float)
        w = np.array(w_seq, dtype=float)

        if lp.shape != w.shape:
            min_len = min(lp.shape[0], w.shape[0])
            print(f'[{now()}] Warning: validation logprobs shape {lp.shape} != weights shape {w.shape}, truncating to {min_len}')
            lp = lp[:min_len]
            w = w[:min_len]

        total += (-lp * w).sum()
        total_w += w.sum()
    return total, total_w

async def evaluate_nll(
    config: Config,
    training_client: tinker.TrainingClient,
    processed_validation_examples: list[types.Datum],
) -> Optional[float]:
    if not processed_validation_examples:
        return None
    total_neglogprob = 0.0
    total_weight = 0.0
    for batch in batch_generator(processed_validation_examples, config.batch_size, shuffle=False):
        batch = collate_batch(batch)
        try:
            fwd_res = await call_forward(training_client, batch)
        except Exception as e:
            print(f'[{now()}] Validation forward failed for a batch: {e}. Skipping batch.')
            continue

        val_logprobs = []
        try:
            lf_outputs = getattr(fwd_res, 'loss_fn_outputs', None)
            if isinstance(lf_outputs, list):
                for entry in lf_outputs:
                    if isinstance(entry, dict) and 'logprobs' in entry:
                        val_logprobs.append(entry['logprobs'])
            elif isinstance(lf_outputs, dict) and 'logprobs' in lf_outputs:
                val_logprobs = [lf_outputs['logprobs']]
        except Exception as e:
            print(f'[{now()}] Could not extract validation loss_fn_outputs: {e}')
            continue

        val_weights = [d.loss_fn_inputs['weights'] for d in batch]
        batch_neglogprob, batch_weight = _accumulate_neglogprob_totals(val_logprobs, val_weights)
        total_neglogprob += batch_neglogprob
        total_weight += batch_weight

    if total_weight > 0:
        return float(total_neglogprob / total_weight)
    return None

async def train_async(
    config: Config,
    training_client: tinker.TrainingClient,
    processed_training_examples: list[types.Datum],
    processed_validation_examples: list[types.Datum],
):
    global_step = 0
    num_examples = len(processed_training_examples)
    steps_per_epoch = max(1, math.ceil(num_examples / config.batch_size))
    print(f'[{now()}] Starting async training: epochs={config.epochs}, batch_size={config.batch_size}, base_lr={config.learning_rate}')

    # Initialize Weights & Biases if enabled
    wb_run = None
    if config.wandb_enabled:
        if wandb is None:
            print(f'[{now()}] Warning: wandb is not installed. Disable with --wandb_mode=disabled or install wandb to enable logging.')
        else:
            try:
                run_name = config.wandb_run_name or config.run_name
                wb_kwargs = {
                    'project': config.wandb_project or 'ARC-AGI-2',
                    'name': run_name,
                    'reinit': True,
                }
                if config.wandb_entity:
                    wb_kwargs['entity'] = config.wandb_entity
                if config.wandb_mode:
                    os.environ['WANDB_MODE'] = config.wandb_mode
                tags = None
                if config.wandb_tags:
                    tags = [t.strip() for t in config.wandb_tags.split(',') if t.strip()]
                    if tags:
                        wb_kwargs['tags'] = tags
                wb_run = wandb.init(**wb_kwargs)
                # Log static config
                wandb.config.update({
                    'model_name': config.model_name,
                    'lora_rank': config.lora_rank,
                    'epochs': config.epochs,
                    'batch_size': config.batch_size,
                    'base_learning_rate': config.learning_rate,
                    'max_sequence_length': config.max_sequence_length,
                    'checkpoint_dir': config.checkpoint_dir,
                    'run_name': config.run_name,
                }, allow_val_change=True)
            except Exception as e:
                print(f'[{now()}] Warning: Failed to initialize wandb: {e}')

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        epoch_loss_accum = 0.0
        epoch_items = 0
        for batch in batch_generator(processed_training_examples, config.batch_size, shuffle=True):
            global_step += 1
            batch = collate_batch(batch)

            # linear LR schedule
            step = global_step - 1
            lr_mult = max(0.0, 1.0 - step / (steps_per_epoch * config.epochs))
            current_lr = config.learning_rate * lr_mult
            adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

            # forward/backward (async-aware)
            try:
                fwd_res = await call_forward_backward(training_client, batch)
            except Exception as e:
                print(f'[{now()}] forward_backward failed at step {global_step}: {e}. Retrying once...')
                try:
                    await asyncio.sleep(1.0)
                    fwd_res = await call_forward_backward(training_client, batch)
                except Exception as e2:
                    print(f'[{now()}] Retry failed: {e2}. Skipping this batch.')
                    continue

            # optimizer step
            try:
                _ = await call_optim_step(training_client, adam_params)
            except Exception as e:
                print(f'[{now()}] optim_step failed at step {global_step}: {e}')

            # extract per-sequence logprobs following cookbook pattern
            train_logprobs = []
            try:
                lf_outputs = getattr(fwd_res, 'loss_fn_outputs', None)
                if isinstance(lf_outputs, list):
                    for entry in lf_outputs:
                        if isinstance(entry, dict) and 'logprobs' in entry:
                            train_logprobs.append(entry['logprobs'])
                elif isinstance(lf_outputs, dict) and 'logprobs' in lf_outputs:
                    train_logprobs = [lf_outputs['logprobs']]
                else:
                    # debug print once if unexpected structure
                    print(f'[{now()}] Unexpected loss_fn_outputs structure (truncated): {str(lf_outputs)[:400]}')
            except Exception as e:
                print(f'[{now()}] Could not extract loss_fn_outputs: {e}')

            train_weights = [d.loss_fn_inputs['weights'] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights) if train_logprobs else None

            
            if train_nll is not None:
                # Compute total number of tokens (sum of weights) for correct averaging
                batch_total_tokens = 0.0
                for w in train_weights:
                    if hasattr(w, 'tolist'):
                        batch_total_tokens += np.array(w.tolist()).sum()
                    elif hasattr(w, 'to_numpy'):
                        batch_total_tokens += w.to_numpy().sum()
                    else:
                        batch_total_tokens += np.array(w).sum()
                
                epoch_loss_accum += train_nll * batch_total_tokens
                epoch_items += batch_total_tokens

            # W&B logging for this step
            if config.wandb_enabled and wandb is not None and train_nll is not None:
                try:
                    wandb.log({
                        'train/nll': train_nll,
                        'train/lr': current_lr,
                        'train/epoch': epoch,
                        'train/step': global_step,
                        'train/batch_size': len(batch),
                    }, step=global_step)
                except Exception as e:
                    print(f'[{now()}] Warning: wandb.log failed at step {global_step}: {e}')

            if global_step % 10 == 0:
                avg_loss_so_far = (epoch_loss_accum / epoch_items) if epoch_items > 0 else None
                print(f'[{now()}] Epoch {epoch} step {global_step}  avg_loss_so_far={avg_loss_so_far} lr={current_lr}')

            if global_step % 200 == 0:
                local_path, train_remote_path, sampler_remote_path = save_checkpoint(
                    training_client,
                    config,
                    global_step,
                    metadata={'epoch': epoch, 'avg_loss_so_far': (epoch_loss_accum / epoch_items) if epoch_items else None}
                )
                if config.wandb_enabled and wandb is not None:
                    try:
                        wandb.log({
                            'checkpoint/step': global_step,
                            'checkpoint/local_path': local_path,
                            'checkpoint/train_remote_path': train_remote_path,
                            'checkpoint/sampler_remote_path': sampler_remote_path,
                        }, step=global_step)
                    except Exception as e:
                        print(f'[{now()}] Warning: wandb.log (checkpoint) failed: {e}')
            # evaluat the validation dataset after every 50 steps
            if global_step % 50 == 0:
                val_start = time.time()
                val_nll = await evaluate_nll(config, training_client, processed_validation_examples)
                val_time = time.time() - val_start
                print(f'[{now()}] Validation at step {global_step}: val_nll={val_nll} (val_time={val_time:.1f}s)')
                if config.wandb_enabled and wandb is not None:
                    try:
                        wandb.log({
                            'val/nll': val_nll,
                            'val/time_sec': val_time,
                        }, step=global_step)
                    except Exception as e:
                        print(f'[{now()}] Warning: wandb.log (validation) failed: {e}')
        # end epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = (epoch_loss_accum / epoch_items) if epoch_items > 0 else None
        # Evaluate on validation set
        val_start = time.time()
        val_nll = await evaluate_nll(config, training_client, processed_validation_examples)
        val_time = time.time() - val_start
        print(f'[{now()}] Finished epoch {epoch}/{config.epochs} time={epoch_time:.1f}s avg_loss={epoch_avg_loss}  val_nll={val_nll} (val_time={val_time:.1f}s)')
        if config.wandb_enabled and wandb is not None:
            try:
                wandb.log({
                    'epoch/avg_loss': epoch_avg_loss,
                    'epoch/time_sec': epoch_time,
                    'epoch/index': epoch,
                    'val/nll': val_nll,
                    'val/time_sec': val_time,
                }, step=global_step)
            except Exception as e:
                print(f'[{now()}] Warning: wandb.log (epoch) failed: {e}')
    local_path, train_remote_path, sampler_remote_path = save_checkpoint(training_client, config, global_step, metadata={'epoch': epoch})
    if config.wandb_enabled and wandb is not None:
        try:
            wandb.log({
                'checkpoint/final_local_path': local_path,
                'checkpoint/final_train_remote_path': train_remote_path,
                'checkpoint/final_sampler_remote_path': sampler_remote_path,
            }, step=global_step)
        except Exception as e:
            print(f'[{now()}] Warning: wandb.log (final checkpoint) failed: {e}')

    print(f'[{now()}] Async training finished. Total steps: {global_step}')
    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass


def main():
    config = parse_args()

    # Build clients and tokenizer
    training_client, tokenizer = build_clients_and_tokenizer(config)

    # Load datasets
    training_prompts, validation_prompts = load_prompts(
        config.training_prompts_path, config.validation_prompts_path
    )
    print(f'Loaded {len(training_prompts)} training prompts and {len(validation_prompts)} validation prompts.')
    # Filter by length
    new_training_prompts = filter_prompts_by_length(training_prompts, tokenizer, config.max_sequence_length)
    new_validation_prompts = filter_prompts_by_length(validation_prompts, tokenizer, config.max_sequence_length)

    print(f'After length filtering: {len(new_training_prompts)} training prompts and {len(new_validation_prompts)} validation prompts.')
    # Convert to Datum
    processed_training_examples = [process_example(ex, tokenizer) for ex in new_training_prompts]
    processed_validation_examples = [process_example(ex, tokenizer) for ex in new_validation_prompts]
    print(f'Processed {len(processed_training_examples)} training examples and {len(processed_validation_examples)} validation examples.')
    # Train
    asyncio.run(train_async(config, training_client, processed_training_examples, processed_validation_examples),debug=True)


if __name__ == "__main__":
    main()
