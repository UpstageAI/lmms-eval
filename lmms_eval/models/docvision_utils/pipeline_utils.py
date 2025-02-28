import datetime
import glob
import json
import os
from contextlib import contextmanager

import torch
import torch.distributed
from deepspeed.accelerator.real_accelerator import get_accelerator
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from torch import nn

import lmms_eval.models.docvision_utils.parallel_state as mpu


def _goes_first(is_main: bool):
    if not is_main:
        torch.distributed.barrier()

    yield

    if is_main:
        torch.distributed.barrier()


def is_main_process():
    return torch.distributed.get_rank() == 0


def is_local_main_process():
    return int(os.environ.get("LOCAL_RANK", -1)) == 0


@contextmanager
def main_process_first():
    yield from _goes_first(is_main_process())


@contextmanager
def local_main_process_first():
    yield from _goes_first(is_local_main_process())


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def gather_results_for_logging(input_ids, output_ids, task_ids, pad_token_id):
    if mpu.get_data_parallel_world_size() > 0:
        gathered_input_ids_length = torch.tensor(input_ids.size(1), dtype=torch.int64, device=input_ids.device)
        gathered_output_ids_length = torch.tensor(output_ids.size(1), dtype=torch.int64, device=output_ids.device)

        torch.distributed.all_reduce(gathered_input_ids_length, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(gathered_output_ids_length, op=torch.distributed.ReduceOp.MAX)

        cur_input_ids_length = input_ids.size(1)
        if cur_input_ids_length < gathered_input_ids_length:
            pad_tensor = torch.full(
                (input_ids.size(0), gathered_input_ids_length - cur_input_ids_length),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad_tensor], dim=1)

        cur_output_ids_length = output_ids.size(1)
        if cur_output_ids_length < gathered_output_ids_length:
            pad_tensor = torch.full(
                (output_ids.size(0), gathered_output_ids_length - cur_output_ids_length),
                pad_token_id,
                dtype=output_ids.dtype,
                device=output_ids.device,
            )
            output_ids = torch.cat([output_ids, pad_tensor], dim=1)

        task_ids = torch.tensor(task_ids, dtype=torch.int64, device=input_ids.device)

        gathered_input_ids = torch.empty(
            input_ids.size(0) * mpu.get_data_parallel_world_size(),
            gathered_input_ids_length,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        gathered_output_ids = torch.empty(
            output_ids.size(0) * mpu.get_data_parallel_world_size(),
            gathered_output_ids_length,
            dtype=output_ids.dtype,
            device=output_ids.device,
        )
        gathered_task_ids = torch.empty(
            task_ids.size(0) * mpu.get_data_parallel_world_size(),
            dtype=task_ids.dtype,
            device=task_ids.device,
        )

        torch.distributed.all_gather_into_tensor(gathered_input_ids, input_ids, group=mpu.get_data_parallel_group())
        torch.distributed.all_gather_into_tensor(gathered_output_ids, output_ids, group=mpu.get_data_parallel_group())
        torch.distributed.all_gather_into_tensor(gathered_task_ids, task_ids, group=mpu.get_data_parallel_group())
    else:
        gathered_input_ids = input_ids
        gathered_output_ids = output_ids

    return gathered_input_ids.detach().cpu(), gathered_output_ids.detach().cpu(), gathered_task_ids.detach().cpu()


def gather_per_task_loss(per_task_losses, all_task_names, dtype, device):
    task_name_loss = {task_name: [] for task_name in all_task_names}
    task_name_validity = {task_name: [] for task_name in all_task_names}
    for task_name in all_task_names:
        losses = per_task_losses.get(task_name, [])

        # get max loss length
        gathered_loss_length = torch.tensor([len(losses)], dtype=torch.int64).to(device)
        torch.distributed.all_reduce(gathered_loss_length, op=torch.distributed.ReduceOp.MAX)

        cur_loss_length = len(losses)

        # padding losses
        if len(losses) < gathered_loss_length:
            losses += [0.0] * (gathered_loss_length - cur_loss_length)  # padding losses
        losses = torch.tensor(losses, dtype=dtype, device=device)
        is_valid = torch.tensor(
            [1] * cur_loss_length + [0] * (gathered_loss_length - cur_loss_length), dtype=torch.int64
        ).to(device)

        gather_loss = torch.empty(
            len(losses) * mpu.get_data_parallel_world_size(),  # loss 갯수
            dtype=dtype,
            device=device,
        )
        torch.distributed.all_gather_into_tensor(gather_loss, losses, group=mpu.get_data_parallel_group())

        gather_validity = torch.empty(
            len(losses) * mpu.get_data_parallel_world_size(),  # loss 갯수
            dtype=torch.int64,
            device=device,
        )
        torch.distributed.all_gather_into_tensor(gather_validity, is_valid, group=mpu.get_data_parallel_group())

        task_name_loss[task_name] = gather_loss
        task_name_validity[task_name] = gather_validity

    return task_name_loss, task_name_validity

def all_gather(input, dim=1):
    # List to hold gathered tensors from all ranks
    gathered_tensors = [torch.empty_like(input, dtype=input.dtype, device=input.device) for _ in range(mpu.get_sequence_parallel_world_size())]
    # Gather tensors from all ranks
    torch.distributed.all_gather(gathered_tensors, input, group=mpu.get_sequence_parallel_group())
    # Concatenate along the specified dimension
    output = torch.cat(gathered_tensors, dim=dim)
    return output

def scatter(input, dim=1):
    input_shape = input.size()
    global_seq_length = input_shape[dim]
    local_seq_length = global_seq_length // mpu.get_sequence_parallel_world_size()
    sub_seq_start = mpu.get_sequence_parallel_rank() * local_seq_length
    sub_seq_end = (mpu.get_sequence_parallel_rank() + 1) * local_seq_length

    # Create slices for each dimension
    slices = [slice(None)] * input.dim()
    slices[dim] = slice(sub_seq_start, sub_seq_end)

    output = input[tuple(slices)].contiguous()
    return output

class AllGatherScatter(torch.autograd.Function):
    """
    input shape : [Batch, local_seq_length, hidden_size] or [Batch, local_seq_length, num_heads, head_dim]
    forward : all-gather
    backward : scatter
    """
    @staticmethod
    def forward(ctx, input, dim=1):
        ctx.dim = dim
        return all_gather(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        return scatter(grad_output, dim=ctx.dim), None

class ScatterAllGather(torch.autograd.Function):
    """
    input shape : [Batch, global_seq_length, hidden_size] or [Batch, global_seq_length, num_heads, head_dim]
    forward : scatter
    backward : all-gather
    """
    @staticmethod
    def forward(ctx, input, dim=1):
        ctx.dim = dim
        return scatter(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        return all_gather(grad_output, dim=ctx.dim), None

def get_input_batch(batch, token_embedding_fn):
    input_batch = {}
    device = get_accelerator().current_device_name()

    try:
        seq_parallel_world_size = mpu.get_sequence_parallel_world_size()
    except:
        seq_parallel_world_size = 1

    if torch.distributed.is_initialized() and seq_parallel_world_size > 1:
        if "inputs_embeds" in batch:
            inputs_embeds = batch["inputs_embeds"].to(device)
        else:
            input_ids = batch["input_ids"].to(device)
            inputs_embeds = token_embedding_fn(input_ids)  # [B, T, D]

        seq_length = inputs_embeds.size(1)
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        if "position_ids" in batch:
            position_ids = batch["position_ids"]
        else:
            position_ids = torch.arange(seq_length).unsqueeze(0)

        if seq_length % mpu.get_sequence_parallel_world_size() != 0:
            # right padding
            pad_length = mpu.get_sequence_parallel_world_size() - (seq_length % mpu.get_sequence_parallel_world_size())
            inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, pad_length), value=0)
            labels = torch.nn.functional.pad(labels, (0, pad_length), value=-100)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_length), value=0)
            position_ids = torch.nn.functional.pad(position_ids, (0, pad_length), value=0)
            seq_length = inputs_embeds.size(1)

        sub_seq_length = seq_length // mpu.get_sequence_parallel_world_size()
        sub_seq_start = mpu.get_sequence_parallel_rank() * sub_seq_length
        sub_seq_end = (mpu.get_sequence_parallel_rank() + 1) * sub_seq_length

        inputs_embeds = ScatterAllGather.apply(inputs_embeds)
        position_ids = position_ids[:, sub_seq_start:sub_seq_end].to(device)
        labels = torch.roll(labels, shifts=-1)
        labels[:, -1] = -100
        labels = labels[:, sub_seq_start:sub_seq_end].to(device)
        attention_mask = attention_mask.to(device)
    else:
        if "inputs_embeds" in batch:
            inputs_embeds = batch["inputs_embeds"].to(device)
        else:
            input_ids = batch["input_ids"].to(device)
            inputs_embeds = token_embedding_fn(input_ids)  # [B, T, D]
        attention_mask = batch["attention_mask"].to(device)

        if batch["labels"] is not None:
            labels = batch["labels"].to(device)
        else:
            labels = None

        if "position_ids" in batch:
            position_ids = batch["position_ids"].to(device)
        else:
            position_ids = None

    input_batch["inputs_embeds"] = inputs_embeds
    input_batch["attention_mask"] = attention_mask
    input_batch["labels"] = labels
    input_batch["position_ids"] = position_ids
    return input_batch


def get_number_of_model_parameters(model: nn.Module) -> int:
    """
    the number of parameters in the model.

    Args:
        model (nn.Module): The model for which the number of parameters will be calculated.

    Returns:
        int: The number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters()) / 1e9


def diff_weight(model: nn.Module, checkpoint_path: str, adapter_name: str = "", is_adapter: bool = False):
    safe_tensor_list = glob.glob(os.path.join(checkpoint_path, adapter_name, "*.safetensors"))
    # check diff test
    tensors = {}
    for safe_tensor_path in safe_tensor_list:
        with safe_open(safe_tensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if is_adapter:
                    split_key = key.split(".")
                    in_key = ".".join(split_key[:-1] + ["default"] + split_key[-1:])
                else:
                    in_key = key
                tensors[in_key] = f.get_tensor(key)
    # diff load
    loaded_model = {k: v for k, v in model.named_parameters()}
    for key in tensors.keys():
        if key in loaded_model.keys():
            # assert torch.allclose(tensors[key], loaded_model[key].to(tensors[key].dtype)), f"key: {key} not equal"
            if not torch.allclose(tensors[key], loaded_model[key].to(tensors[key].dtype)):
                print(f"key: {key} not equal")
                print(f"tensors[key]: {tensors[key]}")
                print(f"loaded_model[key]: {loaded_model[key]}")
                import pdb

                pdb.set_trace()


def freeze_model(cfg: OmegaConf, model: nn.Module) -> nn.Module:
    """
    Freezes the model parameters.

    Args:
        cfg (OmegaConf): The configuration object containing the model freezing settings.
        model (nn.Module): The model to freeze.

    Returns:
        nn.Module: The frozen model.
    """
    if cfg.freeze:
        assert not cfg.memory_efficient.lora.use, "LoRA is not compatible with freezing the model"
        for _, parameter in model.named_parameters():
            parameter.requires_grad = False
    return model


def apply_memory_efficient_feature(cfg: OmegaConf, model: nn.Module) -> nn.Module:
    """
    Applies memory-efficient features to the model based on the provided configuration.

    Args:
        cfg (OmegaConf): The configuration object containing the memory-efficient feature settings.
        model (nn.Module): The model to which the memory-efficient features will be applied.

    Returns:
        nn.Module: The model with the memory-efficient features applied.

    Raises:
        ValueError: If gradient checkpointing is enabled but not supported by the model.
        AssertionError: If LoRA is enabled but incompatible with freezing the language or vision model.

    Notes:
        - If LoRA is enabled, the LoRA configuration is used to modify the model.
        - If gradient checkpointing is enabled, the model is modified to enable gradient checkpointing.
        - If model freezing is enabled, the model parameters are set to not require gradients.
    """
    if cfg.memory_efficient.lora.use and not isinstance(model, PeftModel):
        is_mora = cfg.memory_efficient.lora.get("is_mora", False)
        if is_mora:
            print_rank_0("=" * 30 + "applying MoRA")
            lora_config = LoraConfig(
                use_mora=True,
                mora_type=cfg.memory_efficient.lora.get("mora_type", 6),
                r=cfg.memory_efficient.lora.r,
                target_modules=r".*\.layers\.\d+\.(mlp\.fc\d+|self_attn\.out_proj)|.*(k_proj|v_proj|q_proj|o_proj|gate_proj|up_proj|down_proj|linear_1|linear_2)",
                lora_dropout=cfg.memory_efficient.lora.dropout,
            )
        else:
            print_rank_0("=" * 30 + "applying LoRA")
            lora_config = LoraConfig(
                r=cfg.memory_efficient.lora.r,
                lora_alpha=cfg.memory_efficient.lora.alpha,
                target_modules=r".*\.layers\.\d+\.(mlp\.fc\d+|self_attn\.out_proj)|.*(k_proj|v_proj|q_proj|o_proj|gate_proj|up_proj|down_proj|linear_1|linear_2)",
                lora_dropout=cfg.memory_efficient.lora.dropout,
                bias="none",
            )
        model = get_peft_model(model, lora_config)
        if is_local_main_process():
            model.print_trainable_parameters()

    if cfg.memory_efficient.gradient_checkpointing:
        print_rank_0(f"applying gradient_checkpointing to {cfg.get('pretrained_model_name_or_path', 'model')}")
        model.gradient_checkpointing_enable()
        if not model.is_gradient_checkpointing:
            raise ValueError("Gradient Checkpointing is not enabled!")

    return model


class FileSystemLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_dir = os.path.join(self.output_dir, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.eval_log_dir = os.path.join(self.log_dir, "eval")
        if not os.path.exists(self.eval_log_dir):
            os.makedirs(self.eval_log_dir, exist_ok=True)

        self.train_log_file = os.path.join(self.log_dir, "train.log")
        with open(self.train_log_file, "a") as f:
            f.write(f"Train starts at {datetime.datetime.now()}\n")
            f.flush()

    def log_eval(self, metric: dict, infer_result: list[dict], step: int):
        step_output_dir = os.path.join(self.eval_log_dir, str(step))
        if not os.path.exists(step_output_dir):
            os.makedirs(step_output_dir, exist_ok=True)

        metric_file = os.path.join(step_output_dir, "metric.json")
        result_file = os.path.join(step_output_dir, "result.json")

        with open(metric_file, "w") as f:
            f.write(json.dumps(metric, indent=4, sort_keys=True))
        with open(result_file, "w") as f:
            f.write(json.dumps(infer_result, indent=4))

        eval_result_dir = os.path.join(os.path.join(self.output_dir, f"steps_{step}"))
        if not os.path.exists(eval_result_dir):
            os.makedirs(eval_result_dir, exist_ok=True)
        os.system(f"cp -r {step_output_dir}/* {eval_result_dir}")

        print_rank_0(f"Evaluation results are saved in {step_output_dir} and {eval_result_dir}")

    def log_train(self, metric: dict, step: int):
        with open(self.train_log_file, "a") as f:
            f.write(f"Step: {step:07d}, {json.dumps(metric, sort_keys=True)}\n")
            f.flush()
