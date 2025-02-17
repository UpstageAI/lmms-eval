import gc
import json
import math
import os
import time
from typing import Any, Dict, Literal

import torch
from deepspeed import initialize as deepspeed_initialize
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from hydra.utils import get_class
from omegaconf import ListConfig, OmegaConf
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig

import lmms_eval.models.docvision_utils.parallel_state as mpu
from lmms_eval.models.docvision_utils.pipeline_utils import (
    apply_memory_efficient_feature, diff_weight, freeze_model, print_rank_0)


def init_model(cfg: OmegaConf, resume_from_checkpoint: bool, checkpoint_path: str, test: bool = False):
    """
    Initializes the model, tokenizer, and image processor for the DocFM pipeline.

    Args:
        cfg (OmegaConf): The configuration object containing the pipeline settings.
        resume_from_checkpoint (bool): Whether to resume training from a checkpoint.
        checkpoint_path (str): The path to the checkpoint directory.
        test (bool, optional): Whether to run the model in test mode. Defaults to False.

    Returns:
        Tuple: A tuple containing the initialized model, tokenizer, and image processor.

    Raises:
        AssertionError: If test mode is enabled but resume_from_checkpoint is False.

    """
    lm_cfg = cfg.components.lm
    v_cfg = cfg.components.vision_encoder
    c_cfg = cfg.components.connector
    dtype_dict = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

    # check lm_model and v_model exists if resume_from_checkpoint is True
    load_from_deepspeed_checkpoint = False
    if resume_from_checkpoint and not os.path.exists(f"{checkpoint_path}/lm_model") and not os.path.exists(f"{checkpoint_path}/v_model"):
        load_from_deepspeed_checkpoint = True

    if test:
        mode = "test"
        _ = dtype_dict[cfg.test.precision]
    else:
        mode = "train"
        _ = dtype_dict[cfg.training.precision]

    print_rank_0("=" * 30 + "Initializing Model..")

    def init_language_model(pretrained_model_name_or_path: str):
        # Initialize Language Model
        language_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=lm_cfg.get("trust_remote_code", False)
        )
        # language_config.rope_scaling = {"type": "dynamic", "factor": 2.0}  # allows handling of longer inputs
        attn_implementation = lm_cfg.get("attn_implementation", "eager")
        if test:
            language_config.use_cache = True
        else:
            language_config.use_cache = False
        language_config._attn_implementation = attn_implementation

        language_model = get_class(lm_cfg.class_name).from_pretrained(
            pretrained_model_name_or_path,
            config=language_config,
            trust_remote_code=lm_cfg.get("trust_remote_code", False),
            use_auth_token=lm_cfg.get("use_auth_token", False),
            ignore_mismatched_sizes=lm_cfg.get("ignore_mismatched_sizes", False),
            cache_dir=cfg.cache_dir,
        )
        return language_model

    def init_vision_model(pretrained_model_name_or_path: str):
        vision_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        if test:
            vision_config.use_cache = True
        else:
            if hasattr(vision_config, "vision_config"):
                vision_config = AutoConfig.from_pretrained(pretrained_model_name_or_path).vision_config
            vision_config.use_cache = False
        vision_config.use_flash_attention_2 = v_cfg.get("use_flash_attention_2", False)
        vision_model = get_class(v_cfg.class_name).from_pretrained(
            pretrained_model_name_or_path,
            config=vision_config,
            trust_remote_code=v_cfg.trust_remote_code,
            cache_dir=cfg.cache_dir,
        )
        return vision_model

    def init_llava_config(pretrained_model_name_or_path, vision_config, text_config, tokenizer, resume):
        if resume:
            # Initialize LLava Model config
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        else:
            if cfg.components.lm.image_token in tokenizer.get_vocab():
                image_token_index = tokenizer.get_vocab()[cfg.components.lm.image_token]
            else:
                image_token_index = text_config.vocab_size
            # Initialize Llava Config
            config = get_class(cfg.components.architecture.config_name)(
                vision_config=vision_config,
                text_config=text_config,
                image_token_index=image_token_index,
                projector_hidden_act=c_cfg.projector_hidden_act,
                vision_feature_select_strategy=cfg.components.architecture.get(
                    "vision_feature_select_strategy", "full"
                ),
                vision_feature_layer=v_cfg.vision_feature_layer,
                trust_remote_code=True,
            )
        config.text_num_attention_heads = text_config.num_attention_heads
        config.text_hidden_size = text_config.hidden_size
        config.projector_implementation = cfg.components.architecture.get("projector_implementation", "mlp")
        config.projector_sampling_ratios = cfg.components.architecture.get("projector_sampling_ratios", [(2, 8192), (3, 65536)])
        if isinstance(config.projector_sampling_ratios, ListConfig):
            # Convert ListConfig[ListConfig[int]] -> List[List[int]] for JSON serialization
            config.projector_sampling_ratios = [list(x) for x in config.projector_sampling_ratios]

        if "Next" in v_cfg.image_processor_name:
            min_tiles = v_cfg.image_grid_pinpoints.get("min_tiles", 1)
            max_tiles = v_cfg.image_grid_pinpoints.get("max_tiles", 1)
            tile_range = range(max_tiles + 1)
            pinpoints = [
                [v_cfg.get("image_size", 384) * i, v_cfg.get("image_size", 384) * j]
                for i in tile_range
                for j in tile_range
                if min_tiles <= i * j and i * j <= max_tiles
            ]

            config.image_grid_pinpoints = pinpoints
        return config

    if resume_from_checkpoint and not load_from_deepspeed_checkpoint:
        # Sanity check
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")
        print_rank_0("=" * 30 + f"Resume from checkpoint: {checkpoint_path}")

        # Initialize Language Model
        language_model = init_language_model(f"{checkpoint_path}/lm_model")

        # Initialize Vision Model
        vision_model = init_vision_model(f"{checkpoint_path}/v_model")

        # Initialize Tokenizer
        tokenizer = init_tokenizer(cfg, mode)

        # Initialize Image Processor
        image_processor = init_image_processor(cfg, mode)

        # Initialize LLava Model config
        config = init_llava_config(checkpoint_path, vision_model.config, language_model.config, tokenizer, resume=True)

        # Initialize Llava Model
        model = get_class(cfg.components.architecture.class_name)(
            config, vision_tower=vision_model, multi_modal_projector=None, language_model=language_model
        )
        # Load Connector Weight
        connector_weight = torch.load(f"{checkpoint_path}/c_model", map_location="cpu")
        model.multi_modal_projector.load_state_dict(connector_weight)

        # Load Image Newline
        if os.path.exists(os.path.join(checkpoint_path, "image_newline.pt")):
            image_newline = torch.load(os.path.join(checkpoint_path, "image_newline.pt"), map_location="cpu")
            model.image_newline = image_newline

        # Load architecture weight
        if cfg.components.architecture.get("reload_architecture_weight", False):
            model.from_pretrained(checkpoint_path)

        # diff_weight(model, checkpoint_path, is_adapter=False) # for debugging

    else:
        print_rank_0("=" * 30 + "Init Model weights")

        # Initialize Language Model
        language_model = init_language_model(lm_cfg.pretrained_model_name_or_path)

        # Initialize Vision Model
        vision_model = init_vision_model(v_cfg.pretrained_model_name_or_path)

        # Initialize Tokenizer
        tokenizer = init_tokenizer(cfg, mode)

        # Initialize Image Processor
        image_processor = init_image_processor(cfg, mode)

        # Initialize Llava Config
        config = init_llava_config(None, vision_model.config, language_model.config, tokenizer, resume=False)

        # Initialize Llava Model
        model = get_class(cfg.components.architecture.class_name)(
            config, vision_tower=vision_model, multi_modal_projector=None, language_model=language_model
        )
        if c_cfg.get("pretrained_model_name_or_path", None):
            # Load Connector Weight
            connector_weight = torch.load(c_cfg.pretrained_model_name_or_path, map_location="cpu")
            model.multi_modal_projector.load_state_dict(connector_weight)

        if len(tokenizer) != model.config.text_config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))

    # check if peft exists
    #   - We use the same if-else statement as in line 128 to separate the application of memory-efficient algorithms
    #     from model initialization codes
    if resume_from_checkpoint and not load_from_deepspeed_checkpoint:
        if os.path.exists(os.path.join(checkpoint_path, "lm_adapter")):
            print_rank_0("=" * 30 + "Resume Language Model from PEFT checkpoint")
            model.language_model = PeftModel.from_pretrained(
                model.language_model, os.path.join(checkpoint_path, "lm_adapter")
            )
            diff_weight(model.language_model, checkpoint_path, "lm_adapter", is_adapter=True)
            if not test:
                model.language_model = model.language_model.merge_and_unload()
                model.language_model = apply_memory_efficient_feature(lm_cfg, model.language_model)
        else:
            model.language_model = apply_memory_efficient_feature(lm_cfg, model.language_model)

        if os.path.exists(os.path.join(checkpoint_path, "v_adapter")):
            print_rank_0("=" * 30 + "Resume Vision Tower from PEFT checkpoint")
            model.vision_tower = PeftModel.from_pretrained(
                model.vision_tower, os.path.join(checkpoint_path, "v_adapter")
            )
            diff_weight(model.vision_tower, checkpoint_path, "v_adapter", is_adapter=True)
            if not test:
                model.vision_tower = model.vision_tower.merge_and_unload()
                model.vision_tower = apply_memory_efficient_feature(v_cfg, model.vision_tower)
        else:
            model.vision_tower = apply_memory_efficient_feature(v_cfg, model.vision_tower)

        if os.path.exists(os.path.join(checkpoint_path, "c_adapter")):
            print_rank_0("=" * 30 + "Resume connector from PEFT checkpoint")
            model.multi_modal_projector = PeftModel.from_pretrained(
                model.multi_modal_projector, os.path.join(checkpoint_path, "c_adapter")
            )
            diff_weight(model.multi_modal_projector, checkpoint_path, "c_adapter", is_adapter=True)
            if not test:
                model.multi_modal_projector = model.multi_modal_projector.merge_and_unload()
                model.multi_modal_projector = apply_memory_efficient_feature(c_cfg, model.multi_modal_projector)
        else:
            model.multi_modal_projector = apply_memory_efficient_feature(c_cfg, model.multi_modal_projector)

    else:
        # apply peft
        model.language_model = apply_memory_efficient_feature(lm_cfg, model.language_model)
        model.vision_tower = apply_memory_efficient_feature(v_cfg, model.vision_tower)
        model.multi_modal_projector = apply_memory_efficient_feature(c_cfg, model.multi_modal_projector)

    # freeze Model
    model.language_model = freeze_model(lm_cfg, model.language_model)
    model.vision_tower = freeze_model(v_cfg, model.vision_tower)
    model.multi_modal_projector = freeze_model(c_cfg, model.multi_modal_projector)

    return model, tokenizer, image_processor, load_from_deepspeed_checkpoint


def init_tokenizer(cfg: OmegaConf, mode: Literal["train", "eval", "test"]):
    if mode == "test":
        resume_from_checkpoint = cfg.test.resume_from_checkpoint
        checkpoint_path = cfg.test.checkpoint_path
    else:
        resume_from_checkpoint = cfg.training.resume_from_checkpoint
        checkpoint_path = cfg.training.checkpoint_path

    # Initialize Tokenizer
    if resume_from_checkpoint:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided when resuming from checkpoint")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.components.lm.pretrained_model_name_or_path)

        if cfg.components.lm.image_token not in tokenizer.get_vocab():
            if cfg.components.lm.get("special_tokens", None):
                special_tokens = cfg.components.lm.special_tokens
                for special_token in special_tokens:
                    if special_token not in tokenizer.vocab:
                        tokenizer.add_tokens(special_token, special_tokens=True)
            else:
                tokenizer.add_tokens(cfg.components.lm.image_token, special_tokens=True)
            # 64-element alignment would yield better results
            # reference : https://twitter.com/cHHillee/status/1630274804795445248
            unused_tokens = 64 - len(tokenizer) % 64
            for i in range(unused_tokens):
                tokenizer.add_tokens(f"<unused_special_token_{i}>", special_tokens=True)
    tokenizer.padding_side = cfg.components.lm.get("padding_side", "right")
    return tokenizer


def init_image_processor(cfg: OmegaConf, mode: Literal["train", "eval", "test"]):
    if mode == "test":
        resume_from_checkpoint = cfg.test.resume_from_checkpoint
        checkpoint_path = cfg.test.checkpoint_path
    else:
        resume_from_checkpoint = cfg.training.resume_from_checkpoint
        checkpoint_path = cfg.training.checkpoint_path

    v_cfg = cfg.components.vision_encoder
    if v_cfg.get("force_image_processor_init", False):
        resume_from_checkpoint = False
    if resume_from_checkpoint:
        # Initialize Image Processor
        image_processor = get_class(v_cfg.image_processor_name).from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
    else:
        if v_cfg.get("image_size", None):
            if "Next" not in v_cfg.image_processor_name:
                image_processor = get_class(v_cfg.image_processor_name).from_pretrained(
                    v_cfg.pretrained_model_name_or_path, size={"height": v_cfg.image_size, "width": v_cfg.image_size}
                )
            else:
                min_tiles = v_cfg.image_grid_pinpoints.get("min_tiles", 1)
                max_tiles = v_cfg.image_grid_pinpoints.get("max_tiles", 1)
                tile_range = range(max_tiles + 1)
                pinpoints = [
                    [v_cfg.get("image_size", 384) * i, v_cfg.get("image_size", 384) * j]
                    for i in tile_range
                    for j in tile_range
                    if min_tiles <= i * j and i * j <= max_tiles
                ]

                image_processor = get_class(v_cfg.image_processor_name).from_pretrained(
                    v_cfg.pretrained_model_name_or_path,
                    crop_size={"height": v_cfg.image_size, "width": v_cfg.image_size},
                    image_grid_pinpoints=pinpoints,
                    size={"shortest_edge": v_cfg.image_size},
                )
        else:
            image_processor = get_class(v_cfg.image_processor_name).from_pretrained(
                v_cfg.pretrained_model_name_or_path
            )

    return image_processor

def precision_setting(precision: str, ds_config: Dict[str, Any]):
    # precision setting
    if precision == "fp16":
        ds_config["fp16"]["enabled"] = True
        ds_config["bf16"]["enabled"] = False
    elif precision == "bf16":
        ds_config["fp16"]["enabled"] = False
        ds_config["bf16"]["enabled"] = True
    elif precision == "fp32":
        ds_config["fp16"]["enabled"] = False
        ds_config["bf16"]["enabled"] = False
    else:
        raise ValueError(f"Invalid precision {precision}")
    return ds_config

def init_for_test(cfg: OmegaConf) -> Dict[str, Any]:
    """
    Initializes the necessary components for testing.

    Args:
        cfg (OmegaConf): The configuration object.

    Returns:
        Dict[str, Any]: A dictionary containing the initialized components:
            - "model" (Any): The initialized model.
            - "image_processor" (Any): The initialized image processor.
            - "tokenizer" (Any): The initialized tokenizer.
            - "test_data_loaders" (Any): The initialized test data loaders.
    """
    with open(cfg.ds_config, "r") as file:
        ds_config = json.load(file)
    ds_config["train_micro_batch_size_per_gpu"] = cfg.data.test.batch_size
    ds_config["gradient_accumulation_steps"] = 1
    ds_config["train_batch_size"] = cfg.data.test.batch_size * torch.distributed.get_world_size()

    # precision setting:
    ds_config = precision_setting(cfg.test.precision, ds_config)

    # Initialize Model
    model, tokenizer, image_processor, load_from_deepspeed_checkpoint = init_model(
        cfg=cfg,
        resume_from_checkpoint=True,
        checkpoint_path=cfg.test.checkpoint_path,
        test=True,
    )

    # Initialize DataModule
    print_rank_0("=" * 30 + "Initializing DataModule..")
    data_module = get_class(cfg.data.test.modules.data_module)(
        cfg, tokenizer=tokenizer, image_processor=image_processor
    )

    # Initialize DataLoader
    print_rank_0("=" * 30 + "Initializing DataLoader..")
    test_data_loader = data_module.create_dataloader(mode="test")

    # optimizer config -> not used but required by deepspeed to initialize
    ds_config["optimizer"]["params"]["lr"] = 1e-4 # dummy value
    del ds_config["scheduler"]

    # load deepspeed checkpoint
    if load_from_deepspeed_checkpoint:
        model = load_state_dict_from_zero_checkpoint(model, cfg.test.checkpoint_path)

    # Prepare Everything
    print_rank_0("=" * 30 + "Deepspeed prepare..")
    model_engine, _, _, _ = deepspeed_initialize(
        model=model, model_parameters=model.parameters(), mpu=mpu, config=ds_config
    )

    # cache flush
    gc.collect()
    torch.cuda.empty_cache()

    init_modules = {
        "model": model_engine,
        "image_processor": image_processor,
        "tokenizer": tokenizer,
        "test_data_loader": test_data_loader,
    }
    return init_modules

def init_for_training(cfg: OmegaConf) -> Dict[str, Any]:
    """
    Initializes the training process by setting up the model, data, optimizer, scheduler, and other necessary components.

    Args:
        cfg (OmegaConf): The configuration object containing the training settings.

    Returns:
        init_modules (Dict[str, Any]): A dictionary containing the initialized modules and dataloaders.

    Raises:
        None

    Detailed Description:
        This function performs the following steps to initialize the training process:
        1. Initializes the model, tokenizer, and image processor using the `init_model` function.
        2. Initializes the DataModule using the specified data module class and passes the tokenizer and image processor.
        3. Initializes the DataLoader using the `get_dataloader` method of the DataModule.
        4. Prints the shape of the input_ids, attention_mask, and labels tensors of the first batch in the train_dataloader.
        5. Initializes the deepspeed config for training. The global batch size is divided by the sequence parallel size.
        6. Returns a dictionary containing the initialized modules and dataloaders.

    Example:
        >>> cfg = get_config()
        >>> init_modules = init_for_training(cfg)
    """

    with open(cfg.ds_config, "r") as file:
        ds_config = json.load(file)
    ds_config["train_micro_batch_size_per_gpu"] = cfg.data.train.batch_size
    ds_config["gradient_accumulation_steps"] = cfg.training.gradient_accumulation_steps
    ds_config["train_batch_size"] = (
        cfg.data.train.batch_size * cfg.training.gradient_accumulation_steps * torch.distributed.get_world_size()
    )

    # precision setting
    ds_config = precision_setting(cfg.training.precision, ds_config)

    # for deepspeed zero3
    _ = HfDeepSpeedConfig(ds_config)

    ds_config["train_batch_size"] = ds_config["train_batch_size"] // mpu.get_sequence_parallel_world_size()

    # Initialize Model
    model_time = time.time()
    model, tokenizer, image_processor, load_from_deepspeed_checkpoint = init_model(
        cfg, cfg.training.resume_from_checkpoint, cfg.training.checkpoint_path
    )

    model_time = time.time() - model_time
    data_time = time.time()

    # Initialize DataModule
    print_rank_0("=" * 30 + "Initializing DataModule..")
    data_module = get_class(cfg.data.train.modules.data_module)(
        cfg, tokenizer=tokenizer, image_processor=image_processor
    )

    # Initialize DataLoader
    print_rank_0("=" * 30 + "Initializing DataLoader..")
    train_dataloader = data_module.create_dataloader(mode="train")
    eval_dataloader = data_module.create_dataloader(mode="eval")
    torch.distributed.barrier()

    # test one samples
    for _, batch in enumerate(train_dataloader):
        for k in ["input_ids", "attention_mask", "labels"]:
            print_rank_0(f"{k} shape : {batch[k].shape}")
        break

    # optimizer config
    ds_config["optimizer"]["params"]["lr"] = cfg.training.lr
    data_time = time.time() - data_time
    deepspeed_time = time.time()

    # scheduler config
    if cfg.training.get("resume_from_checkpoint", False) and cfg.training.get("resume_steps", 0) > 0:
        trained_batches = cfg.training.get("resume_steps", 0)
    else:
        trained_batches = 0

    total_steps = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps) + trained_batches
    warmup_steps = int(total_steps * cfg.training.lr_warmup_ratio)
    ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps

    print_rank_0(
        "length of dataloader : "
        f"{len(train_dataloader)} total steps : {total_steps}, warmup steps: {warmup_steps}, max_lr : {cfg.training.lr}"
    )

    # load deepspeed checkpoint
    if load_from_deepspeed_checkpoint:
        model = load_state_dict_from_zero_checkpoint(model, cfg.training.checkpoint_path)

    # Prepare Everything
    print_rank_0("=" * 30 + "Deepspeed prepare..")
    model_engine, optimizer, _, _ = deepspeed_initialize(
        model=model, model_parameters=model.parameters(), mpu=mpu, config=ds_config
    )

    # cache flush
    gc.collect()
    torch.cuda.empty_cache()
    deepspeed_time = time.time() - deepspeed_time

    init_modules = {
        "model": model_engine,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "train_dataloader": train_dataloader,
        "eval_dataloader": eval_dataloader,
        "ds_config": ds_config,
    }
    return init_modules
