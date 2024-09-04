# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from .dpo_model import DpoModel
from .reward_model import RewardModel
from ..utils import load_state_dict_into_model

import time


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False,
                    eval=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True,)
    # print(model_config)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config, trust_remote_code=True)
    else:
        # model = model_class.from_pretrained(
        #     model_name_or_path,
        #     from_tf=bool(".ckpt" in model_name_or_path),
        #     config=model_config,
        #     use_flash_attention_2=True,)    
        if not eval:
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config, 
                trust_remote_code=True)
        else:
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config, 
                trust_remote_code=True, 
                device_map='auto',
                torch_dtype="auto")
        
    # print(model)
    
    # print(model.config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # model_embeds = model._resize_token_embeddings(int(
    #     8 *
    #     math.ceil(len(tokenizer) / 8.0)))
    
    # print(model_embeds.weight.shape[0])

    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    
    print('length of tokenizer is {}'.format(len(tokenizer)))
    print('resize_token_embeddings is {}'.format(int(8 *math.ceil(len(tokenizer) / 8.0))))

    return model

def create_hf_model_flash_attn(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # print(model_config)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config, trust_remote_code=True)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            use_flash_attention_2=True,
            trust_remote_code=True) 
        
    # print(1)
    print(model)
    
    # print(model.config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # model_embeds = model._resize_token_embeddings(int(
    #     8 *
    #     math.ceil(len(tokenizer) / 8.0)))
    
    # print('model_embeds is {}'.format(model_embeds))
    # time.sleep(10)
    
    # print(model_embeds.weight.shape[0])

    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    
    print('resize_token_embeddings is {}'.format(int(8 *math.ceil(len(tokenizer) / 8.0))))

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False,
                        zero_stage=0):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)
    end = time.time()
    # if torch.distributed.get_rank() == 0:
    #     print(f"> Creating model from_config took {end - start} seconds")

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)
    
    # print(critic_model)

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
        end = time.time()
        if torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(critic_model,
                                   model_ckpt_state_dict,
                                   "",
                                   zero_stage=zero_stage)
        end = time.time()
        if torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model

def create_critic_model_flash_attn(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False,
                        zero_stage=0):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    critic_model = create_hf_model_flash_attn(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)
    end = time.time()
    # if torch.distributed.get_rank() == 0:
    #     print(f"> Creating model from_config took {end - start} seconds")

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)
    
    # print(critic_model)

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
        end = time.time()
        if torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(critic_model,
                                   model_ckpt_state_dict,
                                   "",
                                   zero_stage=zero_stage)
        end = time.time()
        if torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model


def create_dpo_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False,
                        zero_stage=0,
                        beta=0.1,
                        for_eval=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    policy_model = create_hf_model(AutoModelForCausalLM, model_name_or_path, tokenizer,
                                    ds_config, rlhf_training, disable_dropout, eval=for_eval)
    end = time.time()
    # if torch.distributed.get_rank() == 0:
    #     print(f"> Creating model from_config took {end - start} seconds")

    policy_model = DpoModel(
        policy_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        beta=beta)
    
    # print(policy_model)

    return policy_model


def create_hf_model_pp(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    # print(model_config)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        # model = model_class.from_pretrained(
        #     model_name_or_path,
        #     from_tf=bool(".ckpt" in model_name_or_path),
        #     config=model_config,
        #     use_flash_attention_2=True,)    
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            device_map='auto')
        
    # print(model)
    
    # print(model.config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # model_embeds = model._resize_token_embeddings(int(
    #     8 *
    #     math.ceil(len(tokenizer) / 8.0)))
    
    # print(model_embeds.weight.shape[0])

    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    
    print('length of tokenizer is {}'.format(len(tokenizer)))
    print('resize_token_embeddings is {}'.format(int(8 *math.ceil(len(tokenizer) / 8.0))))

    return model
