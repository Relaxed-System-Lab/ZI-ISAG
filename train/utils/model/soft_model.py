# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


class SoFTModel(nn.Module):

    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.model = base_model

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                labels=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                inputs_embeds=None,
                use_cache=False):
        outputs = self.model(input_ids=input_ids, labels=labels, output_hidden_states=True, return_dict=True)
        shift_labels = labels[..., 1:].contiguous()
        logits = outputs["logits"][..., :-1, :].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = outputs["loss"]

        layer_outs = outputs['hidden_states']
        for layer in [10, 20, 30, 40]:
            logits = self.model.lm_head(self.model.model.norm(layer_outs[layer]))[..., :-1, :]
            loss += loss_fct(logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        return (loss,) + outputs[1:]
