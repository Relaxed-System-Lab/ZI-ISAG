# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


class Classifier(nn.Module):

    def __init__(self, base_model, n_class=1):
        super().__init__()
        self.config = base_model.config
        self.v_head = nn.Linear(self.config.hidden_size, n_class, bias=False, dtype=torch.bfloat16)
        self.base_model = base_model

    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                labels=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        transformer_outputs = self.base_model(
            input_ids)
        y = self.v_head(transformer_outputs[0][0, -1, :]).to(torch.float32)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(torch.unsqueeze(y, 0), labels)
        return {
            "loss": loss,
            "logit": y,
        }

    def forward_value(self,
                input_ids=None,
                labels=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                inputs_embeds=None,
                use_cache=False):
        transformer_outputs = self.base_model(
            input_ids)
        y = self.v_head(transformer_outputs[0][:, -1, :])

        return {
            "logit": y,
        }

