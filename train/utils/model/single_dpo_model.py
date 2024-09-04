# Reference:
# DPO paper https://arxiv.org/abs/2305.18290
# github code https://github.com/eric-mitchell/direct-preference-optimization

import torch
from torch import nn
from typing import Optional, Dict, List, Union, Tuple

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, pad_id: int, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != pad_id)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == pad_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -torch.nn.functional.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

class SingleDpoModel(nn.Module):

    def __init__(self, model, tokenizer, num_padding_at_beginning=0, beta=0.1):
        super().__init__()
        self.config = model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.model = model
        self.PAD_ID = -100
        self.beta = beta

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                ref_logps=None,
                use_cache=False):
        kwargs = dict()
        # print('inside model input {}'.format(input_ids))

        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2

        outputs = self.model(
            input_ids[:bs],
            past_key_values=past_key_values,
            attention_mask=attention_mask[:bs],
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs)
        
        #logps = _get_batch_logps(outputs['logits'], labels, self.PAD_ID, average_log_prob=False)
        chosen_logps = _get_batch_logps(outputs['logits'], labels[:bs], self.PAD_ID, average_log_prob=False)

        outputs_rej = self.model(
            input_ids[bs:],
            past_key_values=past_key_values,
            attention_mask=attention_mask[bs:],
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs)
        
        #logps = _get_batch_logps(outputs['logits'], labels, self.PAD_ID, average_log_prob=False)
        rejected_logps = _get_batch_logps(outputs_rej['logits'], labels[bs:], self.PAD_ID, average_log_prob=False)

        #chosen_logps = logps[:bs]
        #rejected_logps = logps[bs:]

        ref_chosen_logps = ref_logps[:bs]
        ref_rejected_logps = ref_logps[bs:]

        loss, chosen_rewards, rejected_rewards = dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, self.beta, reference_free=False)
        # print(loss)
        return {
            "loss": loss.mean(),
            "chosen_mean_scores": chosen_rewards,
            "rejected_mean_scores": rejected_rewards,
        }
    
    def forward_logits(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                ref_logps=None,
                use_cache=False):
        kwargs = dict()
        # print('inside model input {}'.format(input_ids))

        outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs)
        # print('inside model output {}'.format(outputs))
        # print('output shape is {}'.format(outputs.shape))
        
        logps = _get_batch_logps(outputs['logits'], labels, self.PAD_ID, average_log_prob=False)

        return logps
