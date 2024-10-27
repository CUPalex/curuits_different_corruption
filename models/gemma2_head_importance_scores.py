import os
from typing import Dict, Any, Optional, Tuple
import functools
import math
import warnings
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from torch import nn
from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb, repeat_kv

from curcuits_corrupted_examples.models.gemma2 import Gemma2Model
from curcuits_corrupted_examples.models.utils.importance_scores.saving_importance_scores import simple_save_imp_scores
from curcuits_corrupted_examples.models.utils.pruning.prune_heads import prune_heads
from curcuits_corrupted_examples.models.utils.data_processing.collate_function import collate_fn_everything
from curcuits_corrupted_examples.models.utils.data_processing.dataset import RequestDataset

class Gemma2HeadImportanceScores(Gemma2Model):
    def run(self, task, limit, batch_size, prune_using_imp_scores, prune_k, log_dir) -> None:
        self.task = task

        self.important_components_by_layer = prune_heads(prune_using_imp_scores, prune_k,
                                                         len(self.model.model.layers), self.model.model.layers[0].self_attn.num_heads, self.task.TOKEN_TYPES,
                                                         log_dir, is_split_by_token_type=True)

        dataset = RequestDataset(task, limit, corrupted=True, tokenizer=self.tokenizer)
        self.model_logs["first_3_dataset_examples"] = [dataset[i] for i in range(3)]
        self.num_requests = len(dataset)

        self.original_activations = {
            layer: {head: None for head in range(self.model.model.layers[0].self_attn.num_heads)}
            for layer in range(len(self.model.model.layers))
        }
        self.importance_scores = torch.zeros(len(self.model.model.layers),
                                self.model.model.layers[0].self_attn.num_heads,
                                task.TOKEN_TYPES)

        self.corrupted_activations = {
            layer: {head: None for head in range(self.model.model.layers[0].self_attn.num_heads)}
            for layer in range(len(self.model.model.layers))
        }

        self.save_importance_scores(task.loss, dataset, batch_size, log_dir)
        self.generate(dataset, batch_size)

    def save_importance_scores(self, loss_function, dataset, batch_size, log_dir) -> None:
        self.generate_mode = False
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=functools.partial(collate_fn_everything,
                                                             padding_left=False, use_corrupted_activations=True,
                                                             tokenizer=self.tokenizer),
                                num_workers=len(os.sched_getaffinity(0)) - 1)
        average_loss = 0
        examples = []
        num_predictive_inds_total = 0

        for batch in dataloader:
            inputs = batch[5].to("cuda") # TODO: change to handle different model device
            target_ids = batch[7].to("cuda") # TODO: change to handle different model device
            lens = batch[2]
            if len(examples) < 3:
                for i in range(inputs["input_ids"].shape[0]):
                    if len(examples) < 3:
                        examples.append(self.tokenizer.decode(inputs["input_ids"][i].detach().cpu()))

            corrupted_contexts = batch[6].to("cuda") # TODO: change to handle different model device

            # TODO: averaging only over non-padding tokens (need to think how to do it in omp scores compute too)
            # For token types computation
            self.tp_inds = self.task.get_token_types_for_contexts_with_targets(self.tokenizer, inputs["input_ids"].detach().cpu())
            self.corrupted_tp_inds = self.task.get_token_types_for_contexts_with_targets(self.tokenizer, corrupted_contexts["input_ids"].detach().cpu())
            assert (self.corrupted_tp_inds == self.tp_inds).all()

            self.is_corrupted_run = True
            with torch.no_grad():
                self.model(**corrupted_contexts)

            self.is_corrupted_run = False
            logits = self.model(**inputs).logits
            loss = loss_function(logits, target_ids, lens, self.tp_inds)
            loss.backward()

            for p in self.model.parameters():
                p.grad = None

            if "run_logs" not in self.model_logs:
                self.model_logs["run_logs"] = []


            predictive_inds = ((self.tp_inds == self.task.TARGET_TYPE) | (self.tp_inds == self.task.LAST_SEP_TYPE))
            predictive_inds[torch.arange(predictive_inds.shape[0]), (predictive_inds.cumsum(dim=-1) * predictive_inds).argmax(dim=-1)] = 0
            num_predictive_inds_total += torch.count_nonzero(predictive_inds).item()
            self.model_logs["run_logs"].append({
                "batch_len": len(batch), 
                "inputs_shape": inputs["input_ids"].shape,
                "target_shape": target_ids.shape,
                "last_3_queries": [self.tokenizer.decode(inputs["input_ids"][-i]) for i in range(min(inputs["input_ids"].shape[0], 3), 0, -1)],
                "last_1_corrupted_contexts_tokenized": [self.tokenizer.convert_ids_to_tokens(corrupted_contexts["input_ids"][-i]) for i in range(min(corrupted_contexts["input_ids"].shape[0], 1), 0, -1)],
                "last_1_queries_tokenized": [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][-i]) for i in range(min(inputs["input_ids"].shape[0], 1), 0, -1)],
                "last_1_tp_inds": [self.tp_inds[-i].tolist() for i in range(min(inputs["input_ids"].shape[0], 1), 0, -1)],
                "last_7_lens": lens.tolist()[-min(inputs["input_ids"].shape[0], 7):],
                "last_7_predictive_inds_in_input": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][predictive_inds].detach().cpu())[-min(inputs["input_ids"].shape[0], 7):],
                "last_7_correct_targets": self.tokenizer.convert_ids_to_tokens(target_ids.detach().cpu())[-min(inputs["input_ids"].shape[0], 7):],
                "last_7_correct_logits_probs": torch.nn.functional.softmax(logits.detach().cpu(), dim=-1)[predictive_inds, target_ids.detach().cpu()].tolist()[-min(inputs["input_ids"].shape[0], 7):],
                "shapes_of_logits_and_targets": (logits[predictive_inds, :].shape, target_ids.shape),
                "loss": loss.item()
            })
            average_loss += loss.item() * torch.count_nonzero(predictive_inds).item()

            del inputs
            del target_ids
            del logits
            del loss
            torch.cuda.empty_cache()

        self.model_logs["first_3_loader_no_generate_exampels"] = examples
        self.model_logs["loss"] = average_loss / num_predictive_inds_total
        simple_save_imp_scores(self.importance_scores, log_dir, len(dataset))

    def generate(self, dataset, batch_size) -> None:
        self.generate_mode = True
        logger.debug("Start generate part")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=functools.partial(collate_fn_everything,
                                                             padding_left=True, use_corrupted_activations=True,
                                                             tokenizer=self.tokenizer),
                                num_workers=len(os.sched_getaffinity(0)) - 1)
        sum_accuracy = 0
        continuations = []
        examples = []

        for batch in dataloader:
            inputs = batch[0].to("cuda") # TODO: change to handle different model device
            targets = batch[3]
            if len(examples) < 3:
                for i in range(inputs["input_ids"].shape[0]):
                    if len(examples) < 3:
                        examples.append(self.tokenizer.decode(inputs["input_ids"][i].detach().cpu()))

            corrupted_contexts = batch[4].to("cuda") # TODO: change to handle different model device

            self.tp_inds = self.task.get_token_types_for_contexts(self.tokenizer, inputs["input_ids"].detach().cpu())
            self.corrupted_tp_inds = self.task.get_token_types_for_contexts(self.tokenizer, corrupted_contexts["input_ids"].detach().cpu())
            assert (self.corrupted_tp_inds == self.tp_inds).all()

            self.is_corrupted_run = True
            with torch.no_grad():
                self.model(**corrupted_contexts)

            self.is_corrupted_run = False
            out = self.model.generate(**inputs, max_new_tokens=10)
            for i in range(out.shape[0]):
                continuation = self.tokenizer.decode(out[i][inputs["input_ids"][i].shape[0]:])
                if len(continuations) < 3:
                    continuations.append(continuation)
                sum_accuracy += int(continuation.strip().startswith(targets[i]))

            del inputs
            torch.cuda.empty_cache()

        self.model_logs["first_3_continuations"] = continuations
        self.model_logs["first_3_loader_generate_exampels"] = examples
        self.model_logs["accuracy"] = sum_accuracy / self.num_requests

    def break_into(self) -> None:
        self.hook_handles = []
        self.prev_forwards = []

        for layer in range(len(self.model.model.layers)):
            self.prev_forwards.append(self.model.model.layers[layer].self_attn.forward)
            forward_partial = functools.partial(self.attn_forward, layer=layer,
                                                self=self.model.model.layers[layer].self_attn,
                                                llama_model=self)
            self.model.model.layers[layer].self_attn.forward = forward_partial

    def break_out(self) -> None:
        for layer, f in enumerate(self.prev_forwards):
            forward_partial = functools.partial(self.prev_forwards[layer],
                                                self=self.model.model.layers[layer].self_attn)
            self.model.model.layers[layer].self_attn.forward = forward_partial
        for h in self.hook_handles:
            h.remove()

    @staticmethod
    def attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        layer: Optional[int] = None,
        llama_model: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # HERE WE CHANGE CODE
        # FROM THIS
        # attn_output = attn_output.view(bsz, q_len, -1)
        # attn_output = self.o_proj(attn_output)
        # TO THIS

        def compute_importance_score(grad, layer, head):
            assert not llama_model.generate_mode, "why are you in attn backward hook in generate mode??"
            if layer % 5 == 0 and head % 5 == 0:
                logger.debug(f"Backward hook attention layer {layer} head {head}")
            bsz, q_len, head_dim = grad.shape[0], grad.shape[1], grad.shape[2]
            assert len(grad.shape) == 3
            for tp in range(llama_model.task.TOKEN_TYPES):
                for batch_elem in range(bsz):
                    num_of_tokens_of_type_tp = (llama_model.tp_inds[batch_elem] == tp).sum()
                    importance_matrix = (
                        (llama_model.original_activations[layer][head][batch_elem][llama_model.tp_inds[batch_elem] == tp].view(num_of_tokens_of_type_tp, head_dim).to(grad.device) -
                        llama_model.corrupted_activations[layer][head][batch_elem][llama_model.tp_inds[batch_elem] == tp].view(num_of_tokens_of_type_tp, head_dim).to(grad.device)) @ 
                        grad.detach()[batch_elem][llama_model.tp_inds[batch_elem] == tp].view(num_of_tokens_of_type_tp, head_dim).transpose(-1, -2))
                    assert importance_matrix.shape == (num_of_tokens_of_type_tp, num_of_tokens_of_type_tp), importance_matrix.shape
                    importance_score = importance_matrix.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
                    llama_model.importance_scores[layer, head, tp] += torch.abs(importance_score).item()
            for l in range(len(llama_model.model.model.layers) - 1, layer + 1, -1):
                for p in llama_model.model.model.layers[l].parameters():
                    p.grad = None
            torch.cuda.empty_cache()

        head_outputs_together = []
        for cur_head in range(self.num_heads):
            if llama_model.generate_mode and past_key_value is not None and q_len == 1:
                # Skipping ablation in generate mode split by token type, in second pass, because of cache
                head_outputs_together.append(attn_output[:, :, cur_head, :] @ self.o_proj.weight.T[cur_head * self.head_dim : (cur_head + 1) * self.head_dim, :])
            elif llama_model.is_corrupted_run:
                head_outputs_together.append(attn_output[:, :, cur_head, :] @ self.o_proj.weight.T[cur_head * self.head_dim : (cur_head + 1) * self.head_dim, :])
                llama_model.corrupted_activations[layer][cur_head] = head_outputs_together[-1].detach().clone().cpu()
            elif llama_model.important_components_by_layer is None:
                head_outputs_together.append(attn_output[:, :, cur_head, :] @ self.o_proj.weight.T[cur_head * self.head_dim : (cur_head + 1) * self.head_dim, :])
                if not llama_model.generate_mode:
                    head_outputs_together[-1].register_hook(functools.partial(compute_importance_score, layer=layer, head=cur_head))
            else:
                head_outputs_together.append(attn_output[:, :, cur_head, :] @ self.o_proj.weight.T[cur_head * self.head_dim : (cur_head + 1) * self.head_dim, :])
                for tp in range(llama_model.task.TOKEN_TYPES):
                    if (cur_head, tp) not in llama_model.important_components_by_layer[layer]["attn"]:
                        head_outputs_together[-1][llama_model.tp_inds == tp] = llama_model.corrupted_activations[layer][cur_head][llama_model.tp_inds == tp].to(attn_output.device)
                if not llama_model.generate_mode and len(llama_model.important_components_by_layer[layer]["attn"]) > 0:
                    head_outputs_together[-1].register_hook(functools.partial(compute_importance_score, layer=layer, head=cur_head))

            if not llama_model.generate_mode and not llama_model.is_corrupted_run:
                llama_model.original_activations[layer][cur_head] = head_outputs_together[-1].detach().clone().cpu()

        assert len(head_outputs_together) == self.num_heads
        head_outputs_together = sum(head_outputs_together)

        # END OF CHANGED CODE

        if not output_attentions:
            attn_weights = None

        return head_outputs_together, attn_weights, past_key_value