import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import torch
import numpy as np

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    calculate_log_probs: bool = False

class ReaVerLLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        ver_actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.ver_actor_rollout_wg = ver_actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor, respond_by: str='reasoner') -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        if respond_by == 'reasoner':
            responses_str_ls = [resp.split('</search>')[0] + '</search>'
                    if '</search>' in resp 
                    else resp.split('</answer>')[0] + '</answer>'
                    if '</answer>' in resp 
                    else resp
                    for resp in responses_str]
        elif respond_by == 'verifier':
            responses_str_ls = []
            for resp in responses_str:
                if '</response>' in resp:
                    responses_str_ls.append(resp.split('</response>')[0] + '</response>')
                elif '</seleected_doc>' in resp:
                    responses_str_ls.append(resp.split('</selected_doc>')[0] + '</selected_doc>')
                elif '</final_answer>' in resp:
                    responses_str_ls.append(resp.split('</final_answer>')[0] + '</final_answer>')
                elif '</verify>' in resp:
                    responses_str_ls.append(resp.split('</verify>')[0] + '</verify>')
                else:
                    responses_str_ls.append(resp)

                    
            # responses_str = [resp.split('</final_answer>')[0] + '</final_answer>'
            #                  if '</final_answer>' in resp 
            #                  else resp.split('</verify>')[0] + '</verify>'
            #                  if '</verify>' in resp 
            #                  else resp
            #                  for resp in responses_str]
            
        else:
            raise ValueError(f"Unknown respond_by: {respond_by}")

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str_ls)
        return responses, responses_str_ls

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids.long()

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                              self_next_obs_ids: torch.Tensor=None, 
                              op_next_obs_ids: torch.Tensor=None) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        tensor_list = [
                rollings.batch['input_ids'],
                cur_responses
            ] if cur_responses is not None else [
                rollings.batch['input_ids']
            ]
        if self_next_obs_ids is not None:
            tensor_list.append(self_next_obs_ids)
        if op_next_obs_ids is not None:
            tensor_list.append(op_next_obs_ids)
        new_input_ids = self.tensor_fn.concatenate_with_padding(tensor_list)    
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                self_info: torch.Tensor = None,
                op_info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response] if response is not None else [prompt]
        tensors_with_mask = [prompt_with_mask, response] if response is not None else [prompt_with_mask]
        if self_info is not None:
            tensors.append(self_info)
            info_mask = torch.full(self_info.size(), pad_id, dtype=self_info.dtype, device=self_info.device) # information mask
            tensors_with_mask.append(info_mask)
        if op_info is not None:
            tensors.append(op_info)
            info_mask = torch.full(op_info.size(), pad_id, dtype=op_info.dtype, device=op_info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _info_masked_concatenate_with_padding_and_probs(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                self_info: torch.Tensor = None,
                op_info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response] if response is not None else [prompt]
        tensors_with_mask = [prompt_with_mask, response] if response is not None else [prompt_with_mask]
        if self_info is not None:
            tensors.append(self_info)
            info_mask = torch.full(self_info.size(), pad_id, dtype=self_info.dtype, device=self_info.device) # information mask
            tensors_with_mask.append(info_mask)
        if op_info is not None:
            tensors.append(op_info)
            info_mask = torch.full(op_info.size(), pad_id, dtype=op_info.dtype, device=op_info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          self_next_obs_ids: torch.Tensor = None,
                          op_next_obs_ids: torch.Tensor = None, keys=['responses, responses_with_info_mask']) -> Dict:
        """Update right side state."""
        responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
            right_side['responses'],
            right_side['responses_with_info_mask'],
            cur_responses,
            self_next_obs_ids,
            op_next_obs_ids,
            pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _update_right_side_with_probs(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          self_next_obs_ids: torch.Tensor = None,
                          op_next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
            right_side['responses'],
            right_side['responses_with_info_mask'],
            cur_responses,
            self_next_obs_ids,
            op_next_obs_ids,
            pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _ver_generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.ver_actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.ver_actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.ver_actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output


    def run_llm_loop(self, gen_batch, ver_gen_batch, initial_input_ids: torch.Tensor, initial_ver_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        original_ver_left_side = {'input_ids': initial_ver_input_ids[:, -self.config.max_start_length:]}
        original_ver_right_side = {'responses': initial_ver_input_ids[:, []], 'responses_with_info_mask': initial_ver_input_ids[:, []]}
        
        if self.config.calculate_log_probs:
            original_right_side['rollout_log_probs'] = initial_input_ids[:, []]
            original_ver_right_side['rollout_log_probs'] = initial_ver_input_ids[:, []]


        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rea_rollings = gen_batch

        ver_active_mask = torch.ones(ver_gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        ver_turns_stats = torch.ones(ver_gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        ver_valid_action_stats = torch.zeros(ver_gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        ver_valid_search_stats = torch.zeros(ver_gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        ver_active_num_list = [ver_active_mask.sum().item()]
        ver_rollings = ver_gen_batch
        ver_responses_ids, ver_self_output_obs_ids = None, None
        rea_ver_output_obs, rea_self_output_obs, dones, valid_action, is_search = [], [], [], [], []
        ver_rea_output_obs, ver_self_output_obs, ver_dones, ver_valid_action, ver_is_search = [], [], [], [], []
        self.rollout_size = gen_batch.batch['input_ids'].shape[0]
        self.final_text_sum = ['' for _ in range(self.rollout_size)]
        self.search_result_cache = [[] for _ in range(self.rollout_size)]
        # Main generation loop
        # self.actor_rollout_wg.start_sharding_manager()
        # self.ver_ctor_rollout_wg.start_sharding_manager()
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rea_rollings.batch = self.tensor_fn.cut_to_effective_len(
                rea_rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            ver_rollings.batch = self.tensor_fn.cut_to_effective_len(
                ver_rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rea_rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rea_rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rea_rollings_active)
            
            meta_info = gen_output.meta_info            
            rea_responses_ids, rea_responses_str = self._postprocess_responses(gen_output.batch['responses'])
            rea_responses_ids, rea_responses_str = self.tensor_fn._example_level_pad(rea_responses_ids, rea_responses_str, active_mask)
            
            # Execute in environment and process observations
            rea_ver_output_obs, rea_self_output_obs, dones, valid_action, is_search = self.execute_reasoner_predictions(
                predictions=rea_responses_str, 
                verifier_states=[ver_rea_output_obs, ver_self_output_obs, ver_dones, ver_valid_action, ver_is_search], 
                active_mask=active_mask
            )
            
            # Update Veirfier States before LLM generation
            rea_ver_output_obs_ids = self._process_next_obs(rea_ver_output_obs)
            rea_self_output_obs_ids = self._process_next_obs(rea_self_output_obs)
            # ver_rollings = self._update_rolling_state(
            #     rollings=ver_rollings,
            #     cur_responses=ver_responses_ids,
            #     self_next_obs_ids=ver_self_output_obs_ids,
            #     op_next_obs_ids=rea_ver_output_obs_ids
            # )
            # original_ver_right_side = self._update_right_side(
            #     right_side=original_ver_right_side,
            #     cur_responses=ver_responses_ids,
            #     self_next_obs_ids=ver_self_output_obs_ids,
            #     op_next_obs_ids=rea_ver_output_obs_ids
            # )

            ver_rollings = self._update_rolling_state(
                rollings=ver_rollings,
                cur_responses=None,
                self_next_obs_ids=None,
                op_next_obs_ids=rea_ver_output_obs_ids
            )
            original_ver_right_side = self._update_right_side(
                right_side=original_ver_right_side,
                cur_responses=None,
                self_next_obs_ids=None,
                op_next_obs_ids=rea_ver_output_obs_ids
            )
            
            
            ver_rollings_active = DataProto.from_dict({
                k: v[ver_active_mask] for k, v in ver_rollings.batch.items()
            })

            ver_gen_output = self._ver_generate_with_gpu_padding(ver_rollings_active)

            ver_meta_info = ver_gen_output.meta_info
            ver_responses_ids, ver_responses_str = self._postprocess_responses(ver_gen_output.batch['responses'], respond_by='verifier')
            ver_responses_ids, ver_responses_str = self.tensor_fn._example_level_pad(ver_responses_ids, ver_responses_str, ver_active_mask)

            ver_rea_output_obs, ver_self_output_obs, ver_dones, ver_valid_action, ver_is_search = self.execute_verifier_predictions(
                predictions=ver_responses_str, 
                reasoner_states=[rea_ver_output_obs, dones, valid_action, is_search],
                active_mask=ver_active_mask
            )

            ver_rea_output_obs_ids = self._process_next_obs(ver_rea_output_obs)
            ver_self_output_obs_ids = self._process_next_obs(ver_self_output_obs)
            rea_rollings = self._update_rolling_state(
                rollings=rea_rollings,
                cur_responses=rea_responses_ids,
                self_next_obs_ids=rea_self_output_obs_ids,
                op_next_obs_ids=ver_rea_output_obs_ids
            )

            original_right_side = self._update_right_side(
                right_side=original_right_side,
                cur_responses=rea_responses_ids,
                self_next_obs_ids=rea_self_output_obs_ids,
                op_next_obs_ids=ver_rea_output_obs_ids
            )

            ver_rollings = self._update_rolling_state(
                rollings=ver_rollings,
                cur_responses=ver_responses_ids,
                self_next_obs_ids=ver_self_output_obs_ids,
                op_next_obs_ids=None
            )
            original_ver_right_side = self._update_right_side(
                right_side=original_ver_right_side,
                cur_responses=ver_responses_ids,
                self_next_obs_ids=ver_self_output_obs_ids,
                op_next_obs_ids=None
            )
            

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)


            ver_curr_active_mask = torch.tensor([not done for done in ver_dones], dtype=torch.bool)
            ver_active_mask = ver_active_mask * ver_curr_active_mask
            ver_active_num_list.append(ver_active_mask.sum().item())
            ver_turns_stats[ver_curr_active_mask] += 1
            ver_valid_action_stats += torch.tensor(ver_valid_action, dtype=torch.int)
            ver_valid_search_stats += torch.tensor(ver_is_search, dtype=torch.int)

            
        # final LLM rollout
        if active_mask.sum():
            rea_rollings.batch = self.tensor_fn.cut_to_effective_len(
                rea_rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rea_rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rea_rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rea_rollings_active)

            meta_info = gen_output.meta_info            
            rea_responses_ids, rea_responses_str = self._postprocess_responses(gen_output.batch['responses'])
            rea_responses_ids, rea_responses_str = self.tensor_fn._example_level_pad(rea_responses_ids, rea_responses_str, active_mask)

            # # Execute in environment and process observations
            rea_ver_output_obs, rea_self_output_obs, dones, valid_action, is_search = self.execute_reasoner_predictions(
                rea_responses_str, 
                verifier_states=[ver_rea_output_obs, ver_self_output_obs, dones, valid_action, is_search], 
                active_mask=active_mask, 
                do_search=False
            )
            
            rea_ver_output_obs_ids = self._process_next_obs(rea_ver_output_obs)
            # ver_rollings = self._update_rolling_state(
            #     rollings=ver_rollings,
            #     cur_responses=ver_responses_ids,
            #     self_next_obs_ids=ver_self_output_obs_ids,
            #     op_next_obs_ids=rea_ver_output_obs_ids
            # )
            # original_ver_right_side = self._update_right_side(
            #     right_side=original_ver_right_side,
            #     cur_responses=ver_responses_ids,
            #     self_next_obs_ids=ver_self_output_obs_ids,
            #     op_next_obs_ids=rea_ver_output_obs_ids
            # )

            ver_rollings = self._update_rolling_state(
                rollings=ver_rollings,
                cur_responses=None,
                self_next_obs_ids=None,
                op_next_obs_ids=rea_ver_output_obs_ids
            )
            original_ver_right_side = self._update_right_side(
                right_side=original_ver_right_side,
                cur_responses=None,
                self_next_obs_ids=None,
                op_next_obs_ids=rea_ver_output_obs_ids
            )
            

            ver_rollings_active = DataProto.from_dict({
                k: v[ver_active_mask] for k, v in ver_rollings.batch.items()
            })
            
            ver_gen_output = self._ver_generate_with_gpu_padding(ver_rollings_active)

            ver_meta_info = ver_gen_output.meta_info
            ver_responses_ids, ver_responses_str = self._postprocess_responses(ver_gen_output.batch['responses'], respond_by='verifier')
            ver_responses_ids, ver_responses_str = self.tensor_fn._example_level_pad(ver_responses_ids, ver_responses_str, ver_active_mask)

            ver_rea_output_obs, ver_self_output_obs, ver_dones, ver_valid_action, ver_is_search = self.execute_verifier_predictions(
                ver_responses_str, 
                reasoner_states=[rea_ver_output_obs, dones, valid_action, is_search],  
                active_mask=ver_active_mask, 
                do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                rea_responses_ids,
            )

            original_ver_right_side = self._update_right_side(
                original_ver_right_side,
                ver_responses_ids,
            )
        
        # self.actor_rollout_wg.exit_sharding_manager()
        # self.ver_ctor_rollout_wg.exit_sharding_manager()
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        ver_meta_info['ver_turns_stats'] = ver_turns_stats.tolist()
        ver_meta_info['ver_active_mask'] = ver_active_mask.tolist()
        ver_meta_info['ver_valid_action_stats'] = ver_valid_action_stats.tolist()
        ver_meta_info['ver_valid_search_stats'] = ver_valid_search_stats.tolist()
        self.final_text_sum = np.array(self.final_text_sum)
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        assert original_right_side['responses'].dtype == torch.int64
        assert original_ver_right_side['responses'].dtype == torch.int64
        final_reasoner_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        final_verifier_output = self._compose_final_output(original_ver_left_side, original_ver_right_side, meta_info)
        return final_reasoner_output, final_verifier_output

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        final_output = DataProto.from_dict(final_output)
        final_output.non_tensor_batch['final_text'] = self.final_text_sum
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_reasoner_predictions(self, predictions: List[str], verifier_states: List[Any], active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents, index = self.postprocess_reasoner_predictions(predictions, verifier_states)
        ver_next_obs, self_next_obs, dones, valid_action, is_search, final_text = [], [], [], [], [], []
        
        search_queries = [(content, i) for action, content, i in zip(cur_actions, contents, index) if action == 'search']
        answers = [content for action, content in zip(cur_actions, contents) if action == 'answer']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                ver_next_obs.append('')
                self_next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                final_text.append('')
            else:
                if action == 'answer':
                    answer_to_pop = answers.pop(0).strip()
                    think_match = re.search(r'<think>(.*?)</think>', predictions[i], re.DOTALL)
                    infos_to_pop = ''
                    if think_match:
                        reasoning_path = think_match.group(1).strip()
                        infos_to_pop += f'The reasoning path by the reasoner is: {reasoning_path}\n'
                    infos_to_pop += f'The answer by the reasoner is: {answer_to_pop}'
                    ver_next_obs.append(f'\n<answer>{infos_to_pop}</answer>\n')
                    self_next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    final_text.append(f'[Reasoner]: Reasoner gave an answer: "{answer_to_pop}"\n')
                elif action == 'search':
                    think_match = re.search(r'<think>(.*?)</think>', predictions[i], re.DOTALL)
                    infos_to_pop = ''
                    if think_match:
                        reasoning_path = think_match.group(1).strip()
                        infos_to_pop += f'The reasoning path by the reasoner is: {reasoning_path}'
                    search_queries_to_pop = search_queries.pop(0)[0].strip()
                    search_results_to_pop = search_results.pop(0).strip()

                    if search_results_to_pop == '':
                        infos_to_pop += f'The search query raised by the reasoner is: "{search_queries_to_pop}"\nThe search engine did not return any new documents.'
                        self_next_obs.append('The search engine did not return any new documents. Therefore, I need to generate more diverse search queries.\n')
                        ver_next_obs.append(f'\n<information>{infos_to_pop}</information>\n')
                        final_text.append(f"[Reasoner]: Reasoner raised a search query: '{search_queries_to_pop}'\nThe search engine did not return any new documents.\n")
                    else:
                        infos_to_pop += f'The search query raised by the reasoner is: "{search_queries_to_pop}"\nThe search engine returned the following result:\n{search_results_to_pop}'
                        ver_next_obs.append(f'\n<information>{infos_to_pop}</information>\n')
                        self_next_obs.append('')
                        final_text.append(f"[Reasoner]: Reasoner raised a search query: '{search_queries_to_pop}'\nThe search engine returned the following result:\n{search_results_to_pop}\n")
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    infos_to_pop = 'The reasoner failed to generate valid search queries or answers.'
                    think_match = re.search(r'<think>(.*?)</think>', predictions[i], re.DOTALL)
                    if think_match:
                        reasoning_path = think_match.group(1).strip()
                        infos_to_pop += f'The reasoning path by the reasoner is: "{reasoning_path}"'
                    
                    ver_next_obs.append(f'\n<information>{infos_to_pop}</information>\n')
                    self_next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    final_text.append('')
            
        assert len(search_results) == 0
        assert len(answers) == 0
        self.update_final_text(final_text)
        return ver_next_obs, self_next_obs, dones, valid_action, is_search
        

    def execute_verifier_predictions(self, predictions: List[str], reasoner_states: List[Any], active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_verifier_predictions(predictions, reasoner_states)
        rea_next_obs, self_next_obs, dones, valid_action, is_search, final_text = [], [], [], [], [], []
        
        infos = [content for action, content in zip(cur_actions, contents) if action == 'response']
        final_answers = [content for action, content in zip(cur_actions, contents) if action == 'final_answer']

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                self_next_obs.append('')
                rea_next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                final_text.append('')
            else:
                if action == 'response':
                    infos_to_pop = infos.pop(0).strip()
                    rea_next_obs.append(f'\n<information>{infos_to_pop}</information>\n')
                    self_next_obs.append('')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    final_text.append(f'[Verifier]: {infos_to_pop}\n')
                elif action == 'final_answer' and reasoner_states[1][i] and reasoner_states[2][i]:
                    # the reasoner gave an answer and the action is valid
                    answer_to_drop = final_answers.pop(0)
                    rea_next_obs.append('')
                    self_next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    final_text.append(f'[Verifier]: The final answer by the verifier is: "{answer_to_drop}"\n')
                elif action == 'final_answer' and (not reasoner_states[1][i]) and reasoner_states[2][i]:
                    # the reasoner have not answered yet and the action is valid
                    # the verifier should not give the final answer yet
                    documents_match = re.search(r'<information>(.*?)</information>', reasoner_states[0][i], re.DOTALL)
                    if 'Doc 1(Title:' in documents_match.group(1).strip():
                        doc_split = documents_match.group(1).strip().split('Doc 1(Title:')
                        doc_to_pop = 'Doc 1(Title:' + ''.join(doc_split[1:])
                    else:
                        doc_to_pop = 'No new document is provided. I need to generate more diverse search queries.'
                        
                    rea_next_obs.append(f'<information>{doc_to_pop.strip()}</information>\n')
                    self_next_obs.append(f'\nI should not give the final answer yet since the reasoner has not finished. I should only give my final answer after the reasoner give its answer between <answer> and </answer>.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(1)
                    answer_to_drop = final_answers.pop(0)
                    final_text.append(f'[Verifier]: The final answer by the verifier is: "{answer_to_drop}"\n')
                elif action == 'final_answer' and not (reasoner_states[1][i] and reasoner_states[2][i]):
                    rea_next_obs.append('')
                    self_next_obs.append(f'\nThe reasoner failed to generate valid search query, I should wait for the reasoner to generate valid query or answer and then perform verification.\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    answer_to_drop = final_answers.pop(0)
                    final_text.append('')
                elif action == 'final_answer':
                    raise ValueError('Unkown error in final answer from verifier')
                else:
                    if (not reasoner_states[1][i]) and reasoner_states[2][i]:
                        documents_match = re.search(r'<information>(.*?)</information>', reasoner_states[0][i], re.DOTALL)
                        if 'Doc 1(Title:' in documents_match.group(1).strip():
                            doc_split = documents_match.group(1).strip().split('Doc 1(Title:')
                            doc_to_pop = 'Doc 1(Title:' + ''.join(doc_split[1:])
                        else:
                            doc_to_pop = 'No new document is provided. I need to generate more diverse search queries.'
                        rea_next_obs.append(f'\n<information>{doc_to_pop.strip()}</information>\n')
                        self_next_obs.append(f'\nMy previous action is invalid. \
If I want to respond to the resasoner, I should put the summarization of documents between <information> and </information> and the verification between <verify> and </verify>. \
If I want to give the final answer, I should put the answer between <final_answer> and </final_answer>. Let me try again.\n')
                    else:
                        rea_next_obs.append(f'\n<information>The verifier\'s previous action is invalid.</information>\n')
                        self_next_obs.append(f'\nMy previous action is invalid. \
If I want to respond to the resasoner, I should put the summarization of documents between <information> and </information> and the verification between <verify> and </verify>. \
If I want to give the final answer, I should put the answer between <final_answer> and </final_answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    final_text.append('')
            
        assert len(final_answers) == 0
            
        self.update_final_text(final_text)
        return rea_next_obs, self_next_obs, dones, valid_action, is_search
    
    def update_verifier_states_with_reasoner_output(self, ver_rollings: DataProto, next_obs_ids: torch.Tensor) -> DataProto:
        """
        Update verifier states with reasoner output.
        Args:
            ver_rollings: Current verifier rolling states
        Returns:
            Updated verifier rollings
        """
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            ver_rollings.batch['input_ids'],
            next_obs_ids
        ])
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_ver_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_ver_rollings.meta_info.update(ver_rollings.meta_info)

        return new_ver_rollings
    
    def postprocess_reasoner_predictions(self, predictions: List[Any], verifier_states: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        idx = []
        length = len(predictions)
        for i in range(length):
            prediction = predictions[i]
            if isinstance(prediction, str): # for llm output
                response_match = re.search(r'<search>(.*?)</search>', prediction, re.DOTALL)
                if response_match:
                    content = response_match.group(1).strip()
                    action = 'search'
                else:
                    final_answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
                    if final_answer_match:
                        content = final_answer_match.group(1).strip()
                        action = 'answer'
                    else:
                        content = ''
                        action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            idx.append(i)
            actions.append(action)
            contents.append(content)
            
        return actions, contents, idx
    
    def postprocess_verifier_predictions(self, predictions: List[Any], reasoner_states: List[Any]) -> Tuple[List[int], List[bool]]:
        actions = []
        contents = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            if isinstance(prediction, str): # for llm output
                response_match = re.search(r'<response>(.*?)</response>', prediction, re.DOTALL)
                select_doc_match = re.search(r'<selected_doc>(.*?)</selected_doc>', prediction, re.DOTALL)
                if response_match or select_doc_match:
                    verification_match = re.search(r'<verify>(.*?)</verify>', prediction, re.DOTALL)
                    content = ''
                    if select_doc_match:
                        selected_doc = select_doc_match.group(1).strip()
                        if len(selected_doc) > 20:
                            document = selected_doc
                        else:
                            document = self.find_document(reasoner_states[0][i], selected_doc)
                        if document:
                            content = content + f'The verifier identifies the following document as the most important: {document}\n'
                    if verification_match or response_match:
                        content += 'The opinion of the verifier is as follows:\n'

                        content += f'{verification_match.group(1).strip()}\n' if verification_match else ''
                        content += f'{response_match.group(1).strip()}\n' if response_match else ''
                    action = 'response'
                else:
                    final_answer_match = re.search(r'<final_answer>(.*?)</final_answer>', prediction, re.DOTALL)
                    if final_answer_match:
                        if not (reasoner_states[1][i] and reasoner_states[2][i]):
                            content_beforefinal = prediction.split('<final_answer>')[0]
                            if len(content_beforefinal.strip()) < 10:
                                content = final_answer_match.group(1).strip()
                                action = 'final_answer'
                            else:
                                content = content_beforefinal.strip()
                                action = 'response'
                        else:
                            content = final_answer_match.group(1).strip()
                            action = 'final_answer'
                    else:
                        content = ''
                        action = None

            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[tuple] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        queries_woidx = [q[0] for q in queries]
        idx = [q[1] for q in queries]
        results = self._batch_search(queries_woidx)['result']
        p2s_results = [self._passages2string(result, i, self.config.topk) for result, i in zip(results, idx)]
        return p2s_results

    def _batch_search(self, queries):
        max_cache = max([len(cached) for cached in self.search_result_cache])
        topk = max_cache + self.config.topk
        
        payload = {
            "queries": queries,
            "topk": topk,
            "return_scores": True
        }

        # return requests.post(self.config.search_url, json=payload).json()
        try:
            response = requests.post(self.config.search_url, json=payload).json()
            return response
        except (requests.RequestException, ValueError, requests.exceptions.JSONDecodeError) as e:
            
            print(f"Full batch search failed: {e}. Falling back to chunked search (max 200 per batch).")

            all_responses = []
            chunk_size = 200

            for i in range(0, len(queries), chunk_size):
                chunk = queries[i:i + chunk_size]
                chunk_payload = {
                    "queries": chunk,
                    "topk": topk,
                    "return_scores": True
                }
                resp = requests.post(self.config.search_url, json=chunk_payload).json()    
                all_responses.extend(resp["result"])

            return {"result": all_responses}     

    def _passages2string(self, retrieval_result, index, topk=3):
        format_reference = ''
        search_history = self.search_result_cache[index]
        valid_doc = 0
        for idx, doc_item in enumerate(retrieval_result):
            if valid_doc >= topk:
                break
            content = doc_item['document']['contents']
            title = content.split("\n")[0]  
            text = "\n".join(content.split("\n")[1:])
            text_to_cache = text[:20]
            if text_to_cache not in search_history:
                self.search_result_cache[index].append(text_to_cache)
                valid_doc += 1
                format_reference += f"Doc {valid_doc}(Title: {title}) {text}\n"
            else:
                continue
        return format_reference

    def find_document(self, document_text, doc_name):
        content = document_text.strip()
        match = re.search(r'<information>(.*?)</information>', document_text, re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()
        
        if len(doc_name) > 10:
            return None
        else:
            numbers = re.findall(r'\d+', doc_name)
            doc_num = [int(n) for n in numbers]
        
        content_list = re.split(r'(?=Doc \d+\(Title:)', content)
        doc_dict = {}
        for c in content_list:
            if c.strip().startswith("Doc"):
                dn = int(c.strip()[4])
                dc = c.strip()[5:]
                doc_dict[dn] = dc
        
        doc2return = ''
        for num in doc_num:
            if num in doc_dict:
                doc2return += doc_dict[num]
        
        if len(doc2return) == 0:
            return None
        else:
            return doc2return
        
    def update_final_text(self, final_text_add):
        length = self.rollout_size
        for i in range(length):
            self.final_text_sum[i] += final_text_add[i]