from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("adv_customed_qa")
class AdvCustomed_QARewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        **kwargs,

    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.out_reward_type = kwargs.get('out_reward_type', 'em')
        self.structure_format_score = kwargs.get('structure_format_score', 0.2)
        self.final_format_score = kwargs.get('final_format_score', 0.0)
        self.retrieval_score = kwargs.get('retrieval_score', 0.0)
        self.ver_clarity_beta = kwargs.get('ver_clarity_beta', 1.0)
        self.adv_out_weight = kwargs.get('adv_out_weight', 0.1)
        self.adv_ver_weight = kwargs.get('adv_ver_weight', 0.1)
        self.compute_score = compute_score or default_compute_score

    def __call__(self, rea_data: DataProto, ver_data: DataProto, return_dict: bool = True):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in rea_data.batch.keys() and 'rm_scores' in ver_data.batch.keys():
            return rea_data.batch['rm_scores'], ver_data.batch['rm_scores']

        rea_reward_tensor = torch.zeros_like(rea_data.batch['responses'], dtype=torch.float32)
        ver_reward_tensor = torch.zeros_like(ver_data.batch['responses'], dtype=torch.float32)
        
        already_print_data_sources = {}

        rea_reward_dict, ver_reward_dict = {}, {}

        assert len(rea_data) == len(ver_data), "The length of rea_data and ver_data should be the same."
        for i in range(len(rea_data)):
            rea_data_item = rea_data[i]  # DataProtoItem
            ver_data_item = ver_data[i]

            rea_prompt_ids = rea_data_item.batch['prompts']
            ver_prompt_ids = ver_data_item.batch['prompts']

            rea_prompt_length = rea_prompt_ids.shape[-1]
            ver_prompt_length = ver_prompt_ids.shape[-1]

            valid_rea_prompt_length = rea_data_item.batch['attention_mask'][:rea_prompt_length].sum()
            valid_rea_prompt_ids = rea_prompt_ids[-valid_rea_prompt_length:]
            valid_ver_prompt_length = ver_data_item.batch['attention_mask'][:ver_prompt_length].sum()
            valid_ver_prompt_ids = ver_prompt_ids[-valid_ver_prompt_length:]

            rea_response_ids = rea_data_item.batch['responses']
            valid_rea_response_length = rea_data_item.batch['attention_mask'][rea_prompt_length:].sum()
            valid_rea_response_ids = rea_response_ids[:valid_rea_response_length]
            ver_response_ids = ver_data_item.batch['responses']
            valid_ver_response_length = ver_data_item.batch['attention_mask'][ver_prompt_length:].sum()
            valid_ver_response_ids = ver_response_ids[:valid_ver_response_length]            
                
            # decode
            rea_sequences = torch.cat((valid_rea_prompt_ids, valid_rea_response_ids))
            rea_sequences_str = self.tokenizer.decode(rea_sequences)
            ver_sequences = torch.cat((valid_ver_prompt_ids, valid_ver_response_ids))
            ver_sequences_str = self.tokenizer.decode(ver_sequences)

            kwargs = {}
            kwargs['tokenizer'] = self.tokenizer
            kwargs['rea_data'], kwargs['ver_data'] = {}, {}
            # kwargs['rea_data']['response_str'], kwargs['ver_data']['response_str'] = rea_sequences_str, ver_sequences_str
            kwargs['rea_data']['entropy'], kwargs['ver_data']['entropy'] = rea_data_item.batch['entropys'], ver_data_item.batch['entropys']
            kwargs['rea_data']['response_ids'], kwargs['ver_data']['response_ids'] = rea_response_ids, ver_response_ids
            kwargs['rea_data']['info_mask'], kwargs['ver_data']['info_mask'] = rea_data_item.batch['info_mask'][rea_prompt_length:], ver_data_item.batch['info_mask'][ver_prompt_length:]
            kwargs['rea_data']['response_length'], kwargs['ver_data']['response_length'] = valid_rea_response_length, valid_ver_response_length

            kwargs['ver_data']['ver_clarity_beta'] = self.ver_clarity_beta
            kwargs['adv_out_weight'], kwargs['adv_ver_weight'] = self.adv_out_weight, self.adv_ver_weight

            ground_truth = rea_data_item.non_tensor_batch['reward_model']['ground_truth']
            assert rea_data_item.non_tensor_batch['id'] == ver_data_item.non_tensor_batch['id'], "The id of rea_data and ver_data should be the same."
            
            # select rm_score
            rea_data_source = rea_data_item.non_tensor_batch['data_source']
            ver_data_source = ver_data_item.non_tensor_batch['data_source']
            # compute_score_fn = self.compute_score(data_source)

            rea_score, ver_score, *optional_output = self.compute_score(
                rea_solution_str=rea_sequences_str, 
                ver_solution_str=ver_sequences_str, 
                ground_truth=ground_truth, 
                out_reward_type=self.out_reward_type,
                structure_format_score=self.structure_format_score,
                final_format_score=self.final_format_score,
                retrieval_score=self.retrieval_score,
                **kwargs)
            
            for key in rea_score.keys():
                if key not in rea_reward_dict.keys():
                    rea_reward_dict[key] = [rea_score[key]]
                else:
                    rea_reward_dict[key].append(rea_score[key])
            for key in ver_score.keys():
                if key not in ver_reward_dict.keys():
                    ver_reward_dict[key] = [ver_score[key]]
                else:
                    ver_reward_dict[key].append(ver_score[key])

            if self.num_examine == 0:
                rea_reward_tensor[i, valid_rea_response_length - 1] = rea_score['rea_total_score']
                ver_reward_tensor[i, valid_ver_response_length - 1] = ver_score['ver_total_score']
            else:
                rea_reward_tensor[i, valid_rea_response_length - 1] = rea_score['rea_raw_score']
                ver_reward_tensor[i, valid_ver_response_length - 1] = ver_score['ver_raw_score']
            # all_scores.append(score)

            if rea_data_source not in already_print_data_sources:
                already_print_data_sources[rea_data_source] = 0
            if ver_data_source not in already_print_data_sources:
                already_print_data_sources[ver_data_source] = 0

            if already_print_data_sources[rea_data_source] < self.num_examine:
                already_print_data_sources[rea_data_source] += 1
                print(rea_sequences_str)
            if already_print_data_sources[ver_data_source] < self.num_examine:
                already_print_data_sources[ver_data_source] += 1
                print(ver_sequences_str)
            

        if self.num_examine == 0:
            rea_reward_extra_info, ver_reward_extra_info = {}, {}
            for key in rea_reward_dict.keys():
                rea_reward_extra_info[f'mean_{key}'] = sum(rea_reward_dict[key]) / len(rea_reward_dict[key])
            for key in ver_reward_dict.keys():
                if isinstance(ver_reward_dict[key][0], torch.Tensor):
                    ver_reward_extra_info[key] = torch.stack(ver_reward_dict[key], dim=0)
                else:
                    ver_reward_extra_info[f'mean_{key}'] = sum(ver_reward_dict[key]) / len(ver_reward_dict[key])
        else:
            rea_reward_extra_info, ver_reward_extra_info = defaultdict(list), defaultdict(list)
            for key in rea_reward_dict.keys():
                rea_reward_extra_info[f'mean_{key}'] = rea_reward_dict[key]
            for key in ver_reward_dict.keys():
                if isinstance(ver_reward_dict[key][0], torch.Tensor):
                    continue
                ver_reward_extra_info[f'mean_{key}'] = ver_reward_dict[key]
        
        if return_dict:
            rea_reward_results = {
                "reward_tensor": rea_reward_tensor,
                "reward_extra_info": rea_reward_extra_info,
            }
            ver_reward_results = {
                "reward_tensor": ver_reward_tensor,
                "reward_extra_info": ver_reward_extra_info,
            }
            return rea_reward_results, ver_reward_results
        else:
            return rea_reward_tensor, ver_reward_tensor
        