from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("final_customed_qa")
class FinalCustomed_QARewardManager(AbstractRewardManager):
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
        self.num_examine = num_examine 
        self.compute_score = compute_score or default_compute_score

    def __call__(self, final_data: DataProto, return_dict: bool = True):
        """We will expand this function gradually based on the available datasets"""

        if 'rm_scores' in final_data.batch.keys():
            return final_data.batch['rm_scores']
        final_reward = []
        final_reward_tensor = torch.zeros_like(final_data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(final_data)):
            final_data_item = final_data[i]
            final_prompt_ids = final_data_item.batch['prompts']
            final_prompt_length = final_prompt_ids.shape[-1]
            valid_prompt_length = final_data_item.batch['attention_mask'][:final_prompt_length].sum()
            valid_prompt_ids = final_prompt_ids[-valid_prompt_length:]
            final_response_ids = final_data_item.batch['responses']
            valid_response_length = final_data_item.batch['attention_mask'][final_prompt_length:].sum()
            valid_final_response_ids = final_response_ids[:valid_response_length]

            final_response_text = self.tokenizer.decode(valid_final_response_ids, skip_special_tokens=True)      
                
            ground_truth = final_data_item.non_tensor_batch['reward_model']['ground_truth']
            # select rm_score
            data_source = final_data_item.non_tensor_batch['data_source']

            if "Qwen3" in self.tokenizer.name_or_path:
                final_score = self.compute_score(final_response_text, ground_truth=ground_truth, tokenizer=self.tokenizer)
            else:
                final_score = self.compute_score(final_response_text, ground_truth=ground_truth, tokenizer=None)
            
            final_reward.append(final_score)
            final_reward_tensor[i, valid_response_length - 1] = final_score


        if self.num_examine == 0:
            final_reward_extra_info = {}
            final_reward_extra_info['mean_final_score'] = sum(final_reward)/len(final_reward)

        else:
            final_reward_extra_info = defaultdict(list)
            final_reward_extra_info['mean_final_score'] = final_reward
        if return_dict:
            final_reward_results = {
                "reward_tensor": final_reward_tensor,
                "reward_extra_info": final_reward_extra_info,
            }
            return final_reward_results
        else:
            return final_reward_tensor
