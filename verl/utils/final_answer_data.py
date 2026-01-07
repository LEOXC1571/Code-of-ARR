import torch
import numpy as np
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def get_prompt_templete(question, rea_response_str, ver_response_str, is_qwen3=False):
    if not is_qwen3:
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        if ver_response_str is not None:
            full_prompt = f"<|im_start|>\nsystem{system_prompt}<|im_end|>\n<|im_start|>user\nThe reasoning path of the reasoner is {rea_response_str}\nThe reasoning path of the verifier is {ver_response_str}\nAnswer the following question. Prior to this, both the reasoner and the verifier have conducted reasoning and verification regarding this question. You are required to provide the answer based on their respective reasoning processes. You should directly answer the question between without detailed illustrations.\nQuestion is {question}\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>\nsystem{system_prompt}<|im_end|>\n<|im_start|>user\nThe rollout text of the reasoner and verifier is:\n{rea_response_str}\nAnswer the following question. Prior to this, both the reasoner and the verifier have conducted reasoning and verification regarding this question. You are required to provide the answer based on their respective reasoning processes. You should directly answer the question between without detailed illustrations.\nQuestion is {question}\n<|im_end|>\n<|im_start|>assistant\n"
    else:
        system_prompt = ""
        if ver_response_str is not None:
            full_prompt = f"<|im_start|>user\nThe reasoning path of the reasoner is {rea_response_str}\nThe reasoning path of the verifier is {ver_response_str}\nAnswer the following question. Prior to this, both the reasoner and the verifier have conducted reasoning and verification regarding this question. You are required to provide the answer based on their respective reasoning processes. Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>.\nQuestion is {question}\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>user\nThe rollout text of the reasoner and verifier is:\n{rea_response_str}\nAnswer the following question. Prior to this, both the reasoner and the verifier have conducted reasoning and verification regarding this question. You are required to provide the answer based on their respective reasoning processes. Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>.\nQuestion is {question}\n<|im_end|>\n<|im_start|>assistant\n"
    return full_prompt

def get_rollout_text(rea_batch, ver_batch, tokenizer):
    if "final_text" in rea_batch.non_tensor_batch.keys():
        return rea_batch.non_tensor_batch['final_text']
    else:
        rea_response_ids = rea_batch.batch['responses']
        rea_resopnse_length = rea_batch.batch['response_mask'].sum(-1)
        rea_response_texts = tokenizer.batch_decode(rea_response_ids, skip_special_tokens=True)
        ver_response_ids = ver_batch.batch['responses']
        ver_resopnse_length = ver_batch.batch['response_mask'].sum(-1)
        ver_response_texts = tokenizer.batch_decode(ver_response_ids, skip_special_tokens=True)
        return (rea_response_texts, ver_response_texts)


# 批量处理版本
def create_batch_training_data(rea_batch, ver_batch, non_tensor_batch, tokenizer, tensor_keys_to_pop=None, non_tensor_keys_to_pop=None, max_start_length=4096):
    batch_dict = {}
    device = rea_batch.batch['responses'].device
    batch_size = rea_batch.batch['responses'].shape[0]
    questions = rea_batch.non_tensor_batch['question']
    
    total_rollout_text = get_rollout_text(rea_batch, ver_batch, tokenizer)

    merged_prompt_texts = []

    for i in range(batch_size):
        question = questions[i]
        if isinstance(total_rollout_text, tuple):
            rea_response_str = total_rollout_text[0][i]
            ver_response_str = total_rollout_text[1][i]
        else:
            rea_response_str = total_rollout_text[i]
            ver_response_str = None
            
        if 'Qwen3' in tokenizer.name_or_path:
            prompt_text = get_prompt_templete(question, rea_response_str, ver_response_str, is_qwen3=True)
        else:
            prompt_text = get_prompt_templete(question, rea_response_str, ver_response_str)
        merged_prompt_texts.append(prompt_text)
    
    tokenized_merged_prompts = tokenizer(merged_prompt_texts, padding=True, padding_side='left')
    pad_token_id = tokenizer.pad_token_ids

    if not tensor_keys_to_pop or not non_tensor_keys_to_pop:
        tensor_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
        non_tensor_keys_to_pop = ['raw_prompt_ids', 'tools_kwargs', 'interaction_kwargs', 'index']
    if 'raw_prompt_ids' in non_tensor_keys_to_pop:
        non_tensor_keys_to_pop.remove('raw_prompt_ids')
    batch_tensor_keys = ['prompts', 'input_ids', 'attention_mask', 'info_mask', 'position_ids', 'response_mask']
    gen_batch_tensor_keys = ['input_ids', 'attention_mask', 'position_ids']
    batch_non_tensor_batch_keys = ['id', 'question', 'golden_answers', 'data_source', 'ability', 'reward_model', 'extra_info', 'metadata', 'uid']
    gen_batch_non_tensor_batch_keys = ['raw_prompt_ids', 'tools_kwargs', 'interaction_kwargs', 'index']
    
    batch_dict['input_ids'] = torch.tensor(tokenized_merged_prompts['input_ids'], device=device)[:, -max_start_length:]
    batch_dict['attention_mask'] = torch.tensor(tokenized_merged_prompts['attention_mask'], device=device)[:, -max_start_length:]
    batch_dict['position_ids'] = compute_position_id_with_mask(batch_dict['attention_mask'])[:, -max_start_length:]
    batch_dict['tools_kwargs'] = non_tensor_batch['tools_kwargs']
    batch_dict['interaction_kwargs'] = non_tensor_batch['interaction_kwargs']
    batch_dict['index'] = non_tensor_batch['index']
    # final_gen_batch = DataProto.from_single_dic÷t(batch_dict)
    
    batch_dict['id'] = rea_batch.non_tensor_batch['id']
    batch_dict['question'] = questions
    batch_dict['golden_answers'] = rea_batch.non_tensor_batch['golden_answers']
    batch_dict['data_source'] = rea_batch.non_tensor_batch['data_source']
    batch_dict['ability'] = rea_batch.non_tensor_batch['ability']
    batch_dict['reward_model'] = rea_batch.non_tensor_batch['reward_model']
    batch_dict['uid'] = rea_batch.non_tensor_batch.get('uid', None)
    final_batch = DataProto.from_single_dict(batch_dict)
    final_gen_batch = final_batch.pop(batch_keys=tensor_keys_to_pop, non_tensor_batch_keys=non_tensor_keys_to_pop)

    return final_batch, final_gen_batch


def create_batch_validation_data(rea_batch, ver_batch, test_batch, test_gen_batch, tokenizer, tensor_keys_to_pop=None, non_tensor_keys_to_pop=None, max_start_length=4096):
    batch_dict = {}
    device = rea_batch.batch['responses'].device
    batch_size = rea_batch.batch['responses'].shape[0]
    questions = test_batch.non_tensor_batch['question']

    total_rollout_text = get_rollout_text(rea_batch, ver_batch, tokenizer)
    
    merged_prompt_texts = []
    num_questions = questions.shape[0]
    if num_questions < batch_size:
        questions = np.pad(questions, ((0, batch_size - num_questions),), mode='edge')
    
    for i in range(batch_size):
        question = questions[i]
        if isinstance(total_rollout_text, tuple):
            rea_response_str = total_rollout_text[0][i]
            ver_response_str = total_rollout_text[1][i]
        else:
            rea_response_str = total_rollout_text[i]
            ver_response_str = None
        if 'Qwen3' in tokenizer.name_or_path:
            prompt_text = get_prompt_templete(question, rea_response_str, ver_response_str, is_qwen3=True)
        else:
            prompt_text = get_prompt_templete(question, rea_response_str, ver_response_str)
        merged_prompt_texts.append(prompt_text)
    
    tokenized_merged_prompts = tokenizer(merged_prompt_texts, padding=True, padding_side='left')
    pad_token_id = tokenizer.pad_token_ids

    if not tensor_keys_to_pop and not non_tensor_keys_to_pop:
        tensor_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
        non_tensor_keys_to_pop = ['raw_prompt_ids', 'tools_kwargs', 'interaction_kwargs', 'index']
    if 'raw_prompt_ids' in non_tensor_keys_to_pop:
        non_tensor_keys_to_pop.remove('raw_prompt_ids')
    batch_tensor_keys = ['prompts', 'input_ids', 'attention_mask', 'info_mask', 'position_ids', 'response_mask']
    gen_batch_tensor_keys = ['input_ids', 'attention_mask', 'position_ids']
    batch_non_tensor_batch_keys = ['id', 'question', 'golden_answers', 'data_source', 'ability', 'reward_model', 'extra_info', 'metadata', 'uid']
    gen_batch_non_tensor_batch_keys = ['raw_prompt_ids', 'tools_kwargs', 'interaction_kwargs', 'index']
    
    batch_dict['input_ids'] = torch.tensor(tokenized_merged_prompts['input_ids'], device=device)[:, -max_start_length:]
    batch_dict['attention_mask'] = torch.tensor(tokenized_merged_prompts['attention_mask'], device=device)[:, -max_start_length:]
    batch_dict['position_ids'] = compute_position_id_with_mask(batch_dict['attention_mask'])[:, -max_start_length:]
    batch_dict['tools_kwargs'] = test_gen_batch.non_tensor_batch['tools_kwargs']
    batch_dict['interaction_kwargs'] = test_gen_batch.non_tensor_batch['interaction_kwargs']
    batch_dict['index'] = test_batch.non_tensor_batch['index']
    # final_gen_batch = DataProto.from_single_dict(batch_dict)
    
    batch_dict['id'] = test_batch.non_tensor_batch['id']
    batch_dict['question'] = questions
    batch_dict['golden_answers'] = test_batch.non_tensor_batch['golden_answers']
    batch_dict['data_source'] = test_batch.non_tensor_batch['data_source']
    batch_dict['ability'] = test_batch.non_tensor_batch['ability']
    batch_dict['reward_model'] = test_batch.non_tensor_batch['reward_model']
    # batch_dict['uid'] = test_batch.non_tensor_batch.get('uid', None)
    if num_questions < batch_size:
        for key in ['tools_kwargs', 'interaction_kwargs', 'id', 'index', 'golden_answers', 'data_source', 'ability', 'reward_model']:
            current_size = batch_dict[key].shape[0]
            assert current_size == num_questions
            pad_size = batch_size - num_questions
            batch_dict[key] = np.pad(batch_dict[key], ((0, pad_size),), mode='edge')

    final_batch = DataProto.from_single_dict(batch_dict)
    final_gen_batch = final_batch.pop(batch_keys=tensor_keys_to_pop, non_tensor_batch_keys=non_tensor_keys_to_pop)

    return final_batch, final_gen_batch