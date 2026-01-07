import re
import torch
from verl.utils.reward_score.sequence_vlidation import check_for_balanced_tags

def match_tag_with_postions(text, role="reasoner", valid_seq_length=0, offset_mapping=None, ignore_until=0):
    if role == "reasoner":
        tags_pattern = r"<(think|search|verify|answer)>(.*?)</\1>"
        # tags = ["think", "search", "verify", "answer"]
    elif role == "verifier":
        tags_pattern = r"<(information|verify|response|final_answer)>(.*?)</\1>"

        # tags_pattern = r"(</?(?:verify|response|final_answer)>)"
        # tags = ["verify", "response", "final_answer"]
    
    char_to_token = {}
    total_tokens = len(offset_mapping)
    response_start_token = total_tokens - valid_seq_length
    for token_idx in range(total_tokens):
        start_char, end_char = offset_mapping[token_idx]
        if end_char <= ignore_until:
            continue
        assert start_char < end_char
        for i in range(start_char, end_char):
            char_to_token[i] = token_idx
    
    matches = re.finditer(tags_pattern, text, re.DOTALL)
    sample_tags = []
    tag_count = {}
    if matches is None:
        return sample_tags, tag_count
    for match in matches:
        tag_name = match.group(1)
        content = match.group(2)
        start_idx = match.start(0)
        end_idx = match.end(0)
        if end_idx <= ignore_until:
            continue
        content_start_idx = match.start(2)
        content_end_idx = match.end(2)
        token_indices = list(range(start_idx, end_idx))
        content_str_indices = list(range(content_start_idx, content_end_idx))
        content_token_indices = []
        
        for c in content_str_indices:
            if char_to_token[c] not in content_token_indices:
                content_token_indices.append(char_to_token[c])
        if tag_name not in tag_count.keys():
            tag_count[tag_name] = 0
        tag_count[tag_name] += 1
        sample_tags.append({
            'tag_name': tag_name,
            'content': content,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'content_start_idx': content_start_idx,
            'content_end_idx': content_end_idx,
            'token_indices': token_indices,
            'content_token_indices': content_token_indices,
            'tag_count': tag_count[tag_name],
        })
    
    return sample_tags, tag_count


def create_offset_mapping(tokenizer, response, valid_response_length):
    tokens = tokenizer.convert_ids_to_tokens(response[:valid_response_length])
    start = 0
    offset_mapping = []
    for idx, tokens in enumerate(tokens):
        offset_mapping.append((start, start + len(tokens)))
        start += len(tokens)
    assert len(offset_mapping) == valid_response_length
    return offset_mapping
    

def get_token_level_weight(tag_count, weight_type='equal'):
    tag_weights = {}
    if weight_type == 'equal':
        for tag_name in tag_count.keys():
            tag_weights[tag_name] = [1 for _ in range(tag_count[tag_name])]
    elif weight_type == 'linear_desc':
        epsilon = 0.2
        for tag_name in tag_count.keys():
            if tag_count[tag_name] == 1:
                tag_weights[tag_name] = [1.0]
                continue
            upper_bound, lower_bound = 1 - epsilon, epsilon
            tag_weights[tag_name] = []
            for i in range(tag_count[tag_name]):
                tag_weights[tag_name].append(upper_bound - i * (upper_bound - lower_bound) / (tag_count[tag_name] - 1))
    elif weight_type == 'linear_asc':
        epsilon = 0.2
        for tag_name in tag_count.keys():
            if tag_count[tag_name] == 1:
                tag_weights[tag_name] = [1.0]
                continue
            upper_bound, lower_bound = 1 - epsilon, epsilon
            tag_weights[tag_name] = []
            for i in range(tag_count[tag_name]):
                tag_weights[tag_name].append(lower_bound + i * (upper_bound - lower_bound) / (tag_count[tag_name] - 1))
    elif weight_type == 'positional_asc':
        for tag_name in tag_count.keys():
            if tag_count[tag_name] == 1:
                tag_weights[tag_name] = [1.0]
                continue
            tag_weights[tag_name] = []
            for i in range(tag_count[tag_name]):
                tag_weights[tag_name].append((i+1)/tag_count[tag_name])
    else:
        raise NotImplementedError
    return tag_weights


def subseq_pos_weight(token_indices, device):
    len_tagged_sequence = len(token_indices)
    # mask_adv_weight = torch.zeros(len_tagged_sequence, dtype=torch.float, device=device)
    pos_idx = torch.arange(1, len_tagged_sequence + 1, dtype=torch.float, device=device)
    relative_pos_idx = pos_idx / len_tagged_sequence
    mask_adv_weight = relative_pos_idx.pow(0.5)
    return mask_adv_weight



def create_weighted_mask(entropy, tag_content_results, tag_weights, tags_to_adv):
    weighted_mask = torch.zeros_like(entropy, dtype=torch.float, device=entropy.device)
    for i in range(len(tag_content_results)):
        tag_name = tag_content_results[i]['tag_name']
        token_indices = tag_content_results[i]['content_token_indices']
        if len(token_indices) == 0:
            continue
        # print("Tag name: ", tag_name
        if tag_name in tags_to_adv:
             # list
            sub_seq_adv_weight = subseq_pos_weight(token_indices, device=entropy.device)
            seq_adv_weight = tag_weights[tag_name][tag_content_results[i]['tag_count'] - 1] # int
            invalid_token_indices = [idx for idx in token_indices if idx >= len(entropy)]
            mask_adv_weight = sub_seq_adv_weight * seq_adv_weight
            weighted_mask[token_indices] = torch.tensor(mask_adv_weight, device=entropy.device)    
    return weighted_mask
