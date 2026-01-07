import re
import string
import torch
import math
from verl.utils.reward_score.sequence_pattern import create_offset_mapping, match_tag_with_postions

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution(solution_str, answered_by='reasoner'):
    """Extract the equation from the solution string."""
    if answered_by == 'reasoner':
        answer_pattern = r"<answer>(.*?)</answer>"
    elif answered_by == 'verifier':
        answer_pattern = r"<final_answer>(.*?)</final_answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def extract_verify_blocks(text: str) -> list[str]:
    pattern = r"<verify>(.*?)</verify>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def extract_response_blocks(text: str) -> list[str]:
    pattern = r"<response>(.*?)</response>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def extract_think_blocks(text: str) -> list[str]:
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def classify_entropy_patterns(tag_entropy_dict, threshold=0.1):
    pattern_to_score = {
        'monotonic_decreasing': 5,
        'increase_then_decrease': 4,
        'flat': 2.0,
        'decrease_then_increase': 1.5,
        'monotonic_increasing': 1.0
    }
    def is_increasing(x, y):
        return x < y - threshold
    def is_decreasing(x, y):
        return x > y + threshold
    def is_equal(x, y):
        return abs(x - y) <= threshold
    def get_pattern(lst):
        n = len(lst)
        if n == 1:
            return "flat"
        elif n == 2:
            a, b = lst
            if is_increasing(a, b):
                return "monotonic_increasing"
            elif is_decreasing(a, b):
                return "monotonic_decreasing"
            else:
                return "flat"
        elif n >= 3:  # n == 3
            
            a, b, c = lst[-3], lst[-2], lst[-1]

            # Check pairwise relations with threshold
            ab_inc = is_increasing(a, b)
            ab_dec = is_decreasing(a, b)
            ab_eq  = is_equal(a, b)

            bc_inc = is_increasing(b, c)
            bc_dec = is_decreasing(b, c)
            bc_eq  = is_equal(b, c)

            # Full flat
            if ab_eq and bc_eq:
                return "flat"
            # Monotonic increasing (allow plateaus)
            elif not (ab_dec and bc_dec):
                return "monotonic_increasing"
            # Monotonic decreasing (allow plateaus)
            elif not (ab_inc and bc_inc):
                return "monotonic_decreasing"
            # Peak: increase then decrease
            elif ab_inc and bc_dec:
                return "increase_then_decrease"
            # Valley: decrease then increase
            elif ab_dec and bc_inc:
                return "decrease_then_increase"
            # If none of the above, treat as flat or irregular
            # In practice, with threshold, most small changes become flat
            else:
                return "flat"  # or "irregular" if you prefer
        else:
            return 'flat'

    return {tag: pattern_to_score[get_pattern(entropy_list)] for tag, entropy_list in tag_entropy_dict.items()}


def get_entropy_pattern(response_str, data, tokenizer, role='reasoner'):
    entropy = data.get('entropy', None)
    response_ids = data.get('response_ids', None)
    info_mask = data.get('info_mask', None)
    response_length = data.get('response_length', None)

    tag_to_extract = {'reasoner': ['think'], 'verifier': ['response', 'verify']}
    tag_entropy = {}
    for tag in tag_to_extract[role]:
        tag_entropy[tag] = []

    offset_mapping = create_offset_mapping(tokenizer, response_ids, response_length)

    norm_entropy = entropy
    
    tag_content, tag_count = match_tag_with_postions(response_str, role=role, valid_seq_length=response_length, offset_mapping=offset_mapping)

    for subseq in tag_content:
        tag_name = subseq['tag_name']
        if tag_name in tag_to_extract[role]:
            subseq_entropy = norm_entropy[subseq['content_token_indices']]
            subseq_entropy = torch.mean(subseq_entropy)
            tag_entropy[tag_name].append(subseq_entropy)
    
    tag_entropy_pattern = classify_entropy_patterns(tag_entropy)
    return tag_entropy_pattern


def calc_ver_clarity(entropy, tag_content, subseq_retrieval, tag_to_compute=['response', 'verify']):
    token_level_clarity = torch.zeros_like(entropy, dtype=torch.float, device=entropy.device)
    
    tag_entropy = {key: [] for key in tag_to_compute}
    for subseq in tag_content:
        tag_name = subseq['tag_name']
        if tag_name in tag_to_compute:
            token_indices = subseq['content_token_indices']
            subseq_entropy = entropy[token_indices]
            length_subseq = len(token_indices)
            if length_subseq != 0:
                mean_subseq_entropy = math.exp(-torch.sum(subseq_entropy).item() / length_subseq)
                # mean_subseq_entropy = torch.sum(subseq_entropy).item() / length_subseq
            else:
                mean_subseq_entropy = 0
            tag_entropy[tag_name].append(mean_subseq_entropy)

            aid, ais = subseq_retrieval[tag_name].pop(0)
            if ais:
                # both in doc and response
                clarity_multiplier = 1.0
            elif aid and not ais:
                # in doc but not in response
                clarity_multiplier = -1.0
            else:
                # not in response
                clarity_multiplier = 0.0
            subseq_tensor = torch.tensor([mean_subseq_entropy * clarity_multiplier] * length_subseq, dtype=torch.float, device=entropy.device)
            token_level_clarity[token_indices] = subseq_tensor
    return tag_entropy, token_level_clarity


def subseq_retrieval_correct(tag_content: list[dict], golden_answers: list[str], tags=['verify']) -> list[str]:
    ans_in_doc = {'information': []}
    ans_in_resp = {key: [] for key in tags}
    last_doc_id = 0
    for i in range(len(tag_content)):
        tc = tag_content[i]
        tag_name = tc['tag_name']
        content = tc['content']
        if tag_name == 'information':
            last_doc_id = tc['tag_count']
            if len(ans_in_doc[tag_name]) > 0 and  ans_in_doc[tag_name][-1] == True:
                aid = True
            else:
                aid = False
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(content):
                        aid = True
                        break
            ans_in_doc[tag_name].append(aid)
        elif tag_name not in tags:
            continue
        else:
            ais = [False, False]
            if last_doc_id > 0:
                ais[0] = ans_in_doc['information'][last_doc_id - 1]
            if len(ans_in_resp[tag_name]) > 0 and ans_in_resp[tag_name][-1][1] == True:
                ais[1] = True
            else:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(content):
                        ais[1] = True
                        break
            ans_in_resp[tag_name].append(ais)
    
    return ans_in_resp


def calc_adv_verifier_reward(rea_data, ver_data, golden_answers, tokenizer):
    rea_response_str = rea_data.get('response_str', '')
    ver_response_str = ver_data.get('response_str', '')
    ver_entropy = ver_data.get('entropy', None)
    
    ver_impact_reward = 0.6
    ## calc verifier calrity reward
    ans_in_verifier_seq = int(is_retrieval_correct(rea_response_str, golden_answers))
    ans_in_search_results = int(is_retrieval_correct(ver_response_str, golden_answers))

    if not ans_in_search_results:
        token_level_clarity = torch.zeros_like(ver_entropy, dtype=torch.float, device=ver_entropy.device)
        return token_level_clarity, ver_impact_reward, ans_in_verifier_seq, ans_in_search_results
    else:
        ver_offset_mapping = create_offset_mapping(tokenizer, ver_data.get('response_ids'), ver_data.get('response_length'))
        tag_content, tag_count = match_tag_with_postions(ver_response_str, role='verifier', valid_seq_length=ver_data.get('response_length'), offset_mapping=ver_offset_mapping)
        tag_to_focus = ['response', 'verify']

        # calculate subseq_correct
        subseq_retrival = subseq_retrieval_correct(tag_content, golden_answers, tags=tag_to_focus)
        for t in tag_to_focus:
            assert len(subseq_retrival[t]) == tag_count.get(t, 0)

        ver_tag2mean_entropy, token_level_clarity = calc_ver_clarity(ver_entropy, tag_content, subseq_retrival, tag_to_focus)

        # calc verifier impact reward
        tag_entropy_pattern = get_entropy_pattern(rea_response_str, rea_data, tokenizer, role='reasoner')
        num_tag, pattern = 0, 0
        for key in tag_entropy_pattern.keys():
            num_tag += 1
            pattern += tag_entropy_pattern[key] / 5
        ver_impact_reward = pattern / num_tag
        # reasoner behavior that need to be promoted
        return token_level_clarity, ver_impact_reward, ans_in_verifier_seq, ans_in_search_results


def calc_adv_outcome_reward(rea_f1_score, ver_f1_score, n_bins=5):
    rea_bin_idx = min(int(rea_f1_score * n_bins), n_bins-1)
    ver_bin_idx = min(int(ver_f1_score * n_bins), n_bins-1)

    out_reward = rea_bin_idx - ver_bin_idx
    return out_reward
