import re, random
import torch
from verl.utils.reward_score.format_reward import extract_solution, is_retrieval_correct, normalize_answer
from verl.utils.reward_score.sequence_vlidation import is_valid_reasoner_sequence, is_valid_verifier_sequence
from verl.utils.reward_score.adv_token_reward import calc_adv_verifier_reward, calc_adv_outcome_reward


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def f1_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = normalize_answer(prediction).split()
    
    score = 0.0
    for golden_answer in golden_answers:
        gt_tokens = normalize_answer(golden_answer).split()
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)

        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1 = float(pred_tokens == gt_tokens)
        elif num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        
        score = max(score, f1)  # 取多个 golden_answer 中的最大值
    return score


def compute_rea_ver_score(rea_solution_str, ver_solution_str, ground_truth, out_reward_type='em', structure_format_score=0, final_format_score=0, retrieval_score=0, **kwargs):
    score = kwargs.get('score', 1.)
    rea_score, ver_score = {}, {}
    assistant_pattern = r"<\|im_start\|>assistant\n"
    rea_assistant_match = re.search(assistant_pattern, rea_solution_str)
    ver_assistant_match = re.search(assistant_pattern, ver_solution_str)

    if not rea_assistant_match or not ver_assistant_match:
        return False, "Missing assistant marker"
    # Extract the content after the assistant marker
    rea_start_pos = rea_assistant_match.end()
    ver_start_pos = ver_assistant_match.end()
    rea_content = rea_solution_str[rea_start_pos:]
    ver_content = ver_solution_str[ver_start_pos:]
    rea_answer = extract_solution(solution_str=rea_content, answered_by='reasoner')
    ver_answer = extract_solution(solution_str=ver_content, answered_by='verifier')
    
    rea_score['rea_retrieval'], ver_score['ver_retrieval'] = 0, 0
    is_valid_rea_format, _ = is_valid_reasoner_sequence(rea_content)
    is_valid_ver_format, _ = is_valid_verifier_sequence(ver_content)
    rea_score['rea_structure_format'] = 1 if is_valid_rea_format else 0
    ver_score['ver_structure_format'] = 1 if is_valid_ver_format else 0

    tokenizer = kwargs.get('tokenizer', None)
    adv_out_weight, adv_ver_weight = kwargs.get('adv_out_weight'), kwargs.get('adv_ver_weight')
    rea_data, ver_data = kwargs.get('rea_data'), kwargs.get('ver_data')
    rea_data['response_str'], ver_data['response_str'] = rea_content, ver_content

    # Calculate the adversarial outcome reward
    rea_score['rea_adv_outcome_reward'], rea_adv_outcome_reward = 0, 0
    ver_score['ver_adv_outcome_reward'], ver_adv_outcome_reward = 0, 0
    ver_score['adv_ver_reward'] = torch.zeros_like(ver_data['entropy'], device=ver_data['entropy'].device)
    adv_ver_reward, weighted_adv_ver_reward = torch.zeros_like(ver_data['entropy'], device=ver_data['entropy'].device), torch.zeros_like(ver_data['entropy'], device=ver_data['entropy'].device)
    ver_score['ver_clarity'] = torch.zeros_like(ver_data['entropy'], device=ver_data['entropy'].device)
    ver_score['ver_impact'] = 0

    if rea_answer is not None and ver_answer is not None:
        # In order to calculate the adversarial outcome reward, we need to check both answers of the reasoner and verifier exist.
        rea_score['rea_final_format'], ver_score['ver_final_format'] = 1, 1
        if out_reward_type == 'em':
            rea_out_score = em_check(rea_answer, ground_truth['target'])
            ver_out_score = em_check(ver_answer, ground_truth['target'])
            rea_score['rea_raw_score'] = rea_out_score
            ver_score['ver_raw_score'] = ver_out_score

        elif out_reward_type == 'f1':
            rea_out_score = f1_check(rea_answer, ground_truth['target'])
            ver_out_score = f1_check(ver_answer, ground_truth['target'])
            rea_score['rea_raw_score'] = rea_out_score
            ver_score['ver_raw_score'] = ver_out_score
            if adv_out_weight > 0:
                adv_out_reward = calc_adv_outcome_reward(rea_f1_score=rea_out_score, ver_f1_score=ver_out_score, n_bins=5)
                if adv_out_reward > 0:
                    rea_adv_outcome_reward = adv_out_reward * adv_out_weight
                    rea_score['rea_adv_outcome_reward'] = adv_out_reward
                elif adv_out_reward < 0:
                    ver_adv_outcome_reward = -adv_out_reward * adv_out_weight
                    ver_score['ver_adv_outcome_reward'] = -adv_out_reward
        else:
            raise ValueError(f"Invalid out_reward_type: {out_reward_type}")

        # Calculate the adversarial verifier reward
        if adv_ver_weight > 0:
            ver_clarity_reward, ver_impact_reward, rea_retrieval, ver_retrieval = calc_adv_verifier_reward(rea_data, ver_data, golden_answers=ground_truth['target'], tokenizer=tokenizer)
            
            clip_rea_out_score = rea_out_score
            adv_ver_reward = clip_rea_out_score * ver_clarity_reward * ver_impact_reward
            weighted_adv_ver_reward = adv_ver_weight * adv_ver_reward

            ver_score['ver_clarity'] = ver_clarity_reward
            ver_score['ver_impact'] = ver_impact_reward
            ver_score['adv_ver_reward'] = weighted_adv_ver_reward
            rea_score['rea_retrieval'] = rea_retrieval
            ver_score['ver_retrieval'] = ver_retrieval
            
        if rea_out_score:
            rea_total_score = rea_out_score + rea_adv_outcome_reward
            if is_valid_rea_format:
                rea_score['rea_total_score'] = rea_total_score
            elif not is_valid_rea_format and out_reward_type == 'em':
                rea_score['rea_total_score'] = rea_total_score - structure_format_score
            elif not is_valid_rea_format and out_reward_type == 'f1':
                rea_score['rea_total_score'] = max(rea_total_score - structure_format_score, 0)
        else:
            rea_score['rea_total_score'] = structure_format_score if is_valid_rea_format else 0

        if ver_out_score:
            ver_total_score = ver_out_score + ver_adv_outcome_reward
            if is_valid_ver_format:
                ver_score['ver_total_score'] = ver_total_score
            elif not is_valid_ver_format and out_reward_type == 'em':
                ver_score['ver_total_score'] = ver_total_score - structure_format_score
            elif not is_valid_ver_format and out_reward_type == 'f1':
                ver_score['ver_total_score'] = max(ver_total_score - structure_format_score, 0)
        else:
            ver_score['ver_total_score'] = structure_format_score if is_valid_ver_format else 0
            
    else:
        # At least one of the reasoner or verifier answer is None
        assert rea_answer is None or ver_answer is None

        if rea_answer is None:
            rea_score['rea_raw_score'] = 0
            rea_score['rea_final_format'] = 0
            rea_score['rea_total_score'] = structure_format_score if is_valid_rea_format else 0
        else:
            rea_score['rea_final_format'] = 1
            if out_reward_type == 'em':
                rea_out_score = em_check(rea_answer, ground_truth['target'])
            elif out_reward_type == 'f1':
                rea_out_score = f1_check(rea_answer, ground_truth['target'])
            rea_score['rea_raw_score'] = rea_out_score
            if is_valid_rea_format:
                rea_score['rea_total_score'] = rea_out_score
            elif not is_valid_rea_format and out_reward_type == 'em':
                rea_score['rea_total_score'] = rea_out_score - structure_format_score
            elif not is_valid_rea_format and out_reward_type == 'f1':
                rea_score['rea_total_score'] = max(rea_out_score - structure_format_score, 0)
        
        if ver_answer is None:
            ver_score['ver_raw_score'] = 0
            ver_score['ver_final_format'] = 0
            ver_score['ver_total_score'] = structure_format_score if is_valid_ver_format else 0
        else:
            ver_score['ver_final_format'] = 1
            if out_reward_type == 'em':
                ver_out_score = em_check(ver_answer, ground_truth['target'])
            elif out_reward_type == 'f1':
                ver_out_score = f1_check(ver_answer, ground_truth['target'])
            ver_score['ver_raw_score'] = ver_out_score
            if is_valid_ver_format:
                ver_score['ver_total_score'] = ver_out_score
            elif not is_valid_ver_format and out_reward_type == 'em':
                ver_score['ver_total_score'] = ver_out_score - structure_format_score
            elif not is_valid_ver_format and out_reward_type == 'f1':
                ver_score['ver_total_score'] = max(ver_out_score - structure_format_score, 0)

        
    return rea_score, ver_score

