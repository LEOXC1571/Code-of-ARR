import re
import random
import string

def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) < 1:
        return None

    return matches[-1].group(1).strip()

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


def compute_em(text, ground_truth, tokenizer=None):
    score = 1.
    extract_answer = text.strip()
    if extract_answer is None:
        em = em_check(prediction=text, golden_answers=ground_truth['target'])
        if em:
            return score - 0.1
        else:
            return 0
    else:
        em = em_check(prediction=extract_answer, golden_answers=ground_truth['target'])
        if em:
            return score
        else:
            return 0
        

def compute_f1(text, ground_truth, tokenizer=None):
    if tokenizer is None:
        extract_answer = text.strip()
    else:
        extract_answer = extract_solution(solution_str=text)
        
    if extract_answer is None:
        f1 = f1_check(prediction=text, golden_answers=ground_truth['target'])
        return max(f1 - 0.1, 0)
    else:
        f1 = f1_check(prediction=extract_answer, golden_answers=ground_truth['target'])
        return f1


