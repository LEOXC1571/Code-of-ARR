import re


def split_tag_positions(content, role='reasoner', return_position=False):
    if role == "reasoner":
        split_pattern = r"(</?(?:think|search|verify|response|answer)>)"
    elif role == "verifier":
        split_pattern = r"(</?(?:verify|search|information|response|answer|final_answer)>)"
    if not return_position:
        parts = re.split(split_pattern, content)
        return parts
    else:
        parts = re.finditer(split_pattern, content)
        return parts
    
def check_for_balanced_tags(text, tags_to_check):
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", text))
        closing_count = len(re.findall(f"</{tag}>", text))
        if opening_count != closing_count:
            return False
    return True


def is_valid_reasoner_sequence(content, parts=None):
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "verify", "answer"]
    tag_balanced = check_for_balanced_tags(content, tags_to_check)
    if not tag_balanced:
        return False, "Unbalanced tags"
    # for tag in tags_to_check:
    #     opening_count = len(re.findall(f"<{tag}>", content))
    #     closing_count = len(re.findall(f"</{tag}>", content))
    #     if opening_count != closing_count:
    #         return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize

    split_pattern = r"(</?(?:think|search|information|verify|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|verify|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "after_verify"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<verify>" and state == "information":
                state = "in_verify"
            elif part == "</verify>" and state == "in_verify":
                state = "after_verify"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_verify", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "after_verify", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def is_valid_verifier_sequence(content):
    # Check for balanced tags
    tags_to_check = ["information", "summary", "verify", "selected_doc", "answer", "final_answer"]
    tag_balanced = check_for_balanced_tags(content, tags_to_check)
    if not tag_balanced:
        return False, "Unbalanced tags"
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:information|response|selected_doc|verify|answer|final_answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
        
        # Check if this is a tag
        if re.match(r"</?(?:information|response|selected_doc|verify|answer|final_answer)>", part):
            # This is a tag, check if it's valid in the current state
            
            if part == "<information>" and state in ["start", "after_response"]:
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<verify>" and state in ["information", "after_answer"]:
                state = "in_verify"
            elif part == "</verify>" and state == "in_verify":
                state = "after_verify"
            elif part == "<selected_doc>" and state == "after_verify":
                state = "in_selected_doc"
            elif part == "</selected_doc>" and state == "in_selected_doc":
                state = "selected_doc"
            elif part == "<response>" and state in ["selected_doc", "after_verify"]:
                state = "in_response"
            elif part == "</response>" and state == "in_response":
                state = "after_response"
            elif part == "<answer>" and state in ["start", "after_response"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "after_answer"
            elif part == "<final_answer>" and state == "after_verify":
                state = "in_final_answer"
            elif part == "</final_answer>" and state == "in_final_answer": 
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_information", "in_response", "in_verify", "in_selected_doc", "in_answer", "in_final_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "information", "after_response", "selected_doc", "after_verify", "after_answer"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"
