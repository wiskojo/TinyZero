import re
import random


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, format_score=0.1, max_score=1.):
    """The scoring function for chess task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing position and candidate moves
        format_score: the score for correct format but wrong answer
        max_score: the maximum score for the best answer
    """
    position = ground_truth['position']
    candidate_moves = ground_truth['candidate_moves']

    move = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Board position: {position}")
        print(f"Candidate moves: {candidate_moves}")
        print(f"Extracted move: {move}")
        print(f"Solution string: {solution_str}")

    if move is None:
        if do_print:
            print(f"No move found")
        return 0
    
    # Check if move is in candidate moves
    for rank, candidate in enumerate(candidate_moves, start=1):
        if move == candidate['move']:
            cpl = candidate['cpl']
            # Normalize CPL to a score between 0 and 1, anything larger than 20 cpl scores 0
            score_value = max(0, 1 - (cpl / 20))
            score = score_value * max_score
            
            if do_print:
                print(f"Extracted move is the rank {rank} candidate move with CPL: {cpl}, Eval: {candidate['eval']}, Score: {score}")

            return score
    
    if do_print:
        print(f"Move not in candidate moves")

    return format_score
