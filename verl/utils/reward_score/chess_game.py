import chess
import re
import random


def extract_solution(solution_str):
    """Extract the move from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    move_pattern = r'<move>(.*?)</move>'
    match = re.finditer(move_pattern, solution_str)
    matches = list(match)
    if matches:
        move = matches[-1].group(1).strip()
    else:
        move = None
    return move


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
    
    # Check if position is valid (this should always be the case)
    try:
        board = chess.Board(position)
    except ValueError:
        if do_print:
            print(f"Invalid board position: {position}")
        return 0

    # Check if move is valid UCI
    try:
        move_obj = chess.Move.from_uci(move)
    except chess.InvalidMoveError:
        if do_print:
            print(f"Move {move} is invalid (bad UCI format).")
        return 0
    
    # Check if move is legal in the current position
    if not board.is_legal(move_obj):
        if do_print:
            print(f"Move {move} is not legal in the given position.")
        return 0
    
    # Check if move is in candidate moves
    for rank, candidate in enumerate(candidate_moves, start=1):
        if move == candidate['move']:
            cpl = candidate['cpl']
            # Normalize CPL to a score between the midpoint of format_score and max_score, and max_score
            midpoint = (format_score + max_score) / 2
            score = max(midpoint, max_score - (cpl / 20) * (max_score - midpoint))
            
            if do_print:
                print(f"Extracted move {move} is the rank {rank} candidate move with CPL: {cpl}, Eval: {candidate['eval']}, Score: {score}")

            return score
    
    if do_print:
        print(f"Move {move} not in candidate moves")

    return format_score
