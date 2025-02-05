"""
Generate dataset for chess task - simulate random games and use Stockfish to determine the best move for each position.
The output is formatted as a conversation prompt similar to the countdown task.
"""

import argparse
import os
import random
from typing import List, Tuple

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm


def gen_dataset(
    num_samples: int,
    max_moves: int = 100,
    seed_value: int = 42,
    top_n: int = 10,
    cpl_threshold: int = 20,
    mate_value: int = 100000,
    stockfish_path: str = "/path/to/stockfish",
    stockfish_depth: int = 50,
    stockfish_time: int = 2,
    stockfish_threads: int = 8,
) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """
    Generate a dataset of positions along with candidate moves and their centipawn losses (CPL)
    relative to the best move.

    Each candidate is stored as a tuple (move in UCI, CPL).

    Note: num_samples refers to the total number of positions sampled.
    """
    random.seed(seed_value)
    dataset = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        engine.configure({"Threads": stockfish_threads})

        with tqdm(total=num_samples, desc="Generating dataset") as pbar:
            while len(dataset) < num_samples:
                board = chess.Board()
                ply_count = 0

                do_print = not dataset or random.randint(1, 64) == 1

                while not board.is_game_over() and ply_count < max_moves * 2:
                    info = engine.analyse(
                        board,
                        chess.engine.Limit(
                            time=stockfish_time, 
                            depth=stockfish_depth,
                        ),
                        multipv=top_n,
                    )
                    best_eval = info[0]["score"].pov(board.turn).score(mate_score=mate_value)

                    candidate_moves = []
                    for candidate in info:
                        cand_eval = (
                            candidate["score"].pov(board.turn).score(mate_score=mate_value)
                        )
                        cpl = best_eval - cand_eval
                        if cpl <= cpl_threshold:
                            candidate_moves.append(
                                {
                                    "move": candidate["pv"][0].uci(),
                                    "cpl": cpl,
                                    "eval": cand_eval,
                                    "depth": candidate["depth"],
                                }
                            )
                    
                    row = (board.fen(), candidate_moves)
                    dataset.append(row)
                    pbar.update(1)

                    if do_print:
                        print(row)
                    
                    board.push_uci(random.choice(candidate_moves)["move"])
                    ply_count += 1

                    if len(dataset) >= num_samples:
                        break

    return dataset


def make_prompt(position: str, template_type: str = "base") -> str:
    """
    Create a conversation-style prompt for the chess task.

    Args:
        position: The chess position in FEN format.
        template_type: Determines the prompt formatting ('base' or 'qwen-instruct').

    Returns:
        A formatted prompt string.
    """
    if template_type == "base":
        prompt = (
            f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
            f'User: Given the chess position "{position}", what is the best move? Show your work in <think> </think> tags. And return the best move in UCI format within <move> </move> tags.\n'
            f"Assistant: Let me think step by step.\n"
            f"<think>"
        )
    elif template_type == "qwen-instruct":
        prompt = (
            f"<|im_start|>system\nYou are a helpful chess assistant. Analyze the chess position and provide the best move along with your reasoning.\n<|im_end|>\n"
            f'<|im_start|>user\nGiven the chess position "{position}", what is the best move? Show your work in <think> </think> tags. And return the best move in UCI format within <move> </move> tags.<|im_end|>\n'
            f"<|im_start|>assistant\nLet me think step by step.\n<think>"
        )
    else:
        raise ValueError(f"Unknown template_type: {template_type}")
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="~/data/chess",
        help="Local directory to save the dataset",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--max_moves", type=int, default=100, help="Maximum number of moves per game"
    )
    parser.add_argument(
        "--stockfish_path", type=str, required=True, help="Path to the Stockfish engine"
    )
    parser.add_argument(
        "--template_type",
        type=str,
        default="base",
        choices=["base", "qwen-instruct"],
        help="Prompt formatting style",
    )
    parser.add_argument(
        "--train_size", type=int, default=9500, help="Size of the training dataset"
    )
    parser.add_argument(
        "--test_size", type=int, default=500, help="Size of the test dataset"
    )
    args = parser.parse_args()

    DATA_SOURCE = "chess"
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Generate raw chess samples (position and best move tuples).
    dataset_raw = gen_dataset(
        num_samples=args.num_samples,
        max_moves=args.max_moves,
        stockfish_path=args.stockfish_path,
    )

    assert len(dataset_raw) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = dataset_raw[:TRAIN_SIZE]
    test_dataset = dataset_raw[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]

    def make_map_fn(split):
        def process_fn(example, idx):
            position, candidate_moves = example
            prompt = make_prompt(position, template_type=args.template_type)
            data_entry = {
                "data_source": DATA_SOURCE,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "chess",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "position": position,
                        "candidate_moves": candidate_moves,
                    },
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data_entry

        return process_fn

    train_processed = [
        make_map_fn("train")(example, idx) for idx, example in enumerate(train_dataset)
    ]
    test_processed = [
        make_map_fn("test")(example, idx) for idx, example in enumerate(test_dataset)
    ]

    # Save the processed datasets to Parquet files.
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_df = pd.DataFrame(train_processed)
    train_output_path = os.path.join(local_dir, "train.parquet")
    train_df.to_parquet(train_output_path)
    print(f"Training dataset saved to: {train_output_path}")

    test_df = pd.DataFrame(test_processed)
    test_output_path = os.path.join(local_dir, "test.parquet")
    test_df.to_parquet(test_output_path)
    print(f"Test dataset saved to: {test_output_path}")
