"""
Custom Chess Tokenizer for the Chess Challenge.

This tokenizer treats each move as a single token using the extended UCI notation
from the Lichess dataset (e.g., WPe2e4, BNg8f6).

The dataset format uses:
- W/B prefix for White/Black
- Piece letter: P=Pawn, N=Knight, B=Bishop, R=Rook, Q=Queen, K=King
- Source and destination squares (e.g., e2e4)
- Special suffixes: (x)=capture, (+)=check, (+*)=checkmate, (o)/(O)=castling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


class ChessTokenizer(PreTrainedTokenizer):
    """
    A custom tokenizer for chess moves using extended UCI notation.
    
    This tokenizer maps each possible chess move to a unique token ID.
    The vocabulary is built from the training dataset to ensure all moves
    encountered during training have a corresponding token.
    
    Example:
        >>> tokenizer = ChessTokenizer()
        >>> tokenizer.encode("WPe2e4 BPe7e5")
        [1, 42, 87, 2]  # [BOS, e2e4, e7e5, EOS]
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """
        Initialize the chess tokenizer.
        
        Args:
            vocab_file: Path to a JSON file containing the vocabulary mapping.
            vocab: Dictionary mapping tokens to IDs (alternative to vocab_file).
            **kwargs: Additional arguments passed to PreTrainedTokenizer.
        """
        # Initialize special tokens
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN
        
        # Load or create vocabulary
        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            # Create a minimal vocabulary with just special tokens
            # The full vocabulary should be built from the dataset
            self._vocab = self._create_default_vocab()
        
        # Create reverse mapping
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        # Call parent init AFTER setting up vocab
        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )
    
    def _create_default_vocab(self) -> Dict[str, int]:
        """
        Create a default vocabulary with special tokens and common moves.
        
        For the full vocabulary, use `build_vocab_from_dataset()` or
        `build_vocab_from_iterator()`.
        """
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        
        # Generate all possible moves in the dataset format
        all_moves = self._generate_all_possible_moves()
        
        vocab = {}
        for idx, token in enumerate(special_tokens + all_moves):
            vocab[token] = idx
        
        return vocab
    
    def _generate_all_possible_moves(self) -> List[str]:
        """
        Generate all possible chess moves in the extended UCI format.
        
        Format: [W|B][P|N|B|R|Q|K][source][dest][suffix]
        Suffixes: (x), (+), (+*), (o), (O), (x+), (x+*)
        
        Returns:
            List of all possible move strings.
        """
        moves = []
        
        files = "abcdefgh"
        ranks = "12345678"
        squares = [f + r for f in files for r in ranks]
        
        colors = ["W", "B"]
        pieces = ["P", "N", "B", "R", "Q", "K"]
        
        # Regular moves: piece from square to square
        suffixes = ["", "(x)", "(+)", "(x+)", "(+*)", "(x+*)"]
        
        for color in colors:
            for piece in pieces:
                for src in squares:
                    for dst in squares:
                        if src != dst:
                            base_move = f"{color}{piece}{src}{dst}"
                            for suffix in suffixes:
                                moves.append(base_move + suffix)
        
        # Pawn promotions: add promotion piece
        promotion_pieces = ["Q", "R", "B", "N"]
        for color in colors:
            # White promotions (rank 7 to 8)
            if color == "W":
                src_rank, dst_rank = "7", "8"
            else:  # Black promotions (rank 2 to 1)
                src_rank, dst_rank = "2", "1"
            
            for src_file in files:
                for dst_file in files:
                    # Only allow same file or adjacent file (for captures)
                    if abs(ord(src_file) - ord(dst_file)) <= 1:
                        src = src_file + src_rank
                        dst = dst_file + dst_rank
                        for promo in promotion_pieces:
                            for suffix in suffixes:
                                moves.append(f"{color}P{src}{dst}={promo}{suffix}")
        
        # Castling
        for color in colors:
            # Kingside castling
            moves.append(f"{color}Ke1g1(o)" if color == "W" else f"{color}Ke8g8(o)")
            # Queenside castling
            moves.append(f"{color}Ke1c1(O)" if color == "W" else f"{color}Ke8c8(O)")
        
        return moves
    
    @classmethod
    def build_vocab_from_iterator(
        cls,
        iterator,
        min_frequency: int = 1,
    ) -> "ChessTokenizer":
        """
        Build a tokenizer vocabulary from an iterator of game strings.
        
        Args:
            iterator: An iterator yielding game strings (space-separated moves).
            min_frequency: Minimum frequency for a token to be included.
        
        Returns:
            A ChessTokenizer with the built vocabulary.
        """
        from collections import Counter
        
        token_counts = Counter()
        
        for game in iterator:
            moves = game.strip().split()
            token_counts.update(moves)
        
        # Filter by frequency
        tokens = [
            token for token, count in token_counts.items()
            if count >= min_frequency
        ]
        
        # Sort for reproducibility
        tokens = sorted(tokens)
        
        # Build vocabulary
        special_tokens = [cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN]
        vocab = {token: idx for idx, token in enumerate(special_tokens + tokens)}
        
        return cls(vocab=vocab)
    
    @classmethod
    def build_vocab_from_dataset(
        cls,
        dataset_name: str = "dlouapre/lichess_2025-01_1M",
        split: str = "train",
        column: str = "text",
        min_frequency: int = 1,
        max_samples: Optional[int] = None,
    ) -> "ChessTokenizer":
        """
        Build a tokenizer vocabulary from a Hugging Face dataset.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub.
            split: Dataset split to use.
            column: Column containing the game strings.
            min_frequency: Minimum frequency for a token to be included.
            max_samples: Maximum number of samples to process.
        
        Returns:
            A ChessTokenizer with the built vocabulary.
        """
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        def game_iterator():
            for example in dataset:
                yield example[column]
        
        return cls.build_vocab_from_iterator(game_iterator(), min_frequency=min_frequency)
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        return dict(self._vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string of moves into a list of tokens.
        
        Args:
            text: A string of space-separated moves.
        
        Returns:
            List of move tokens.
        """
        return text.strip().split()
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID."""
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token."""
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of tokens back to a string."""
        # Filter out special tokens for cleaner output
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        return " ".join(t for t in tokens if t not in special)
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        """
        Save the vocabulary to a JSON file.
        
        Args:
            save_directory: Directory to save the vocabulary.
            filename_prefix: Optional prefix for the filename.
        
        Returns:
            Tuple containing the path to the saved vocabulary file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)


def count_vocab_from_dataset(
    dataset_name: str = "dlouapre/lichess_2025-01_1M",
    split: str = "train",
    column: str = "text",
    max_samples: Optional[int] = 10000,
) -> Dict[str, int]:
    """
    Count token frequencies in a dataset (useful for vocabulary analysis).
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        split: Dataset split to use.
        column: Column containing the game strings.
        max_samples: Maximum number of samples to process.
    
    Returns:
        Dictionary mapping tokens to their frequencies.
    """
    from collections import Counter
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    token_counts = Counter()
    
    for example in dataset:
        moves = example[column].strip().split()
        token_counts.update(moves)
    
    return dict(token_counts)
