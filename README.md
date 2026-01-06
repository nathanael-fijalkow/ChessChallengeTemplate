# Chess Challenge

Train a 1M parameter LLM to play chess!

## Objective

Design and train a transformer-based language model to predict chess moves. Your model must:

1. **Stay under 1M parameters** - This is the hard constraint!
2. **Use a custom tokenizer** - Design an efficient move-level tokenizer
3. **Play legal chess** - The model should learn to generate valid moves
4. **Beat Stockfish** - Your ELO will be measured against Stockfish Level 1

## Dataset

We use the Lichess dataset: [`dlouapre/lichess_2025-01_1M`](https://huggingface.co/datasets/dlouapre/lichess_2025-01_1M)

The dataset uses an extended UCI notation:
- `W`/`B` prefix for White/Black
- Piece letter: `P`=Pawn, `N`=Knight, `B`=Bishop, `R`=Rook, `Q`=Queen, `K`=King
- Source and destination squares (e.g., `e2e4`)
- Special suffixes: `(x)`=capture, `(+)`=check, `(+*)`=checkmate, `(o)`/`(O)`=castling

Example game:
```
WPe2e4 BPe7e5 WNg1f3 BNb8c6 WBf1b5 BPa7a6 WBb5c6(x) BPd7c6(x) ...
```

## Quick Start

### Installation

```bash
# Clone the template
git clone https://github.com/nathanael-fijalkow/ChessChallengeTemplate.git
cd ChessChallengeTemplate

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### Train a Model

```bash
# Basic training
python -m src.train \
    --output_dir ./my_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32

# Push to Hugging Face Hub
python -m src.train \
    --output_dir ./my_model \
    --push_to_hub \
    --hub_model_id my-chess-model \
    --hub_organization your-org
```

### Evaluate Your Model

Evaluation happens in two phases:

```bash
# Phase 1: Legal Move Evaluation (quick sanity check)
python -m src.evaluate \
    --model_path ./my_model/final_model \
    --mode legal \
    --n_positions 500

# Phase 2: Win Rate Evaluation (full games against Stockfish)
python -m src.evaluate \
    --model_path ./my_model/final_model \
    --mode winrate \
    --n_games 100 \
    --stockfish_level 1

# Or run both phases:
python -m src.evaluate \
    --model_path ./my_model/final_model \
    --mode both
```

## Parameter Budget

Here's a typical budget:

| Component | Calculation | Params |
|-----------|-------------|--------|
| Token Embeddings | V × d | ~154,000 |
| Position Embeddings | n_ctx × d | ~33,000 |
| Transformer Layers | L × ~120k | ~720,000 |
| LM Head | (tied) | 0 |
| **Total** | | **~907,000** |

Use the utility function to check your budget:

```python
from src import ChessConfig, print_parameter_budget

config = ChessConfig(
    vocab_size=1200,
    n_embd=128,
    n_layer=6,
    n_head=4,
)
print_parameter_budget(config)
```

### Pro Tips

1. **Weight Tying**: The default config ties the embedding and output layer weights, saving ~154k parameters
2. **Vocabulary Size**: Keep it small! ~1200 tokens covers all moves
3. **Depth vs Width**: With limited parameters, experiment with shallow-but-wide vs deep-but-narrow

## Customization

### Custom Tokenizer

The template provides a move-level tokenizer. You can customize it:

```python
from src import ChessTokenizer

# Build from dataset
tokenizer = ChessTokenizer.build_vocab_from_dataset(
    dataset_name="dlouapre/lichess_2025-01_1M",
    min_frequency=10,  # Only keep frequent moves
)

# Or create a custom one by inheriting from PreTrainedTokenizer
```

### Custom Architecture

Modify the model in `src/model.py`:

```python
from src import ChessConfig, ChessForCausalLM

# Customize configuration
config = ChessConfig(
    vocab_size=1200,
    n_embd=128,      # Try 96, 128, or 192
    n_layer=6,       # Try 4, 6, or 8
    n_head=4,        # Try 4 or 8
    n_inner=512,     # Feed-forward dimension
    dropout=0.1,
    tie_weights=True,
)

model = ChessForCausalLM(config)
```

### Training Strategies

The template supports several training approaches:

1. **Standard Pre-training**: Next-token prediction on game sequences
2. **Fine-tuning on GM Games**: Filter for high-rated games only
3. **RL Fine-tuning**: Use Stockfish evaluations as rewards

## Evaluation Metrics

### Phase 1: Legal Move Evaluation

Tests if your model generates valid chess moves:

| Metric | Description |
|--------|-------------|
| **Legal Rate (1st try)** | % of legal moves on first attempt |
| **Legal Rate (with retry)** | % of legal moves within 3 attempts |

> **Target**: >90% legal rate before proceeding to Phase 2

### Phase 2: Win Rate Evaluation

Full games against Stockfish to measure playing strength:

| Metric | Description |
|--------|-------------|
| **Win Rate** | % of games won against Stockfish |
| **ELO Rating** | Estimated rating based on game results |
| **Avg Game Length** | Average number of moves per game |
| **Illegal Move Rate** | % of illegal moves during games |


## Submission

1. Train your model
2. Push to Hugging Face Hub under the class organization
3. Your model will be automatically evaluated

```python
# Push your model
model.push_to_hub("your-model-name", organization="LLM-course")
tokenizer.push_to_hub("your-model-name", organization="LLM-course")
```
