# Quick Start: Training from Datasets

## Step 1: Ensure Dependencies

```bash
cd grillcheese/backend
pip install tqdm  # For progress bars (optional but recommended)
```

## Step 2: Run Training

```bash
python train_from_datasets.py
```

This will:
- Load conversations from `../../datasets/conversations_dataset_anonymized_cleaned.jsonl`
- Load instructions from `../../datasets/instruct_anonymized_cleaned.json`
- Train STDP learning from conversation pairs
- Store memories in the database
- Train brain emotional understanding

## Step 3: Test Your Trained Brain

```bash
python cli/cli.py --brain --learning "Hello, how are you?"
```

## Example Output

```
2024-01-XX XX:XX:XX - INFO - Initializing GrillCheese components...
2024-01-XX XX:XX:XX - INFO - Memory store initialized: memories.db
2024-01-XX XX:XX:XX - INFO - SNN compute initialized
2024-01-XX XX:XX:XX - INFO - Model initialized: models/phi-3-mini.gguf
2024-01-XX XX:XX:XX - INFO - Continuous learner initialized
2024-01-XX XX:XX:XX - INFO - Unified brain initialized

============================================================
Training from conversations dataset
============================================================
Training from conversations: 100%|████████| 880/880 [02:15<00:00, 6.51it/s]

Conversation training complete:
  Conversations processed: 880
  Memories stored: 1760
  STDP updates: 5234
  Brain experiences: 880

============================================================
Training from instructions dataset
============================================================
Training from instructions: 100%|████████| 17523/17523 [45:23<00:00, 6.45it/s]

Instruction training complete:
  Instructions processed: 17523
  Memories stored: 17523
  STDP updates: 45234
  Brain experiences: 17523

Training complete!
```

## Custom Paths

If your datasets are in a different location:

```bash
python train_from_datasets.py \
    --conversations /path/to/conversations.jsonl \
    --instructions /path/to/instructions.json
```

## Test with Limited Data

Test with just 100 items first:

```bash
python train_from_datasets.py --limit 100
```

## What Gets Saved

After training, these files are updated:
- `memories.db` - Conversation embeddings
- `learning_state/` - STDP learning state
- `brain_state/` - Brain component states

These persist across sessions!
