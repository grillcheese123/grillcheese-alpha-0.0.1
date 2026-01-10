# Training GrillCheese Brain from Datasets

This guide explains how to use the created datasets to train your GrillCheese brain system.

## Quick Start

```bash
cd grillcheese/backend
python train_from_datasets.py
```

This will:
1. Load conversations and instructions datasets
2. Train the ContinuousLearner (STDP learning)
3. Store memories from conversations
4. Train emotional understanding (if enabled)

## Usage

### Basic Training

Train from both datasets with default settings:

```bash
python train_from_datasets.py
```

### Train Only Conversations

```bash
python train_from_datasets.py \
    --conversations ../../datasets/conversations_dataset_anonymized_cleaned.jsonl \
    --instructions ""  # Skip instructions
```

### Train Only Instructions

```bash
python train_from_datasets.py \
    --conversations ""  # Skip conversations
    --instructions ../../datasets/instruct_anonymized_cleaned.json
```

### Limit Training Data (for testing)

```bash
python train_from_datasets.py --limit 100
```

### Train Emotional Understanding

```bash
python train_from_datasets.py --train-emotions --emotion-epochs 5
```

### Skip Memory Storage (faster, less memory)

```bash
python train_from_datasets.py --no-memories
```

### Custom Database Path

```bash
python train_from_datasets.py --db custom_memory.db
```

## What Gets Trained

### 1. ContinuousLearner (STDP Learning)
- Learns temporal associations between user inputs and responses
- Builds spike-timing dependent plasticity patterns
- Processes through SNN (Spiking Neural Network)

### 2. Memory Store
- Stores conversation embeddings for retrieval
- Builds semantic memory of interactions
- Enables context-aware responses

### 3. UnifiedBrain (Emotional Understanding)
- Trains Amygdala for emotional processing
- Learns emotional patterns from conversations
- Calibrates affect prediction (valence/arousal)

## Training Process

1. **Load Datasets**: Reads JSONL/JSON files
2. **Process Conversations**: Extracts user-assistant pairs
3. **Generate Embeddings**: Uses model to create embeddings
4. **STDP Learning**: Updates neural connections
5. **Memory Storage**: Stores important patterns
6. **Brain Training**: Trains emotional understanding

## Expected Output

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
Training from conversations: 100%|████████| 880/880 [XX:XX<00:00, X.XXit/s]

Conversation training complete:
  Conversations processed: 880
  Memories stored: 1760
  STDP updates: 5234
  Brain experiences: 880

============================================================
Training from instructions dataset
============================================================
Training from instructions: 100%|████████| 17523/17523 [XX:XX<00:00, X.XXit/s]

Instruction training complete:
  Instructions processed: 17523
  Memories stored: 17523
  STDP updates: 45234
  Brain experiences: 17523

Training complete!
```

## Integration with CLI

After training, use the CLI with your trained brain:

```bash
python cli/cli.py --learning --brain "Your prompt here"
```

The `--learning` flag enables continuous learning, and `--brain` enables emotional intelligence.

## Training Statistics

After training, check statistics:

```bash
python cli/cli.py --stats
```

This shows:
- Memory statistics
- Learning statistics (STDP updates, conversations learned)
- Brain statistics (experiences learned, emotional calibration)

## Advanced Usage

### Custom Dataset Paths

```bash
python train_from_datasets.py \
    --conversations /path/to/conversations.jsonl \
    --instructions /path/to/instructions.json
```

### Batch Processing

For very large datasets, process in batches:

```bash
# First batch
python train_from_datasets.py --limit 5000

# Second batch (continues from where it left off)
python train_from_datasets.py --limit 10000
```

### Emotion Training Only

```bash
python train_from_datasets.py \
    --conversations ../../datasets/conversations_dataset_anonymized_cleaned.jsonl \
    --train-emotions \
    --emotion-epochs 10 \
    --limit 1000
```

## Troubleshooting

### Model Not Found
If you see "No model found", ensure you have a GGUF model:
```bash
python download_model.py
```

### Out of Memory
Use `--no-memories` to skip memory storage:
```bash
python train_from_datasets.py --no-memories
```

### Slow Training
- Use `--limit` to test with smaller datasets first
- Use `--no-memories` to reduce memory usage
- Ensure GPU acceleration is enabled (check brain initialization)

## What Happens After Training

1. **Memory Database**: Updated with conversation embeddings
2. **Learning State**: Saved to `learning_state/` directory
3. **Brain State**: Saved to `brain_state/` directory
4. **STDP Weights**: Updated neural connection weights

These persist across sessions, so training accumulates over time.

## Next Steps

After training:
1. Test the brain with: `python cli/cli.py --brain "test prompt"`
2. Check statistics: `python cli/cli.py --stats`
3. Continue learning: Use `--learning` flag during conversations
4. Fine-tune: Adjust learning rates in `LearningConfig` if needed
