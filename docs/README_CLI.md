# GrillCheese CLI Usage

Command-line interface for interacting with GrillCheese AI.

## Basic Usage

### Single Prompt
```bash
python cli.py "What is machine learning?"
```

### Interactive Mode
```bash
python cli.py --interactive
# or
python cli.py -i
```

### From stdin
```bash
echo "Tell me about AI" | python cli.py
```

## Commands

### Show Memory Statistics
```bash
python cli.py --stats
```

### Clear All Memories
```bash
python cli.py --clear
```

### Update System Identity
```bash
python cli.py --identity "I am a helpful coding assistant"
```

### Custom Database Path
```bash
python cli.py --db /path/to/custom.db "Your prompt"
```

## Interactive Mode Commands

When in interactive mode, you can use:

- Type your message and press Enter
- `stats` - Show memory statistics
- `quit`, `exit`, `bye`, or `q` - Exit interactive mode
- `Ctrl+C` - Exit

## Examples

```bash
# Quick question
python cli.py "Explain quantum computing"

# Start a conversation
python cli.py -i

# Check how many memories are stored
python cli.py --stats

# Clear all memories (careful!)
python cli.py --clear
```

## Output Format

Each response includes:
- The AI's response text
- Spike activity count (from SNN visualization)
- Total number of memories stored

Example:
```
You: What is AI?

GrillCheese: AI stands for Artificial Intelligence...

[Spike activity: 45 | Memories: 12]
```

