# Protected Teaching Mode

## Overview
Protected teaching mode allows you to create permanent, high-priority memories that will never be deleted during memory pruning. This is perfect for:
- Core knowledge you want GrillCheese to always remember
- Important facts about yourself
- Domain-specific information
- Training examples for specific behaviors

## How to Use

### Enter Teach Mode
```bash
python cli.py --teach
```

### Commands

#### `teach <text>` - Store a single protected memory
```
Teach> teach Nick prefers concise, technical responses without emojis
✓ Protected memory stored: Nick prefers concise, technical responses...
```

#### `file <path>` - Import training data from file
Create a text file with one training example per line:

**training.txt:**
```
Nick works on Shopify migrations for clients
Nick is developing GrillCheese AI as a side project
Nick has an AMD RX 6750 XT GPU
Nick prefers direct technical explanations
```

Then import:
```
Teach> file training.txt
Found 4 lines in training.txt
Store all 4 as protected memories? (yes/no): yes
[4/4] Storing...
✓ Imported 4 protected memories from training.txt
```

#### `list` - Show all protected memories
```
Teach> list

=== Protected Memories (4) ===
1. [2026-01-03 19:30:45] Nick works on Shopify migrations for clients
2. [2026-01-03 19:30:45] Nick is developing GrillCheese AI as a side project
3. [2026-01-03 19:30:45] Nick has an AMD RX 6750 XT GPU
4. [2026-01-03 19:30:45] Nick prefers direct technical explanations
```

#### `stats` - Show memory statistics
```
Teach> stats

=== Memory Statistics ===
Total memories: 47
Protected memories: 4
Oldest: 2026-01-01 14:23:10
Newest: 2026-01-03 19:30:45
```

#### `quit` - Exit teach mode
```
Teach> quit
Exiting teach mode...
```

## Features

### Protected Memories Are:
- ✅ Never deleted during pruning
- ✅ Higher priority in retrieval
- ✅ Preserved when running `--clear` (unless you use `--clear-all`)
- ✅ Persistent across sessions

### File Import Supports:
- `.txt` files (one example per line)
- UTF-8 encoding
- Automatic empty line filtering

## Use Cases

### Training Examples
Create `examples.txt`:
```
When Nick asks about performance, always provide benchmark numbers
When discussing code, follow AMD's review standards (no emojis, concise)
Nick values privacy - remind about local processing when relevant
```

### Personal Context
Create `context.txt`:
```
Nick lives in Shawinigan, Quebec, Canada
Nick recently recovered from H3N2 influenza A
Nick is cautious about healthcare due to past adverse reactions
Nick enjoys strategy games and true crime documentaries
```

### Domain Knowledge
Create `domain.txt`:
```
Shopify's Liquid template engine uses {% %} for logic
Phi-3 mini uses 3.8B parameters with 4k context window
AMD RDNA2 architecture supports Vulkan 1.2+ compute shaders
```

## Management

### View Protected Count
```bash
python cli.py --stats
```

### Clear Non-Protected Memories
```bash
python cli.py --clear  # Preserves protected memories
```

### Clear Everything (Including Protected)
Modify `memory_store.py` call:
```python
memory.clear(include_protected=True)
```

## Best Practices

1. **Be specific**: "Nick prefers technical responses" > "Nick likes tech"
2. **One concept per line**: Each line becomes a separate retrievable memory
3. **Use present tense**: "Nick works on X" > "Nick worked on X"
4. **Include context**: "Nick uses AMD RX 6750 XT for GrillCheese development"
5. **Keep it factual**: Protected memories influence all responses

## Examples

### Session 1: Create Training Data
```
Teach> teach Nick is a web developer specializing in Shopify migrations
✓ Protected memory stored

Teach> teach Nick values privacy and runs AI locally on his machine
✓ Protected memory stored

Teach> file preferences.txt
Store all 5 as protected memories? (yes/no): yes
✓ Imported 5 protected memories
```

### Session 2: Normal Chat (Uses Protected Memories)
```
You: What do I do for work?
GrillCheese: You're a web developer who specializes in Shopify migrations for clients.

You: Why did I build you?
GrillCheese: You built me as a privacy-focused AI that runs entirely on your local machine...
```

## Technical Details

### Database Schema
Protected memories have `is_protected = 1` in the SQLite database.

### Retrieval Priority
Protected memories are retrieved alongside regular memories but are never pruned when memory limits are reached.

### Memory Limits
Protected memories count toward the total memory limit but are exempt from pruning. Plan accordingly.

## Summary

Protected teaching mode gives you control over GrillCheese's permanent knowledge base. Use it to:
- Define your preferences
- Store important personal context
- Teach domain-specific knowledge
- Create training examples for desired behaviors

All protected memories persist across sessions and survive memory clearing operations.
