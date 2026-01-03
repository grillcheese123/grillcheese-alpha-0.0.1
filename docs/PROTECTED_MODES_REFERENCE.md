# GrillCheese Protected Modes - Quick Reference

## 🔓 Teach Mode (Public)
**Purpose**: Create permanent memories that will never be deleted  
**Access**: `python cli.py --teach`  
**Authentication**: None required

### Commands
| Command | Description | Example |
|---------|-------------|---------|
| `teach <text>` | Store protected memory | `teach Nick prefers technical responses` |
| `file <path>` | Import from text file | `file training_data.txt` |
| `list` | Show protected memories | `list` |
| `stats` | Memory statistics | `stats` |
| `quit` | Exit teach mode | `quit` |

### Use Cases
- Personal preferences
- Core knowledge
- Domain-specific facts
- Training examples

---

## 🔒 Developer Mode (Restricted)
**Purpose**: Advanced model improvement and analysis  
**Access**: `python cli.py --dev`  
**Authentication**: Password required (`grillcheese_dev_2026`)

### Commands
| Command | Description | Output |
|---------|-------------|--------|
| `export-training [file]` | Export training data | JSONL format |
| `analyze-memory` | Deep memory analysis | Statistics + patterns |
| `edit-identity` | Modify system prompt | Interactive editor |
| `tune-params` | View parameters | Current config |
| `test-retrieval <query>` | Test memory search | Top 10 results |
| `export-embeddings [file]` | Export vectors | NPZ archive |
| `brain-dump` | Full brain state | Complete dump |
| `create-dataset [file]` | Fine-tuning dataset | JSONL format |
| `stats` | System statistics | Comprehensive |
| `quit` | Exit developer mode | Exit |

### Security
- **Max attempts**: 3
- **Hash**: SHA-256
- **Env override**: `GRILLCHEESE_DEV_PASSWORD_HASH`

---

## Mode Comparison

| Feature | Teach Mode | Developer Mode |
|---------|-----------|----------------|
| **Access** | Public | Password-protected |
| **Purpose** | Memory creation | Model improvement |
| **Memory types** | Protected only | All memories |
| **Analysis** | Basic stats | Deep analysis |
| **Exports** | None | Training data, embeddings |
| **Identity editing** | No | Yes |
| **Brain inspection** | No | Yes |

---

## Quick Workflows

### Teaching GrillCheese About You
```bash
# Create training file
echo "Nick is a web developer" > about_me.txt
echo "Nick specializes in Shopify" >> about_me.txt
echo "Nick values privacy" >> about_me.txt

# Import as protected memories
python cli.py --teach
Teach> file about_me.txt
```

### Analyzing Model Performance (Developers)
```bash
python cli.py --dev
# Password: grillcheese_dev_2026

Dev> analyze-memory
Dev> test-retrieval "Shopify"
Dev> export-training training.jsonl
```

### Refining System Behavior (Developers)
```bash
python cli.py --dev

Dev> edit-identity
# Modify system prompt
# Save changes

Dev> test-retrieval "test query"
# Verify behavior
```

---

## File Formats

### Text Files (Teach Mode Input)
```
One fact per line
Another important detail
User preference or knowledge
```

### Training Export (JSONL)
```json
{"text": "...", "metadata": {...}, "timestamp": "...", "access_count": 5}
{"text": "...", "metadata": {...}, "timestamp": "...", "access_count": 2}
```

### Embeddings Export (NPZ)
```python
import numpy as np
data = np.load('embeddings.npz')
keys = data['keys']      # Query embeddings
values = data['values']  # Memory embeddings
texts = data['texts']    # Associated text
```

---

## Best Practices

### Teach Mode ✅
- One concept per line
- Be specific and factual
- Use present tense
- Review with `list` command
- Check stats regularly

### Developer Mode 🔐
- Change default password
- Export data securely
- Review changes in `edit-identity`
- Test retrieval after changes
- Backup before major edits

---

## Troubleshooting

### Teach Mode
**Problem**: File not importing  
**Solution**: Check file path, verify UTF-8 encoding

**Problem**: Memory not appearing  
**Solution**: Use `list` to verify, check `stats`

### Developer Mode
**Problem**: Authentication failed  
**Solution**: Check password, verify hash in `dev_auth.py`

**Problem**: Export failed  
**Solution**: Check permissions, verify output directory

**Problem**: Brain dump shows no data  
**Solution**: Ensure brain module is loaded (`--brain` in normal mode)

---

## Documentation Links

- Full Teach Mode Guide: `docs/TEACH_MODE.md`
- Full Developer Guide: `docs/DEVELOPER_MODE.md`
- Memory Store API: `memory_store.py`
- Authentication: `dev_auth.py`

---

**Remember**:
- 🔓 **Teach Mode**: Anyone can use, creates protected memories
- 🔒 **Developer Mode**: Password-protected, for model improvement only
