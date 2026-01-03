# Developer Mode - RESTRICTED ACCESS

## ⚠️ CREATOR-ONLY MODE

This mode is **password-protected** and only accessible to GrillCheese developers. It provides advanced tools for model improvement, analysis, and fine-tuning.

## Authentication

### Default Credentials
- **Password**: `grillcheese_dev_2026`
- **Hash**: `398182f2b68931c2e2dbd9a7f65c90ed5ae682ef8399793f1afdb3dcf8fa9c74`

### Changing the Password

Generate new hash:
```bash
python -c "import hashlib; print(hashlib.sha256(b'your_new_password').hexdigest())"
```

Update in `dev_auth.py`:
```python
DEV_PASSWORD_HASH = "your_generated_hash"
```

Or use environment variable:
```bash
export GRILLCHEESE_DEV_PASSWORD_HASH="your_generated_hash"
python cli.py --dev
```

## Accessing Developer Mode

```bash
python cli.py --dev
```

You'll be prompted for authentication:
```
DEVELOPER MODE - AUTHENTICATION REQUIRED
Developer password: **********************
✓ Authentication successful
```

**Failed attempts**: 3 maximum before lockout.

## Developer Commands

### `export-training` - Export Training Data
Export conversation pairs for fine-tuning:

```
Dev> export-training training_data.jsonl
Exporting training data to training_data.jsonl...
✓ Exported 1,247 entries to training_data.jsonl
```

**Output format** (JSONL):
```json
{"text": "conversation text", "metadata": {...}, "timestamp": "2026-01-03T19:30:45", "access_count": 5}
{"text": "another entry", "metadata": {...}, "timestamp": "2026-01-03T19:31:12", "access_count": 2}
```

**Use case**: Create training datasets for model fine-tuning.

### `analyze-memory` - Deep Memory Analysis
Comprehensive memory statistics and access patterns:

```
Dev> analyze-memory

=== Deep Memory Analysis ===

Total memories: 1,247
Protected: 42 (3.4%)
Identity: 1
Regular: 1,204

Access Patterns:
Average access count: 3.24
Max access count: 47

Most Accessed Memories:
1. [47 accesses] Nick prefers concise, technical responses...
2. [32 accesses] Nick works on Shopify migrations...
3. [28 accesses] Nick is developing GrillCheese AI...
4. [24 accesses] Nick has AMD RX 6750 XT GPU...
5. [21 accesses] Nick lives in Shawinigan, Quebec...

Memories Created (Last 7 Days):
2026-01-03: 184 memories
2026-01-02: 143 memories
2026-01-01: 97 memories
```

**Use case**: Understand memory usage patterns, identify important topics.

### `edit-identity` - Modify System Identity
Edit the core system prompt interactively:

```
Dev> edit-identity

=== Edit System Identity ===

Current identity:
------------------------------------------------------------
You are GrillCheese, a helpful AI assistant running locally...
------------------------------------------------------------

Options:
1. Edit in external editor
2. Replace with new text
3. Cancel

Choice (1-3): 1
[Opens in $EDITOR]
✓ Identity updated
```

**Use case**: Refine system behavior, adjust personality, update capabilities.

### `tune-params` - View Model Parameters
Display current runtime parameters:

```
Dev> tune-params

=== Model Parameter Tuning ===

Current parameters:
  Temperature: 0.7
  Top-P: 0.9
  Max tokens (GPU): 256
  Max context items: 5

Note: Changes are runtime only. Edit config.py for persistence.
```

**Use case**: Quick reference for current configuration.

### `test-retrieval` - Test Memory Search
Test how well memories are retrieved:

```
Dev> test-retrieval Shopify

Testing retrieval for: 'Shopify'

Retrieved 10 memories:
1. Nick works on Shopify migrations for clients
2. Shopify's Liquid template engine uses {% %} for logic
3. Recent Shopify migration project completed
4. Client requested custom Shopify checkout flow
5. Shopify theme development best practices
...
```

**Use case**: Validate memory retrieval quality, tune embedding strategy.

### `export-embeddings` - Export Embedding Space
Export embeddings for external analysis:

```
Dev> export-embeddings embeddings.npz
Exporting embeddings to embeddings.npz...
✓ Exported 1,247 embeddings
```

**Output**: NumPy compressed archive with:
- `keys`: Embedding vectors (queries)
- `values`: Embedding vectors (memories)
- `texts`: Associated text content

**Use case**: Analyze embedding space, visualize clusters, detect drift.

### `brain-dump` - Full Brain State
Complete brain module inspection:

```
Dev> brain-dump

=== Brain State Dump ===

Amygdala State:
  Valence: 0.123
  Arousal: 0.456
  Dominant: neutral

CNS State:
  Consciousness: ALERT
  Stress: 0.234

Basal Ganglia:
  Current strategy: empathetic
  Hebbian weights shape: (384, 64)

Experience:
  Total interactions: 1,247
  Average quality: 0.723
  Best strategy: empathetic
```

**Use case**: Debug emotional intelligence, analyze strategy selection.

### `create-dataset` - Fine-Tuning Dataset
Create structured fine-tuning dataset:

```
Dev> create-dataset finetune.jsonl
Creating fine-tuning dataset: finetune.jsonl
```

**Note**: Requires conversation history tracking implementation.

**Use case**: Prepare data for model fine-tuning on specific behaviors.

### `stats` - Comprehensive Statistics
Full system statistics:

```
Dev> stats

=== Comprehensive System Statistics ===

=== Memory Statistics ===
Total memories: 1,247
...

=== Brain Statistics ===
Total interactions: 1,247
Current strategy: empathetic
Stress level: 0.23

=== GPU Statistics ===
Vulkan compute: ENABLED
Embedding dimension: 384
```

**Use case**: System health check, performance monitoring.

## Advanced Workflows

### Workflow 1: Improve Model Responses

1. **Analyze what's being remembered**:
   ```
   Dev> analyze-memory
   ```

2. **Test retrieval quality**:
   ```
   Dev> test-retrieval "your test query"
   ```

3. **Export for analysis**:
   ```
   Dev> export-embeddings analysis.npz
   ```

4. **Adjust identity if needed**:
   ```
   Dev> edit-identity
   ```

### Workflow 2: Create Fine-Tuning Dataset

1. **Export training data**:
   ```
   Dev> export-training training.jsonl
   ```

2. **Analyze access patterns**:
   ```
   Dev> analyze-memory
   ```

3. **Filter high-quality examples**:
   - Sort by access_count
   - Include well-rated interactions
   - Remove noise

4. **Format for fine-tuning**:
   - Convert to prompt/completion pairs
   - Add system prompts
   - Validate quality

### Workflow 3: Debug Emotional Intelligence

1. **Check brain state**:
   ```
   Dev> brain-dump
   ```

2. **Analyze strategy performance**:
   ```
   Dev> stats
   ```

3. **Test different scenarios**:
   - Create test interactions
   - Monitor state changes
   - Validate responses

## Security Best Practices

### Password Management
1. **Change default password immediately**
2. **Use strong, unique password**
3. **Never commit password to git**
4. **Use environment variable in production**

### Access Control
- Only share password with core team
- Rotate password periodically
- Monitor access attempts
- Review exported data

### Data Protection
- Exported data may contain user information
- Handle training data with care
- Follow privacy regulations
- Secure storage for exports

## Troubleshooting

### Authentication Failed
```
✗ Invalid password. 2 attempts remaining.
```
**Solution**: Verify password, check environment variable.

### Command Not Found
```
Unknown command: xyz
```
**Solution**: Type command name exactly, check spelling.

### Export Failed
```
Error: Permission denied
```
**Solution**: Check file permissions, verify output directory exists.

## Development Roadmap

### Planned Features
- [ ] Conversation tracking for training pairs
- [ ] Automated quality scoring
- [ ] Embedding space visualization
- [ ] A/B testing framework
- [ ] Response quality metrics
- [ ] Batch processing tools

### Contribution
Developer mode is for internal use only. Features can be added to:
- `cli.py` - Add new commands
- `dev_auth.py` - Enhance security
- New modules - Advanced analysis tools

## Summary

Developer mode provides GrillCheese creators with powerful tools to:
- ✅ Export training data
- ✅ Analyze memory patterns  
- ✅ Edit system identity
- ✅ Test retrieval quality
- ✅ Export embeddings
- ✅ Debug brain states
- ✅ Create datasets
- ✅ Monitor system health

**Default password**: `grillcheese_dev_2026`

**Access**: `python cli.py --dev`

**Remember**: This mode is password-protected for a reason. Use responsibly and secure all exported data.
