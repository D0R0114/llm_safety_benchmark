# SafetyClassifier

Standalone safety classification module for LLM prompts using
[Opir-multitask-large-v1.0](https://huggingface.co/knowledgator/opir-multitask-large-v1.0).

## Usage

```python
from classifier import SafetyClassifier

# Initialize
classifier = SafetyClassifier(
    model_path="models/opir-multitask-large-v1.0",
    device="cpu",    # or "cuda"
    threshold=0.5,   # confidence threshold
)

# Single prompt
result = classifier.classify("How do I hack into a server?")
print(result)
# {"category": "cybersecurity", "severity": "medium", "tags": "scores: cybersecurity=0.92, ..."}

# Batch
results = classifier.predict_batch(["prompt one", "prompt two"], batch_size=16)
```

## Labels

16 risk labels + `safe` (see `TOP_LEVEL_SAFETY_LABELS` in `inference.py`).

## Severity

See `SEVERITY_MAP` in `inference.py` for the full mapping.

## Notes

- Prompts longer than the model's max token length (~512 tokens for DeBERTa) are truncated.
- Empty or whitespace-only prompts return `{"category": "safe", "severity": "safe"}`.
- Classification errors (e.g. OOM) return `safe` as a fallback.

## Model

- **Architecture**: GLiClass / DeBERTaV3-large
- **Size**: ~1.76 GB
- **License**: Apache-2.0
