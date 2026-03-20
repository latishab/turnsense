# Turnsense: Turn-Detector Model

![GitHub forks](https://img.shields.io/github/forks/latishab/turnsense?style=social)
![GitHub stars](https://img.shields.io/github/stars/latishab/turnsense?style=social)
![License](https://img.shields.io/github/license/latishab/turnsense)

## Overview

Turnsense is an open-source end-of-utterance (EOU) detection model for real-time voice AI applications. Built on SmolLM2-135M and optimized for low-power devices like Raspberry Pi.

End-of-utterance detection determines when an AI should respond to human speech. Traditional systems rely on simple Voice Activity Detection (VAD). Turnsense instead analyzes linguistic and semantic patterns from the text output of an STT system.

Supports: ONNX (for transformers & ONNX Runtime)

**Model Repository**: 
- GitHub: https://github.com/latishab/turnsense
- Hugging Face: https://huggingface.co/latishab/turnsense

## Key Features

- **Lightweight**: Built on SmolLM2-135M (~135M parameters)
- **High accuracy**: 97.50% (standard) / 93.75% (quantized)
- **Edge-ready**: Runs on Raspberry Pi and similar hardware
- **ONNX support**: Works with ONNX Runtime and Hugging Face Transformers

## Performance

- **Standard model**: 97.50% accuracy
- **Quantized model**: 93.75% accuracy
- **Average probability difference**: 0.0323 between versions

![confusion_matrices](https://github.com/user-attachments/assets/1824aae3-41a9-459e-bcaf-1afb83997689)

### Speed

![speed](https://github.com/user-attachments/assets/1d6e4666-01c2-4a75-a3f2-f445c21033bd)

## Limitations

- **Punctuation dependence**: Trained on text with proper punctuation. Short utterances without punctuation (e.g., "Hello") may be ambiguous.
- **STT quality**: Performance depends on the quality of the upstream STT system. Better STT with proper punctuation leads to better turn detection.

## Installation
```bash
pip install transformers onnxruntime numpy huggingface_hub
```

## Quick Start

```python
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Download and load tokenizer and model
model_id = "latishab/turnsense"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")

# Initialize ONNX Runtime session
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Prepare input
# Note: The special token <|user|> is included, but <|im_end|> is not.
text = "Hello, how are you?"
inputs = tokenizer(
    f"<|user|> {text}",
    padding="max_length",
    max_length=256,
    return_tensors="np"  
)

# Run inference
ort_inputs = {
    'input_ids': inputs['input_ids'].numpy(),
    'attention_mask': inputs['attention_mask'].numpy()
}
all_logits = session.run(None, ort_inputs)[0]
logits_for_item = all_logits[0]
prediction = np.argmax(logits_for_item)

print(f"Text: '{text}'")
print(f"Prediction (0 or 1): {prediction}")
```

## Dataset: [TURNS-2K](https://huggingface.co/datasets/latishab/turns-2k)

Trained on TURNS-2K, a dataset built for end-of-utterance detection. It covers:

- Backchannels and self-corrections
- Code-switching and language mixing
- Multiple text formatting styles
- Variations in STT output across different systems

## Motivation and current state

I built Turnsense because I couldn't find a good open-source turn detection model for edge devices. Most options were either proprietary or too heavy to run on something like a Raspberry Pi.

The model is trained on English speech patterns using 2,000 samples via LoRA fine-tuning on SmolLM2-135M. It handles common STT outputs well, but there are edge cases and complex conversational patterns it doesn't cover yet. ONNX was a deliberate choice for device compatibility, though a port to Apple MLX is on the table.

## License
Apache 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Some areas that could use help: dataset expansion, model optimization, documentation, and bug reports. Feel free to open a PR or issue.

## Citation
If you use this model in your research:

```bibtex
@software{latishab2025turnsense,
  author       = {Latisha Besariani HENDRA},
  title        = {Turnsense: A Lightweight End-of-Utterance Detection Model},
  month        = mar,
  year         = 2025,
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/latishab/turnsense},
  note         = {https://huggingface.co/latishab/turnsense}
}
```
