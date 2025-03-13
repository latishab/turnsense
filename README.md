# Turnsense: Turn-Detector Model

![GitHub forks](https://img.shields.io/github/forks/deepsealabs/libdc-swift?style=social)
![GitHub stars](https://img.shields.io/github/stars/deepsealabs/libdc-swift?style=social)
![License](https://img.shields.io/github/license/deepsealabs/libdc-swift)

A lightweight end-of-utterance (EOU) detection model fine-tuned on SmolLM2-135M, optimized for Raspberry Pi and low-power devices. The model is trained on TURNS-2K, a diverse dataset designed to capture various Speech-to-Text (STT) output patterns, including backchannels, mispronunciations, code-switching, and different text formatting styles. This makes the model robust across different STT systems and their output variations.

🚀 Supports: ONNX (for transformers & ONNX Runtime)

📦 **Model Repository**: [Hugging Face - latishab/turnsense](https://huggingface.co/latishab/turnsense)

## 🛠 Model Details
Model: SmolLM2-135M fine-tuned for end-of-utterance detection.
Size: ~135M parameters (optimized for efficiency).
Formats:
- ONNX (for Hugging Face transformers & ONNX Runtime)

## 📊 Performance
The model achieves 97.50% accuracy with the preprocessed version and 93.75% with the quantized version, with an average probability difference of only 0.0323 between them. Both versions maintain high F1-scores (0.97-0.98 for preprocessed, 0.93-0.95 for quantized) across EOU and non-EOU classes.

![confusion_matrices](https://github.com/user-attachments/assets/1824aae3-41a9-459e-bcaf-1afb83997689)

![speed](https://github.com/user-attachments/assets/1d6e4666-01c2-4a75-a3f2-f445c21033bd)

## 🔹 Installation
```bash
pip install transformers onnxruntime numpy huggingface_hub
```

## 🚀 Quick Start

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
text = "Hello, how are you?"
inputs = tokenizer(
    f"<|user|> {text} <|im_end|>",
    padding="max_length",
    max_length=256,
    return_tensors="pt"
)

# Run inference
ort_inputs = {
    'input_ids': inputs['input_ids'].numpy(),
    'attention_mask': inputs['attention_mask'].numpy()
}
probabilities = session.run(None, ort_inputs)[0]
```

## 📝 Dataset
The model is trained on TURNS-2K, a diverse dataset designed to capture various Speech-to-Text (STT) output patterns, including:
- Backchannels, mispronunciations, self-corrections
- Code-switching and language mixing
- Different text formatting styles (clean punctuation, lowercase without punctuation)
- Makes the model robust across different STT systems and their output variations

## 📄 License
Apache 2.0

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
