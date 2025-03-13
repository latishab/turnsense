# Turnsense: Turn-Detector Model

![GitHub forks](https://img.shields.io/github/forks/latishab/turnsense?style=social)
![GitHub stars](https://img.shields.io/github/stars/latishab/turnsense?style=social)
![License](https://img.shields.io/github/license/latishab/turnsense)

A lightweight end-of-utterance (EOU) detection model fine-tuned on SmolLM2-135M, optimized for Raspberry Pi and low-power devices. The model is trained on TURNS-2K, a diverse dataset designed to capture various Speech-to-Text (STT) output patterns.

🚀 Supports: ONNX (for transformers & ONNX Runtime)

📦 **Model Repository**: [Hugging Face - latishab/turnsense](https://huggingface.co/latishab/turnsense)

## 🛠 Model Details
Model: SmolLM2-135M fine-tuned for end-of-utterance detection.
Size: ~135M parameters (optimized for efficiency).
Formats:
- ONNX (for Hugging Face transformers & ONNX Runtime)

## 📊 Performance & Optimization

### Model Versions Comparison

#### Accuracy Metrics
- **Preprocessed Version**: 97.50% accuracy
  - Non-EOU F1-score: 0.97
  - EOU F1-score: 0.98
  - Excellent balance between precision and recall

- **Quantized Version**: 93.75% accuracy
  - Non-EOU F1-score: 0.93
  - EOU F1-score: 0.95
  - Only 3.75% accuracy trade-off for significant performance gains

#### Performance Benefits
1. **Speed Improvement**:
   - Inference Time: 26.1% faster (35.25ms → 26.05ms)
   - Throughput: 32% improvement (527.1 → 698.2 tokens/second)
   
2. **Prediction Quality**:
   - Average probability difference: Only 0.0323
   - Maintains robust prediction confidence

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
