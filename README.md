# Turnsense: Turn-Detector Model

![GitHub forks](https://img.shields.io/github/forks/latishab/turnsense?style=social)
![GitHub stars](https://img.shields.io/github/stars/latishab/turnsense?style=social)
![License](https://img.shields.io/github/license/latishab/turnsense)

## Overview

Turnsense is an open-source end-of-utterance (EOU) detection model, designed specifically for real-time voice AI applications. Built on SmolLM2-135M and optimized for low-power devices like Raspberry Pi, it offers high accuracy while maintaining efficient performance.

End-of-utterance detection is crucial in conversational AI systems, determining when an AI should respond to human speech. While traditional systems rely on simple Voice Activity Detection (VAD), Turnsense takes a more sophisticated approach by analyzing both linguistic and semantic patterns.

🚀 Supports: ONNX (for transformers & ONNX Runtime)

📦 **Model Repository**: 
- GitHub: https://github.com/latishab/turnsense
- Hugging Face: https://huggingface.co/latishab/turnsense

## 🔑 Key Features

- **Lightweight Architecture**: Built on SmolLM2-135M (~135M parameters)
- **High Performance**: 97.50% accuracy (standard) / 93.75% accuracy (quantized)
- **Resource Efficient**: Optimized for edge devices and low-power hardware
- **ONNX Support**: Compatible with ONNX Runtime and Hugging Face Transformers

## 📊 Performance Metrics

The model demonstrates robust performance across different configurations:

- **Standard Model**: 97.50% accuracy
- **Quantized Model**: 93.75% accuracy
- **Average Probability Difference**: 0.0323 between versions

![confusion_matrices](https://github.com/user-attachments/assets/1824aae3-41a9-459e-bcaf-1afb83997689)

### Speed Performance

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

## 📚 Dataset: TURNS-2K

The model is trained on TURNS-2K, a comprehensive dataset specifically designed for end-of-utterance detection. It captures diverse speech patterns including:

- Backchannels and self-corrections
- Code-switching and language mixing
- Multiple text formatting styles
- Speech-to-Text (STT) output variations

This diverse training data ensures robustness across different:
- Speech patterns and dialects
- STT systems and their output formats
- Use cases and deployment scenarios

## 💭 Motivation & Current State

The inspiration for Turnsense came from a notable gap in the open-source AI landscape - the scarcity of efficient, lightweight turn detection models. While building a local conversational AI agent, I found that most available solutions were either proprietary or too resource-intensive for edge devices. This led to the development of Turnsense, a practical solution designed specifically for real-world deployment on hardware like Raspberry Pi.

Currently, the model is trained primarily on English speech patterns using a modest dataset of 2,000 samples through LoRA fine-tuning on SmolLM2-135M. While it handles common speech-to-text outputs effectively, there are certainly edge cases and complex conversational patterns yet to be addressed. The choice of ONNX format was deliberate, prioritizing compatibility with low-power devices, though we're exploring potential ports to platforms like Apple MLX.

The project's success relies heavily on community involvement. Whether it's expanding the dataset, adding multilingual support, or improving pattern recognition for complex conversational scenarios, contributions of all kinds can help evolve Turnsense into a more robust and versatile tool.

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Areas where you can help:
- Dataset expansion
- Model optimization
- Documentation improvements
- Bug reports and fixes

Please feel free to submit a Pull Request or open an Issue.

## 📚 Citation
If you use this model in your research, please cite it using:

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
