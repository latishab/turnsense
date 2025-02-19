# turnsense
A lightweight end-of-utterance detection model fine-tuned on SmolLM-135M, optimized for Raspberry Pi and low-power devices.
🚀 Supports: ONNX (for transformers & ONNX Runtime) | GGUF (for llama.cpp)


## 🛠 Model Details
Model: SmolLM-135M fine-tuned for end-of-utterance detection.
Size: ~135M parameters (optimized for efficiency).
Formats:
- ONNX (for Hugging Face transformers & ONNX Runtime)
- GGUF (for llama.cpp, CPU inference on Raspberry Pi)

## 🔹 Installation
Option 1: ONNX (Hugging Face Transformers)
Best for integration with existing Transformers-based pipelines.

### Install dependencies
```
pip install transformers onnxruntime optimum
```

### Run inference
```
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model_path = "your-username/turnsense-onnx"
model = ORTModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_text = "User: Wait, wait, so if I do that, then… hold on, I think I messed up the—"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

print(outputs)  # Should return an "eou_probability" score
```

## Option 2: GGUF (Optimized for Raspberry Pi)
Best for fast, CPU-only inference on edge devices.

## Install llama.cpp
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

### Download and run the model
```
wget https://huggingface.co/your-username/turnsense-gguf/blob/main/model.gguf
./main -m model.gguf -p "User: Uh, I was thinking maybe we could—"
```
