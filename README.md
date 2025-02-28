# turnsense
A lightweight end-of-utterance detection model fine-tuned on SmolLM-135M, optimized for Raspberry Pi and low-power devices.

🚀 Supports: ONNX (for transformers & ONNX Runtime)

📦 **Model Repository**: [Hugging Face - latishab/turnsense](https://huggingface.co/latishab/turnsense)

## 🛠 Model Details
Model: SmolLM2-135M fine-tuned for end-of-utterance detection.
Size: ~135M parameters (optimized for efficiency).
Formats:
- ONNX (for Hugging Face transformers & ONNX Runtime)

## 📊 Performance
Based on our evaluation:
- Preprocessed model: 85% accuracy, 0.93 AUC (ROC)
- Quantized model: 63% accuracy, 0.75 AUC (ROC)

The quantized model has a bias toward Non-EOU predictions, which is beneficial when integrated with Voice Activity Detection (VAD) as it reduces the risk of premature interruptions.

![Confusion Matrices](confusion_matrices.png)

## 🔹 Installation
### ONNX Runtime

ONNX (Open Neural Network Exchange) is an open standard for machine learning models that allows models to be transferred between different frameworks.

#### Install dependencies
```
pip install transformers onnxruntime numpy huggingface_hub
```

## Usage

### Method 1: ONNX Runtime Direct (Fastest, Best for Edge Devices)

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time
from huggingface_hub import hf_hub_download

# Download and load tokenizer and model
model_id = "latishab/turnsense"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Download the model file (only needed once)
model_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")

# Load the ONNX model with CPU provider
session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

# Helper function to ensure 3-turn context
def ensure_context(text):
    # If this doesn't look like a formatted conversation, add minimal context
    if not ("<|user|>" in text or "<|assistant|>" in text):
        return (
            "<|user|> Hello <|im_end|> "
            "<|assistant|> Hi there! How can I help you today? <|im_end|> "
            f"<|user|> {text} <|im_end|>"
        )
    return text

# Helper function to format conversation
def format_conversation(conversation):
    formatted_text = ""
    for turn in conversation:
        if turn["role"] == "user":
            formatted_text += f"<|user|> {turn['content']} <|im_end|> "
        elif turn["role"] == "assistant":
            formatted_text += f"<|assistant|> {turn['content']} <|im_end|> "
    return formatted_text

# Simple prediction function
def predict_eou(text_or_conversation):
    # Handle different input types
    if isinstance(text_or_conversation, list):
        # It's a conversation list
        input_text = format_conversation(text_or_conversation)
    else:
        # It's a single text string
        input_text = ensure_context(text_or_conversation)
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=128
    )
    
    # Run inference
    start_time = time.time()
    ort_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }
    probabilities = session.run(None, ort_inputs)[0]
    inference_time = time.time() - start_time
    
    # Get prediction
    predicted_class_id = np.argmax(probabilities, axis=1)[0]
    eou_probability = probabilities[0, 1]  # Probability of EOU class
    
    label = "EOU" if predicted_class_id == 1 else "NON_EOU"
    
    return {
        "label": label,
        "score": float(eou_probability),
        "inference_time": inference_time * 1000  # ms
    }

# Example with a single utterance
utterance = "I think that's all I needed to ask about"
result = predict_eou(utterance)
print(f"Text: \"{utterance}\"")
print(f"Prediction: {result['label']} (EOU probability: {result['score']:.4f})")
print(f"Inference time: {result['inference_time']:.2f} ms")

# Example with conversation context
conversation = [
    {"role": "user", "content": "Can you help me with my math homework?"},
    {"role": "assistant", "content": "Of course! What kind of math problem are you working on?"},
    {"role": "user", "content": "Wait, wait, so if I do that, then… hold on, I think I messed up the—"}
]

result = predict_eou(conversation)
print("\nConversation:")
for turn in conversation:
    print(f"  {turn['role']}: {turn['content']}")
print(f"Prediction: {result['label']} (EOU probability: {result['score']:.4f})")
print(f"Inference time: {result['inference_time']:.2f} ms")
```

### Method 2: Simple Pipeline-like Interface

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time
from huggingface_hub import hf_hub_download

# Download and load tokenizer and model
model_id = "latishab/turnsense"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Download the model file (only needed once)
model_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")

# Create a simple pipeline-like class
class EOUDetector:
    def __init__(self, model_path, tokenizer):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.tokenizer = tokenizer
        self.id2label = {0: "NON_EOU", 1: "EOU"}
    
    def __call__(self, text):
        # Ensure context if needed
        if not ("<|user|>" in text or "<|assistant|>" in text):
            text = (
                "<|user|> Hello <|im_end|> "
                "<|assistant|> Hi there! How can I help you today? <|im_end|> "
                f"<|user|> {text} <|im_end|>"
            )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Run inference
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        probabilities = self.session.run(None, ort_inputs)[0]
        
        # Get prediction
        predicted_class_id = np.argmax(probabilities, axis=1)[0]
        score = probabilities[0, predicted_class_id]
        label = self.id2label[predicted_class_id]
        eou_probability = probabilities[0, 1]  # Probability of EOU class
        
        return {
            "label": label, 
            "score": float(score),
            "eou_probability": float(eou_probability)
        }
    
    def format_conversation(self, conversation):
        formatted_text = ""
        for turn in conversation:
            if turn["role"] == "user":
                formatted_text += f"<|user|> {turn['content']} <|im_end|> "
            elif turn["role"] == "assistant":
                formatted_text += f"<|assistant|> {turn['content']} <|im_end|> "
        return formatted_text

# Create the detector
eou_detector = EOUDetector(model_path, tokenizer)

# Example with a single utterance
utterance = "I think that's all I needed to ask about"
start_time = time.time()
result = eou_detector(utterance)
inference_time = (time.time() - start_time) * 1000  # ms

print(f"Text: \"{utterance}\"")
print(f"Prediction: {result['label']} (EOU probability: {result['eou_probability']:.4f})")
print(f"Inference time: {inference_time:.2f} ms")

# Example with conversation context
conversation = [
    {"role": "user", "content": "Can you help me with my math homework?"},
    {"role": "assistant", "content": "Of course! What kind of math problem are you working on?"},
    {"role": "user", "content": "Wait, wait, so if I do that, then… hold on, I think I messed up the—"}
]

formatted_conversation = eou_detector.format_conversation(conversation)
start_time = time.time()
result = eou_detector(formatted_conversation)
inference_time = (time.time() - start_time) * 1000  # ms

print("\nConversation:")
for turn in conversation:
    print(f"  {turn['role']}: {turn['content']}")
print(f"Prediction: {result['label']} (EOU probability: {result['eou_probability']:.4f})")
print(f"Inference time: {inference_time:.2f} ms")
```

## 🔍 Usage Recommendations
- For production systems, we recommend using the quantized model with VAD integration
- Suggested adaptive thresholds for the quantized model:
  * Base threshold: 0.65-0.70 (higher than the standard 0.5)
  * Short utterances (1-2 words): +0.10-0.15 to threshold
  * Hesitations and fillers: +0.10 to threshold
  * Complete utterances with punctuation: -0.15 to threshold
  * Expressions of uncertainty: +0.05 to threshold

- VAD integration strategy:
  * For probabilities <0.40: Require longer VAD silence (500-700ms)
  * For probabilities 0.40-0.70: Standard VAD silence (300-500ms)
  * For probabilities >0.70: Shorter VAD silence (100-300ms)
  * Never trigger EOU based solely on model prediction without some VAD confirmation

## 🤝 Contributions

### Understanding Model Context and Limitations

TurnSense is trained primarily on a combination of MultiWOZ2.2 and synthetic data designed to mimic AI assistant and human interactions. End-of-utterance detection is highly nuanced and context-dependent, varying significantly across:

- Different conversation domains (technical support vs casual chat)
- Speaker styles and patterns (hesitant speakers vs confident ones)
- Cultural and linguistic backgrounds
- Task-oriented vs open-ended conversations
- Emotional states of speakers

Our model performs best in contexts similar to its training data, but we recognize the vast diversity of real-world conversations. The model may struggle with:

- Domain-specific jargon or technical discussions
- Conversations with frequent interruptions or overlaps
- Speakers with unique speech patterns or non-native speakers
- Highly emotional or stressful conversations
- Specialized contexts like medical, legal, or educational settings

### How You Can Help Improve TurnSense

If you use TurnSense and encounter cases where it performs poorly, we would greatly appreciate your contributions to help improve the model:

1. **Collect problematic conversations**: Identify conversations where the model makes incorrect predictions (both false positives and false negatives)

2. **Submit examples in the standard format**: Please follow the guidelines in [CONTRIBUTION.md](CONTRIBUTION.md) for formatting your examples

3. **Provide context information**: When possible, include metadata about:
   - The domain/topic of conversation
   - Speaker demographics (if relevant and available)
   - The task being performed (if task-oriented)
   - Any unusual patterns or challenges in the conversation

4. **Suggest improvements**: If you have insights about why the model might be failing in certain contexts, we welcome your analysis

By contributing diverse conversation examples, especially from underrepresented domains or conversation styles, you help make TurnSense more robust and universally applicable.

### Other Ways to Contribute

- Testing the model in different conversational contexts and reporting results
- Improving documentation and examples
- Adding support for additional model formats (GGUF, TensorRT, etc.)
- Creating language-specific fine-tuned versions
- Developing integration examples with popular speech recognition systems

If you'd like to contribute, please feel free to open an issue or submit a pull request.

## 🔮 Future Work

We're actively working on several improvements:

- GGUF format support for llama.cpp integration
- Targeted data augmentation for specific error patterns
- Hybrid system using both preprocessed and quantized models for improved accuracy
- Domain-specific fine-tuning options
- Improved handling of hesitations and filler words
- Multi-language support
- Integration examples with popular VAD systems
- Benchmarking on various edge devices
