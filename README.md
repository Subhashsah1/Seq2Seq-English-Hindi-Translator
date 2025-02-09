# 🌐 Advanced English-Hindi Neural Machine Translation

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)

A state-of-the-art bidirectional English-Hindi translation model built using Facebook's mBART architecture and fine-tuned on a comprehensive parallel corpus of 15,000+ sentence pairs.

## 🚀 Key Features

- Bidirectional translation support (English ↔ Hindi)
- Built on Facebook's powerful mBART-large-50 architecture
- Fine-tuned on 15,000+ high-quality parallel sentence pairs
- Supports both formal and conversational text
- GPU-accelerated inference for fast translation
- Integrated with Hugging Face's Transformers library
- Easy-to-use Python interface

## 💻 Technical Details

- Base Model: facebook/mbart-large-50-many-to-many-mmt
- Training Dataset: 15,013 parallel sentence pairs
- Optimizations:
  - Mixed precision training (FP16)
  - Batch size: 8
  - Learning rate: 0.01
  - Weight decay: 0.01
  - Beam search decoding with num_beams=5

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/english-hindi-translator.git

# Install dependencies
pip install transformers torch datasets evaluate sentencepiece
```

## 📚 Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def translate(text, direction="en-hi"):
    # Initialize model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("your-model-path")
    tokenizer = AutoTokenizer.from_pretrained("your-model-path")
    
    # Prepare prompt
    if direction == "en-hi":
        prompt = f"translate English to Hindi: {text}"
    else:
        prompt = f"translate Hindi to English: {text}"

    # Generate translation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                      padding="max_length", max_length=128)
    outputs = model.generate(**inputs, max_length=100, num_beams=5)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
english_text = "Mr. Subash is working on machine learning system. He is 25 years old."
hindi_translation = translate(english_text, direction="en-hi")
```

## 📊 Model Architecture

The model utilizes the mBART architecture, which includes:
- 12 encoder layers
- 12 decoder layers
- 16 attention heads
- 1024 hidden dimension size
- 250K vocabulary size

## 🎯 Future Improvements

- Increase training epochs for better convergence
- Implement more sophisticated data augmentation
- Add support for specialized domains (technical, medical, legal)
- Create a web interface for easy access
- Optimize model size for faster inference

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Facebook AI Research for the mBART architecture
- Hugging Face team for the Transformers library
- The open-source NLP community

---
⭐ If you found this project helpful, please consider giving it a star!