---
base_model: unsloth/tinyllama-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/tinyllama-bnb-4bit
- lora
- transformers
- unsloth
- instruction-tuned
- tamil
- english
---

# TinyLlama Instruct Lite v1

## 📌 Model Summary
`rogersam/tinyllama-instruct-lite-v1` is a **LoRA fine-tuned TinyLlama model** using [Unsloth](https://github.com/unslothai/unsloth).  
It is designed for **instruction-following tasks** in **English + Tamil**, such as:
- General Q&A  
- Summarization  
- Basic math & reasoning  
- English ↔ Tamil translation  

This project demonstrates how a **lightweight 1B model** can be adapted for multiple domains with limited resources.  

---

## 🔎 Model Details
- **Developed by:** Roger Samuel J ([Hugging Face Profile](https://huggingface.co/rogersam))  
- **Model type:** Causal LM (decoder-only)  
- **Languages:** English, Tamil  
- **License:** Same as base model (TinyLlama)  
- **Fine-tuned from:** `unsloth/tinyllama-bnb-4bit`  
- **Method:** LoRA via PEFT + Unsloth  

---

## 📂 Model Sources
- **Model Repo:** [rogersam/tinyllama-instruct-lite-v1](https://huggingface.co/rogersam/tinyllama-instruct-lite-v1)  
- **Base Model:** [unsloth/tinyllama-bnb-4bit](https://huggingface.co/unsloth/tinyllama-bnb-4bit)  

---

## 💡 Uses

### Direct Use
- Running lightweight instruction tasks on CPU/GPU  
- Translating English ↔ Tamil sentences  
- Answering short questions and reasoning queries  
- Summarizing small texts  

### Out-of-Scope
- Sensitive decision-making (finance, healthcare, law)  
- Long context generation (>512 tokens)  
- Production-grade chatbots  

---

## ⚠️ Bias, Risks & Limitations
- Small dataset → may hallucinate facts  
- Not aligned for safety or toxicity filtering  
- Limited Tamil coverage (basic sentences only)  

**Recommendation:** Use for demo & educational purposes only.  

---

## 🚀 How to Get Started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "rogersam/tinyllama-instruct-lite-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Translate English to Tamil: How are you?"
print(pipe(prompt, max_new_tokens=50)[0]["generated_text"])
