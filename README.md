# Falcon-LLM-Skincare-eCommerce-Chatbot
Project on showcasing the power of Falcon Open Source LLM to tailor the need of skincare e-commerce to guide and pick the right choices of product

## Project Overview
This project demonstrates the fine-tuning of the Falcon LLM (Large Language Model) for specific tasks. <br/>The model was fine-tuned using [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PeftModel) techniques to customize its behavior for specialized use cases. The final model is deployed and hosted on Hugging Face, ready for real-time interaction.

## Model Information
- **Base Model**: Falcon 7B
- **Fine-Tuning Method**: PEFT - Quantized
- **Framework**: PyTorch

## Files in This Repository
- `fine_tuning_llm_falcon.py`: Python script used for fine-tuning the Falcon LLM.
- `requirements.txt`: the project requirements to install dependencies
- `README.md`: Documentation for the project (this file).
- `skincare_ecommerce_chatbot_data.csv`: Dataset for the fine-tuning process in Question and Answer column format
- `app.py`: Simple UI for Gradio App to run inference of this model

## Quick Start Guide

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/llm-falcon-finetuning.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd llm-falcon-finetuning
   ```
3. **Installing Dependencies**:
   Ensure you have Python installed. Then, install the required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## Fine-Tuning the Model
To fine-tune the model, run the fine_tuning_llm_falcon.py script:
```bash
python fine_tuning_llm_falcon.py
```
This script will load the Falcon 7B model and fine-tune it according to the dataset and parameters defined in the script.

## Deployment
Once fine-tuning is complete, you can upload the model to Hugging Face's Model Hub for deployment:
```bash
huggingface-cli upload /path/to/your/fine-tuned-model
```
Replace /path/to/your/fine-tuned-model with the actual path to your model directory.

## Usage
After deployment, the fine-tuned model can be accessed via the Hugging Face API or through an interactive space. You can start using the model for various tasks like text generation, summarization, or custom chatbot applications.

# Example Usage
Here’s a basic example of how to use the fine-tuned model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yourusername/your-finetuned-model")
model = AutoModelForCausalLM.from_pretrained("yourusername/your-finetuned-model")

# Generate text
input_text = "Do you know which moisturizer to match my Dry skin?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model.generate(input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

# Model and Deployment Links
#### [Result for Hugging Face Model](https://huggingface.co/UrFavB0i/Fine-tuned-Falcon7B-skincare-chatbot)
#### [Run Hugging Face Space](https://huggingface.co/spaces/UrFavB0i/Skincare-ecommerce-assistant?logs=container)

# License
This project is licensed under the MIT License. See the LICENSE file for details.
   
