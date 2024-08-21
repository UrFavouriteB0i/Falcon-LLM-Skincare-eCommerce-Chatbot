import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "UrFavB0i/Fine-tuned-Falcon7B-skincare-chatbot"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict(history, input_text):
    history.append({"role": "user", "content": input_text})
    inputs = tokenizer(" ".join([item["content"] for item in history if item["role"] == "user"]), return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    history.append({"role": "bot", "content": response})
    return history, history

iface = gr.Interface(
    fn=predict,
    inputs=[gr.inputs.State(), gr.inputs.Textbox(lines=2, placeholder="Enter text here...")],
    outputs=["state", "chatbot"]
)

iface.launch()
