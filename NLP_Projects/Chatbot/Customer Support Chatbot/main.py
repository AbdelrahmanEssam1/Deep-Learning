from flask import Flask, request, render_template, jsonify
from transformers import T5Tokenizer , T5ForConditionalGeneration
import re

app = Flask(__name__)

model = T5ForConditionalGeneration.from_pretrained("./chatbot_model")
tokenizer = T5Tokenizer.from_pretrained("./chatbot_model")
device = model.device

def clean_text(text):
    text = re.sub(r'\r\n' , " " , text)
    text = re.sub(r'\s+' , " " , text)
    text = re.sub(r"<.*?>" , "" , text)
    text = text.strip().lower()
    return text

def chatbot(user_message):
    user_message = clean_text(user_message)
    inputs = tokenizer(user_message, return_tensors="pt" , truncation = True , padding = "max_length", max_length=250)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output = model.generate(
        inputs['input_ids'],
        max_length=250,
        num_beams=5,
        early_stopping=True,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response



#routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat' , methods = ['POST'])
def chat():
    
    user_message = request.json.get("message" , "")

    if not user_message:
        return jsonify({"error" : "Message is required"}), 400
    response = chatbot(user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug = True)