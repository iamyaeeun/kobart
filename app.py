from flask import Flask,request,jsonify
import pickle
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

app = Flask(__name__)

tokenizer=PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model=BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

@app.route('/sum',methods=['POST'])
def summary():
    text = request.form['text']
    text = text.replace('\n', ' ')

    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

    summary_ids = model.generate(torch.tensor([input_ids]), num_beams=4, max_length=512, eos_token_id=1)
    s = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

    result={'text':s}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) #host="192.168.0.7",port=5000
