from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
import os
app = Flask(__name__)

model_path = "ai_model"
if not os.path.exists(model_path):
    print("No trained model found. Not opening website.")
    quit(0)
    
# Modeli yükle
pwd = os.getcwd()
pwd = os.path.join(pwd, model_path)
pwd = os.path.abspath(pwd)
model = TFBertForSequenceClassification.from_pretrained(pwd)

# Tokenizer'ı yükle
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

# Maksimum sekans uzunluğu (model eğitimi sırasında kullandığınız uzunluk)
max_len = 50

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['user_input']
    
    # Metni tokenizer ile vektörleştir
    inputs = tokenizer(user_input, return_tensors="tf", truncation=True, padding="max_length", max_len=max_len)

    # Tahmini yap
    prediction = model(inputs)
    predicted_emotion = np.argmax(prediction.logits, axis=1)[0]

    # Duygu etiketleri
    emotions = ["Mutlu", "Üzgün", "Kızgın", "Neşeli", "Korkmuş", "Şaşkın", "Kıskanç", "Heyecanlı", "İnatçı"]
    predicted_emotion_label = emotions[predicted_emotion]

    return render_template("results.html", prediction=predicted_emotion_label)

if __name__ == '__main__':
    app.run(debug=True)