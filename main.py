import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import os
import shutil

# Load and process data
def load_and_process_data(filepath):
    df = pd.read_csv(filepath, encoding="latin1")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    label_encoder = LabelEncoder()
    train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
    test_df['Label'] = label_encoder.fit_transform(test_df['Label'])
    return train_df, test_df

# Tokenizer and encodings
def tokenize_data(df, tokenizer):
    texts = df['Sentence'].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='tf')
    labels = df['Label'].tolist()
    return encodings, labels

# Train the model
def train_model(train_encodings, train_labels):
    model = TFBertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
    train_dataset = train_dataset.shuffle(len(train_labels)).batch(8)

    model.fit(train_dataset, epochs=3, batch_size=8)
    return model

# Evaluate the model
def evaluate_model(model, test_encodings, test_labels):
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels)).batch(8)
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

def main():
    # File path
    filepath = "dataset.csv"
    model_path = "ai_model"
    # Load and process data
    train_df, test_df = load_and_process_data(filepath)

    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

    # Tokenize training data
    train_encodings, train_labels = tokenize_data(train_df, tokenizer)

    # Tokenize test data
    test_encodings, test_labels = tokenize_data(test_df, tokenizer)
    
    model = None
    if os.path.exists(model_path):
        option = input("Model is already trained. Do you want to retrain it? (Y/N): ")
        if option.lower() == 'y':
            shutil.rmtree(model_path)
            model = train_model(train_encodings, train_labels)
        
    else:
        model = train_model(train_encodings, train_labels)

    if model is None:
        return 0
    
    # Evaluate the model
    evaluate_model(model, test_encodings, test_labels)

    # Save the model (optional)
    model.save_pretrained('ai_model')

if __name__ == '__main__':
    main()