# P3_SevvalDenizci_1904010028
Project 3

This project aims to train and evaluate the BERT model to perform text classification using Turkish text data. Within the scope of the project, a classification model was developed on a specific dataset using the dbmdz/bert-base-turkish-cased pre-trained model. The project consists of data loading, pre-processing, model training, evaluation and model saving stages.

Libraries Used
Pandas: Used for data loading and processing.
Scikit-learn: It was used to separate the dataset into training and test sets and code the labels.
TensorFlow: Used for model training and evaluation.
Transformers: Used to install and use the BERT model and tokenizer.
Os and Shutil: Used for management of model files.

The dataset used in the project is loaded from a CSV file named dataset.csv. The data is read using the Latin1 character set and the dataset is split into 80% training and 20% testing.

Tokenization of Data
Data needs to be tokenized to be compatible with the BERT model. BertTokenizer was used for this purpose. During the tokenization process, data is truncated and padding is added.

Model Training
The BERT model was trained using TensorFlow. For training the model, Adam was used as the optimizer and SparseCategoricalCrossentropy was used as the loss function. The model was trained with mini-batches of 8 for 3 epochs.

Evaluation of the Model
The trained model was evaluated on test data. Evaluation results are reported as accuracy and 

Saving the Model
After the training and evaluation processes, the model was recorded. The user was asked whether to retrain an existing model and the action was taken accordingly. 
important note : Please don't forget to train the model before prediction.

Conclusion
This project shows that the BERT model can be used effectively for Turkish text classification tasks. High accuracy was achieved during the training and evaluation processes. These results demonstrate the power and flexibility of BERT models in the field of natural language processing.

A virtual environment is used in this project. To run it, you must go to the directory where the project is located in the terminal. Then (Source venv/bin/activate) should be written. Then, the project is run by typing python app.py.
