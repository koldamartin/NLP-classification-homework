import pathlib
import json
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from train import TextClassifier  # Import your TextClassifier class

# Load the saved model using pathlib
model_path = pathlib.Path('model/text_classifier_model.keras')
model = load_model(model_path)

# Create an instance of TextClassifier
classifier = TextClassifier(model_params={
        'max_sequence_length': 150,
        'lstm_units': [128, 128],
        'dense_units': [64, 8],
        'dropout_rate': 0.25,
        'learning_rate': 0.0001
    },
        w2v_params = {
            'vector_size': 100,
            'window': 4,
            'min_count': 2,
            'workers': 2
    })

# Load and preprocess your test data
test_data_path = pathlib.Path('data/test.jsonl')
test_data = classifier.load_data(test_data_path, test_data_path)[0]  # load_data returns a tuple, we only need the first element
test_sequences, _ = classifier.process_data(test_data, is_training=False)

# Get predictions
y_test_pred = model.predict(test_sequences)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_pred_classes_words = classifier.label_encoder.inverse_transform(y_test_pred_classes)

# Save the input JSONL with [category](cci:4://file:///C:/Users/MartinKolda/PycharmProjects/NLP_Classification_Homework/data/test.jsonl:470:0-473:0) field set to model's predicted value
output_path = pathlib.Path('data/test_output.jsonl')
with output_path.open('w') as f:
    for data, pred in zip(test_data, y_test_pred_classes_words):
        data['category'] = pred
        json.dump(data, f)
        f.write('\n')