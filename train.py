import pathlib
import json
import string
import re
import pickle
import numpy as np
from gensim.models import Word2Vec
from keras.api import Sequential
from keras.api.layers import Dense, LSTM, Dropout, Embedding, Bidirectional, Conv1D, MaxPooling1D
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


class TextClassifier:
    def __init__(self, w2v_params=None, model_params=None):
        # Set default model parameters
        self.model_params = {
            'max_sequence_length': 150,
            'conv_filters': [128, 64],
            'conv_kernel_sizes': [5, 3],
            'lstm_units': [64, 32],
            'dense_units': [64, 32],
            'dropout_rate': [0.5, 0.3],
            'learning_rate': 0.0001
        }

        # Set default Word2Vec parameters
        self.w2v_params = {
            'vector_size': 100,
            'window': 4,
            'min_count': 2,
            'workers': 2
        }

        self.word2vec_model = None
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.embedding_matrix = None
        self.vocab_size = None

        if model_params:
            self.model_params.update(model_params)
        if w2v_params:
            self.w2v_params.update(w2v_params)

    def load_data(self, train_file: str, val_file: str) -> tuple[list, list]:
        """Load and process the JSONL data files"""
        train_file_path = pathlib.Path(train_file)
        val_file_path = pathlib.Path(val_file)

        def load_jsonl(file_path):
            data = []
            with file_path.open('r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data

        train_data = load_jsonl(train_file_path)
        val_data = load_jsonl(val_file_path)

        return train_data, val_data

    def preprocess_text(self, text: str) -> str:
        """Tokenize and prepare text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove non word characters
        text = re.sub(r'[^\w\s]', '', text)
        stop_words = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        stop_words.update(punctuation)
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    def combine_features(self, data: list) -> list:
        """Combine headline and description columns into one"""
        combined_text = [(f"{self.preprocess_text(item['short_description'])} "
                          f"{self.preprocess_text(item['headline'])}") for item in data]
        return combined_text

    def process_data(self, data: list, is_training: bool) -> tuple[np.ndarray, np.ndarray]:
        """Fit/load the tokenizer and Word2Vec model"""
        # Combine features
        combined_texts = self.combine_features(data)

        if is_training:

            # Fit tokenizer
            self.tokenizer.fit_on_texts(combined_texts)

            # Train Word2Vec model
            tokenized_texts = [text.split() for text in combined_texts]
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                **self.w2v_params
            )

            # Save Word2Vec model and tokenizer
            model_path = pathlib.Path('model')
            model_path.mkdir(parents=True, exist_ok=True)
            self.word2vec_model.save(str(model_path / 'word2vec_model'))
            with open(model_path / 'tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)

        else:
            # Load Word2Vec model and tokenizer
            model_path = pathlib.Path('model')
            self.word2vec_model = Word2Vec.load(str(model_path / 'word2vec_model'))
            with open(model_path / 'tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)

        # Create embedding matrix from Word2Vec model
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.embedding_matrix = np.zeros((self.vocab_size, self.w2v_params['vector_size']))
        for word, i in self.tokenizer.word_index.items():
            if word in self.word2vec_model.wv:
                self.embedding_matrix[i] = self.word2vec_model.wv[word]

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(combined_texts)

        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.model_params['max_sequence_length'],
            padding='post',
            truncating='post'
        )

        # Encode labels
        labels = self.label_encoder.fit_transform([item['category'] for item in data])

        return padded_sequences, labels

    def build_model(self):
        """Build the sequential neural network model"""
        model = Sequential([
            # Trained Word2Vec Embedding layer
            Embedding(
                input_dim=classifier.vocab_size,
                output_dim=classifier.w2v_params['vector_size'],
                weights=[classifier.embedding_matrix] if classifier.embedding_matrix is not None else None,
                trainable=False if classifier.embedding_matrix is not None else True
            ),

            # Convolutional layers for feature extraction
            Conv1D(self.model_params['conv_filters'][0], self.model_params['conv_kernel_sizes'][0], activation='relu'),
            MaxPooling1D(2),
            Conv1D(self.model_params['conv_filters'][1], self.model_params['conv_kernel_sizes'][1], activation='relu'),
            MaxPooling1D(2),

            # Bidirectional LSTM for sequence processing
            Bidirectional(LSTM(self.model_params['lstm_units'][0], return_sequences=True)),
            Bidirectional(LSTM(self.model_params['lstm_units'][1], return_sequences=False)),

            # Dense layers for classification
            Dense(self.model_params['dense_units'][0], activation='relu'),
            Dropout(self.model_params['dropout_rate'][0]),
            Dense(self.model_params['dense_units'][1], activation='relu'),
            Dropout(self.model_params['dropout_rate'][1]),
            Dense(len(classifier.label_encoder.classes_), activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=classifier.model_params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, train_file: str, val_file: str, epochs: int, batch_size: int):
        """Train the NN model"""
        # Load data
        train_data, val_data = self.load_data(train_file, val_file)

        # Process training data
        train_sequences, train_labels = self.process_data(train_data, is_training=True)

        # Process validation data
        val_sequences, val_labels = self.process_data(val_data, is_training=False)

        # Build and train model
        model = self.build_model()
        early_stop = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True, verbose=1)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
        history = model.fit(
            train_sequences,
            train_labels,
            validation_data=(val_sequences, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, learning_rate_reduction]
        )

        # Save NN model
        model_path = pathlib.Path('model/text_classifier_model.keras')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        return model, history


if __name__ == '__main__':

    # Initialize with custom parameters
    model_params = {
        'max_sequence_length': 150,
        'conv_filters': [128, 64],
        'conv_kernel_sizes': [5, 3],
        'lstm_units': [64, 32],
        'dense_units': [64, 32],
        'dropout_rate': [0.5, 0.3],
        'learning_rate': 0.0001
    }

    w2v_params = {
        'vector_size': 100,
        'window': 4,
        'min_count': 2,
        'workers': 2
    }

    # Create Classification class and train model
    classifier = TextClassifier(w2v_params=w2v_params, model_params=model_params)
    model, history = classifier.train('data/train.jsonl', 'data/dev.jsonl', epochs=5, batch_size=256)
    print(model.summary())
    print(history.history)
