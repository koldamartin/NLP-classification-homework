from pathlib import Path
import json
import numpy as np
import keras
import argparse

from train import TextClassifier


def predict_classes(nn_model, classifier, test_data: list, test_sequences: np.ndarray) -> None:
    # Get predictions
    y_test_pred = nn_model.predict(test_sequences)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_pred_classes_words = classifier.label_encoder.inverse_transform(y_test_pred_classes)

    # Save predictions into test_output.jsonl
    save_path = Path('data/test_output.jsonl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w') as f:
        for data, pred in zip(test_data, y_test_pred_classes_words):
            data['category'] = pred
            json.dump(data, f)
            f.write('\n')
    print(f"Predictions saved to {save_path}")


def main(args):
    model_path = Path(args.model_path)
    model = keras.models.load_model(model_path)
    classifier = TextClassifier()
    test_data, _ = classifier.load_data(args.test_data, args.test_data)
    X, y = classifier.process_data(test_data, is_training=False)
    predict_classes(model, classifier, test_data, X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify text using a trained model")
    parser.add_argument("--model_path", type=str, default='model/text_classifier_model.keras',
                        help="Path to the trained model")
    parser.add_argument("--test_data", type=str, default='data/test.jsonl',
                        help="Path to the test data")
    args = parser.parse_args()

    main(args)
