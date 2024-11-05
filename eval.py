from pathlib import Path
import keras
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix, accuracy_score

from train import TextClassifier


def evaluate(nn_model, classifier, eval_sequences, eval_labels) -> None:
    # Get predictions
    y_eval_pred = nn_model.predict(eval_sequences)
    y_eval_pred_classes = np.argmax(y_eval_pred, axis=1)
    y_eval_pred_classes_words = classifier.label_encoder.inverse_transform(y_eval_pred_classes)
    # Convert true labels to words
    y_eval_true = classifier.label_encoder.inverse_transform(eval_labels)

    # Calculate confusion matrix
    cm = multilabel_confusion_matrix(y_eval_true, y_eval_pred_classes_words,
                                     labels=["POLITICS", "ENTERTAINMENT", "WELLNESS"])

    # Calculate precision, recall, f1, _ for Politics and Entertainment categories
    precision, recall, f1, _ = precision_recall_fscore_support(y_eval_true,
                                                               y_eval_pred_classes_words,
                                                               average=None,
                                                               labels=['POLITICS', 'ENTERTAINMENT', 'WELLNESS'])

    # Calculate accuracy for all data
    accuracy = accuracy_score(y_eval_true, y_eval_pred_classes_words)

    # Printing results
    print(f"Confusion Matrix: \n {cm}\n")
    print(f"Accuracy score:' {accuracy}\n")
    print('Precision for Politics:', precision[0])
    print('Precision for Entertainment:', precision[1])
    print('Precision for Wellness:', precision[2])
    print('Recall for Politics:', recall[0])
    print('Recall for Entertainment:', recall[1])
    print('Recall for Wellness:', recall[2])
    print('F1 Score for Politics:', f1[0])
    print('F1 Score for Entertainment:', f1[1])
    print('F1 Score for Wellness:', f1[2])


if __name__ == "__main__":

    model_path = Path('model/text_classifier_model.keras')
    model = keras.models.load_model(model_path)
    classifier = TextClassifier()
    train_data, eval_data = classifier.load_data('data/train.jsonl', 'data/dev.jsonl')
    X, y = classifier.process_data(eval_data, is_training=False)
    evaluate(model, classifier, X, y)
