# NLP - Text classification model for news category

## Solution
 - Combining the 'headline' and 'short_description' columns into one
 - Tokenizing the text and training a Word2Vec model on the vocabulary corpus
 - Using the Word2Vec model as an input Embedding layer in a Keras sequential model
 - Adding multiple layers into the neural network
 - Model is trained to predict the news category based on the 'category' column

## About the scripts
 - train.py: Input training/evaluation data and parameters and get trained Keras model
 - evaluate.py: Input trained model path and evaluation data and get evaluation results of the model
 - classify.py: Input trained model path and test data and get classification predictions on the test dataset

## Get help for running the scripts
 - run `python train.py --help` (or evaluate.py or classify.py)

## Running the scripts (On Windows using PowerShell)
 - Train model: eg. `python train.py --training_params '{"""epochs""": 20}'` 
 - Evaluate model: `python evaluate.py --model_path path/to/your/model.keras --dev_data path/to/your/dev_data.jsonl`
 - Classify text: `python classify.py --model_path path/to/your/model.keras --test_data path/to/your/test_data.jsonl`
 - check all possible parameters using --help
