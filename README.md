# NLP - Text classification model for news category
 - Train a model for multi-class classification of 42 unique news categories

### Tech stack
 - Keras API
 - Word2Vec

### Solution
 - Combining the 'headline' and 'short_description' columns into one
 - Label encoding the 'category' column
 - Tokenizing the text and training a Word2Vec model on the vocabulary corpus
 - Using the Word2Vec model as an input Embedding layer in a Keras sequential model
 - Adding multiple layers into the neural network
 - Model is trained to predict the news category based on the 'category' column

### Results
 - Very low accuracy, only 20%
 - Techniques to deal with imbalanced dataset: Resampling, SMOTE, Class weighting

### About the scripts
 - train.py: Input training/evaluation data and parameters and get trained Keras model 
    - **output** trained keras model, word2vec model and tokenizer stored in model folder
 - evaluate.py: Input trained model path and evaluation data and get evaluation results of the model
    - **output** evaluation results printed into console when the script is run 
 - classify.py: Input trained model path and test data and get classification predictions on the test dataset
    - **output** test_output.jsonl file with predictions saved into data folder 

### Get help to see parameters for running the scripts
 - run `python train.py --help` (or evaluate.py or classify.py)

### Running the scripts (On Windows using PowerShell)
 - Train model: eg. `python train.py --training_params '{"""epochs""": 20}'` 
 - Evaluate model: `python evaluate.py --model_path path/to/your/model.keras --dev_data path/to/your/dev_data.jsonl`
 - Classify text: `python classify.py --model_path path/to/your/model.keras --test_data path/to/your/test_data.jsonl`
 - check all possible parameters using --help
