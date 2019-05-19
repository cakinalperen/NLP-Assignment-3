import os
import csv
from document_classifier import DocumentClassifier
from embedding_model import EmbeddingModel
from sklearn.linear_model import LogisticRegression

# Define globals for the first task

# Define file names
MOVIE_CSV_FILE = 'tagged_plots_movielens.csv'
PRE_TRAINED_MODEL_FILE = '54movie_documents.model'

# Define globals for the classifier
CLASSIFIER_SOLVER = 'lbfgs'
CLASSIFIER_MAX_ITER = 1000
CLASSIFIER_MULTI_CLASS = 'auto'
CLASSIFIER_TOL = 0.5

# Define globals for the second task

# Define categories to be tested
TEST_CATEGORY_LIST = ['capital-world', 'currency', 'city-in-state', 'family', 'gram1-adjective-to-adverb',
                      'gram2-opposite', 'gram3-comparative', 'gram6-nationality-adjective']

# Define file names
ANALOGY_TEST_FILE = 'word-test.v1.txt'
VECTORS_FILE = 'GoogleNews-vectors-negative300.bin'


# Main method
def main():

    # Test word analogies
    test_word_analogies()

    # Test document classification
    test_document_classification()


# Test the first part of the assignment
def test_word_analogies():

    # Declare and initialize the last category variable
    last_category_read = None

    # List to hold necessary analogy tests
    analogy_tests = []

    # Open the file that contains analogy tests
    with open(ANALOGY_TEST_FILE) as test_file:

        # For each line in the file
        for line in test_file:

            # Remove EOL character from the line
            line = line.rstrip()

            # If the line indicates a category
            if line.startswith(':'):

                # Get the category
                last_category_read = line[2:]

            # If the last category read from file (as a result of the above statement) is in the category list
            elif last_category_read in TEST_CATEGORY_LIST:

                # Append analogy test to the list
                analogy_tests.append(line)

    # Get absolute path of the file from its relative path
    absolute_path = os.path.abspath(VECTORS_FILE)

    # Instantiate the model using the absolute path
    embedding_model = EmbeddingModel(absolute_path)

    # Start evaluating the model
    embedding_model.evaluate_model(analogy_tests)


# Tests the second part of the assignment
def test_document_classification():

    # Open the file and read all lines
    with open(MOVIE_CSV_FILE) as movie_csv:

        # Initialize csv reader using the csv file
        csv_reader = csv.reader(movie_csv, delimiter=',')

        # Skip the first legend line
        next(csv_reader)

        # Get all lines from the file
        lines = [line for line in csv_reader]

    # Get train and test categories
    train_categories = [line[-1] for line in lines[:2000]]
    test_categories = [line[-1] for line in lines[2000:]]

    # Get raw documents from the lines
    documents = [line[2] for line in lines]

    # Instantiate the classifier
    document_classifier = DocumentClassifier(documents, train_categories, test_categories)

    # Train a new model
    # document_classifier.train_doc2vec_model()

    # OR

    # Load a pre-trained model
    document_classifier.load_doc2vec_model(PRE_TRAINED_MODEL_FILE)

    # Instantiate a logistic regression classifier
    logistic_regression_classifier = LogisticRegression(solver=CLASSIFIER_SOLVER, multi_class=CLASSIFIER_MULTI_CLASS,
                                                        max_iter=CLASSIFIER_MAX_ITER, tol=CLASSIFIER_TOL)

    # Train the logistic regression classifier
    document_classifier.train_classifier(logistic_regression_classifier)

    # Get predictions
    predictions = document_classifier.classify_documents()

    # Initialize correct count
    correct_count = 0

    # Count the correctly categorized movie plots
    for i in range(len(test_categories)):

        # If the prediction is correct, increase the count
        if predictions[i] == test_categories[i]:
            correct_count += 1

    # Print the total accuracy
    print('Accuracy: {}%'.format((correct_count / len(test_categories)) * 100))


# Execute the main method
if __name__ == '__main__':
    main()
