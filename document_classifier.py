from stop_words import stop_words
from nltk import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# Declare and initialize global constants
MODEL_NAME = 'movie_documents.model'
EPOCHS = 50
VECTOR_SIZE = 300
MIN_COUNT = 5
DM = 0
WORKERS = 4
WINDOW_SIZE = 5


# Class to predict document classes using Doc2vec
class DocumentClassifier:

    # Constructor
    def __init__(self, documents, train_categories, test_categories):

        # Declare and initialize doc2vec model instance variable
        self.model = None

        # Declare and initialize the classifier instance variable
        self.classifier = None

        # Declare and initialize instance variables for test and train vectors
        self.train_vectors = None
        self.test_vectors = None

        # Save train and test categories to self
        self.train_categories = train_categories
        self.test_categories = test_categories

        # Save processed documents to self
        self.documents = self.clean_up_documents(documents)

    # Cleans up the given documents
    @staticmethod
    def clean_up_documents(documents):

        # Tokenize documents
        documents = [word_tokenize(document.lower()) for document in documents]

        # Remove stop words and punctuations from the documents
        documents = [[word for word in doc if (word not in stop_words) and word.isalpha()] for doc in documents]

        # Return the processed documents
        return documents

    # Creates a doc2vec model using processed documents
    def train_doc2vec_model(self):

        # Prepare documents to be used in Doc2Vec model
        tagged_documents = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(self.documents)]

        # Instantiate Doc2Vec model with global constants
        model = Doc2Vec(vector_size=VECTOR_SIZE, dm=DM, workers=WORKERS, window=WINDOW_SIZE, epochs=EPOCHS,
                        min_count=MIN_COUNT)

        # Build the vocabulary
        model.build_vocab(tagged_documents)

        # Train the model
        model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

        # Save the trained model
        model.save(MODEL_NAME)

        # Save test and train vectors to self
        self.save_vectors(model)

        # Save the trained model to self
        self.model = model

    # Loads the given model
    def load_doc2vec_model(self, model_name):

        # Load the model from the given file
        self.model = Doc2Vec.load(model_name)

        # Save test and train vectors to self
        self.save_vectors(self.model)

    # Creates and saves test and train vector lists to self from the given model instance
    def save_vectors(self, model):

        # Get train vectors
        self.train_vectors = [model.docvecs[str(i)] for i in range(len(self.train_categories))]

        # Get test vectors
        self.test_vectors = [model.docvecs[str(i)] for i in range(len(self.train_categories), len(self.documents))]

    # Trains the given classifier and saves to self
    def train_classifier(self, classifier):

        # Assert a model exists in self
        assert self.model, 'You either have to train or load a model.'

        # Train the classifier using train vectors and train categories
        classifier.fit(self.train_vectors, self.train_categories)

        # Save the trained classifier to self
        self.classifier = classifier

    # Classifies the test documents
    def classify_documents(self):

        # Assert a classifier is set
        assert self.classifier, 'You have to train a classifier before predicting.'

        # Get and return predictions from the classifier
        return self.classifier.predict(self.test_vectors)
