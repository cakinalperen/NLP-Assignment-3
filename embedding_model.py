from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import numpy as np
from most_similar_thread import MostSimilarThread

# Declare and initialize globals
VECTOR_LIMIT = 2000000  # Can be None if the machine has enough memory
THREAD_COUNT = 4


# Class that defines a word embedding model
class EmbeddingModel:

    # Constructor
    def __init__(self, binary_vector_path):

        # Load vectors from the given input file and save them to self
        self.word_vectors = KeyedVectors.load_word2vec_format(datapath(binary_vector_path), binary=True,
                                                              limit=VECTOR_LIMIT)
        #  Normalize vectors (L2)
        #  v  --normalize--> v / ||v||
        #  Don't replace
        self.word_vectors.init_sims()

    # Helper method to use gensim's most similar. This method is used for accuracy and execution time comparison
    # Input words must be given in order: a,b,c for this helper to work
    @staticmethod
    def get_most_similar_gensim_helper(v1, input_words, word_vectors):
        return word_vectors.most_similar(positive=[input_words[1], input_words[2]],
                                         negative=[input_words[0]], topn=1)[0][0]

    # Returns the most similar word for given vector v1
    # input_words is used to filter out input words
    # word_vectors should contain Word2VecKeyedVectors instance
    @staticmethod
    def get_most_similar_fast(v1, input_words, word_vectors):

        # Thanks to L2 normalization, vector A is now stored in syn0norm as A / ||A||
        # and vector B is stored as B / ||B||
        # Cosine similarity formula is A . B / ||A|| . ||B||
        # Calculating normalizations beforehand allows the program to calculate all cosine similarities just using
        # numpy.dot.
        #
        # This is a lot faster than trying to calculate ||A|| and ||B|| for each word vector one by one in every step
        # of the process.

        # Calculate cosine similarities using dot product
        similarities = np.dot(word_vectors.syn0norm, v1)

        # Get the largest 4 elements
        # 4 because most similar ones may be input words. Since 3 words are supplied as input,
        # getting the largest 4 elements will suffice in any case
        four_largest = np.argpartition(similarities, -4)[-4:]

        # Declare and initialize variables for the position of the maximum value and the maximum value
        max_value = -float('inf')
        position_max = 0

        # For each position in the 4 largest list
        for i in range(len(four_largest)):

            # If the position does not contain a word index that corresponds to an input word
            if word_vectors.index2word[four_largest[i]] not in input_words:

                # If the position contains a word index that corresponds to a word vector which has a similarity to the
                # target vector larger than the value stored in max_value
                if similarities[four_largest[i]] > max_value:

                    # Assign maximum value and position newer values
                    max_value = similarities[four_largest[i]]
                    position_max = i

        # Return the corresponding word of the word index selected using the position_max
        return word_vectors.index2word[four_largest[position_max]]

    # Method that performs the given analogy tests
    def evaluate_model(self, analogy_tests):

        # Split the analogy list into N parts (where N is the number of threads)
        analogies_split = np.array_split(analogy_tests, THREAD_COUNT)

        # Declare and initialize the thread list
        threads = []

        # For each element in the split list
        for analogy_list in analogies_split:

            # Instantiate a thread
            # Method to find most similar word should be set here
            thread = MostSimilarThread(analogy_list, self.word_vectors, self.get_most_similar_fast)

            # Append the thread to the thread list
            threads.append(thread)

            # Start the thread
            thread.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Declare and initialize counts
        correct = 0
        unknown = 0

        # For each thread
        for t in threads:

            # Get the number of correct answers
            correct += t.correct_count

            # Get the number of tests that failed due to unknown words
            unknown += t.unknown_count

        # Print statistics
        print('Accuracy:', correct / len(analogy_tests))
        print('Number of correct words predicted:', correct)
        print('Number of tests that failed due to unknown words:', unknown)
        print('Vector limit:', VECTOR_LIMIT)
        print('Thread Count:', THREAD_COUNT)
