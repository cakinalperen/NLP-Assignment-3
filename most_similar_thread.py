import threading
from numpy import linalg


# Thread to perform the word analogy task with given work load
class MostSimilarThread(threading.Thread):

    # Constructor
    def __init__(self, work_load, word_vectors, most_similar_method):

        # Call the constructor of super
        threading.Thread.__init__(self)

        # Save the given word vectors
        self.word_vectors = word_vectors

        # Save the given work load
        self.work_load = work_load

        # Declare and initialize counts for this thread
        self.correct_count = 0
        self.unknown_count = 0

        # Save the given method to find most similar word for a given vector
        self.most_similar_method = most_similar_method

    # Run method
    def run(self):

        # For each analogy test in the work load
        for analogy_test in self.work_load:

            # Get words a, b, c, d
            a, b, c, d = analogy_test.split()

            # If the word exists in the model
            try:

                # Get word vectors for words a, b, c, d
                va, vb, vc = self.word_vectors[a], self.word_vectors[b], self.word_vectors[c]

                # Calculate the target vector according to given formula
                target = vb - va + vc

                # Normalize the target
                target_length = linalg.norm(target)

                target = target / target_length

                # Get the most similar word to the target vector
                prediction = self.most_similar_method(target, [a, b, c], self.word_vectors)

                # If the most similar word is d
                if d == prediction:

                    # Increase correct count
                    self.correct_count += 1

            # Else
            except KeyError:
                self.unknown_count += 1
