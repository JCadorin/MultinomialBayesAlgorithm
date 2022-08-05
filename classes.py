import numpy as np

class MNNaiveBayes:
    def __init__(self, alpha=0.5):
        # alpha = corrective variable
        self.alpha = alpha
        # classification split in two categories (news e pm)
        self.cat0_count = 0
        self.cat1_count = 0
        self.total_count = self.cat0_count + self.cat1_count
        self.cat_0_prior = 0
        self.cat_1_prior = 0
        self.cat_0_prior, self.cat_1_prior
        self.word_probs = []
        self.vocab = []

    def clean(self, document):
        # turn text in a list of words, with no special character
        doc = document.lower()
        stop_chars = '''0123456789!()-[]{};:'"\,<>./?@#$%^&*_~'''
        tokens = ""
        for char in doc:
            if char not in stop_chars:
                tokens += char

        # returning a list of characters
        return tokens.split()

    def count_words(self, X, y):
        # X is the variable with texts
        # y is the classification target, in this case is 0 e 1
        # returning a dictionary of each word and the counts of it in each category
        counts = {}
        for document, category in zip(X, y):
            for token in self.clean(document):
                # creating a new empty dictionary for the word, in case it's not already classified
                if token not in counts:
                    counts[token] = [0, 0]
                # add the count
                counts[token][category] += 1
        # after loop of each category in each word, we return the dictionary
        return counts

    def prior_prob(self, counts):
        # going through each word and counts the value of each category
        cat0_word_count = 0
        cat1_word_count = 0
        for word, (cat0_count, cat1_count) in counts.items():
            cat0_word_count += cat0_count
            cat1_word_count += cat1_count

        # saving the information
        self.cat0_count = cat0_word_count
        self.cat1_count = cat1_word_count
        self.total_count = self.cat0_count + self.cat1_count

        # Finding probability a priori with the division of each word by the total
        cat_0_prior = cat0_word_count / self.total_count
        cat_1_prior = cat1_word_count / self.total_count
        return cat_0_prior, cat_1_prior

    def word_probabilities(self, counts):
        # turning the variable 'word_counts' in a triple list contains the word, prior 0 and prior 1
        self.vocab = [word for word, (cat0, cat1) in counts.items()]
        # adding alpha to avoid none values if the word was not previously classified
        return [(word,
                 (cat0 + self.alpha) / (self.cat0_count + 2 * self.alpha),
                 (cat1 + self.alpha) / (self.cat1_count + 2 * self.alpha))
                for word, (cat0, cat1) in counts.items()]

    def fit(self, X, y):
        # Fit Function
        # Establishing the probability of each word in the train with our functions
        counts = self.count_words(X, y)
        self.cat_0_prior, self.cat_1_prior = self.prior_prob(counts)
        self.word_probs = self.word_probabilities(counts)

    def predict(self, test_corpus):
        # Split the text into tokens,
        # For each category: calculate the probability of each word in that cat
        # find the product of all of them and the prior prob of that cat
        y_pred = []
        for document in test_corpus:
            # Initializing a prediction of probability in each document
            log_prob_cat0 = 0.0
            log_prob_cat1 = 0.0
            # Cleaning text
            tokens = self.clean(document)
            # Going through the train data and add the probability we found
            for word, prob_cat0, prob_cat1 in self.word_probs:
                if word in tokens:
                    # Add log to avoid poblems with complex numbers close to 0
                    log_prob_cat0 += np.log(prob_cat0)
                    log_prob_cat1 += np.log(prob_cat1)
                    # getting each predict by category
            cat_0_pred = self.cat_0_prior * np.exp(log_prob_cat0)
            cat_1_pred = self.cat_1_prior * np.exp(log_prob_cat1)
            # adding the category with more score
            if cat_0_pred >= cat_1_pred:
                y_pred.append([document, "News"])
            else:
                y_pred.append([document, "PM"])
        return y_pred
