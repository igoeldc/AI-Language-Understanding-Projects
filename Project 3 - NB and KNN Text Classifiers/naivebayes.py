import math
from textclassifier import TextClassifier

class NaiveBayes(TextClassifier):
    def __init__(self, train_size=80):
        super().__init__(train_size)
        self.class_probs = {}
        self.word_probs = {}
        self.classes = set()
        
    def train(self):
        # Count class occurrences
        class_counts = {}
        for _, label in self.training_data:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        # Count number of documents    
        total_docs = len(self.training_data)
        
        # Calculate class probabilities
        self.classes = set(class_counts.keys())
        for class_label in self.classes:
            self.class_probs[class_label] = (class_counts[class_label] + 1) / (total_docs + len(self.classes))
        
        # Count word occurrences per class
        word_counts = {}
        total_words = {}
        
        for class_label in self.classes:
            word_counts[class_label] = {}
            total_words[class_label] = 0
        
        for text, label in self.training_data:
            words = self.preprocess_text(text)
            for word in words:
                if word not in word_counts[label]:
                    word_counts[label][word] = 0
                word_counts[label][word] += 1
                total_words[label] += 1
        
        # Calculate word probabilities with add-1 smoothing
        vocab_size = len(self.vocabulary)
        self.word_probs = {}
        for class_label in self.classes:
            self.word_probs[class_label] = {}
            for word in self.vocabulary:
                count = word_counts[class_label].get(word, 0)
                self.word_probs[class_label][word] = (count + 1) / (total_words[class_label] + vocab_size)
    
    def classify(self, text):
        # Classify text using log probabilities.
        words = self.preprocess_text(text)
        scores = {}
        vocab_probs = self.word_probs  # Local reference for efficiency

        # Calculate log-probability scores
        for class_label in self.classes:
            log_prob = math.log(self.class_probs[class_label])
            word_probs = vocab_probs[class_label]
            for word in words:
                if word in self.vocabulary:
                    log_prob += math.log(word_probs.get(word, 0))
            scores[class_label] = log_prob

        # Find the maximum score and normalize probabilities
        max_label, max_log_score = max(scores.items(), key=lambda item: item[1])
        total = sum(math.exp(score - max_log_score) for score in scores.values())

        # Normalize the scores
        normalized_scores = {label: math.exp(score - max_log_score) / total for label, score in scores.items()}
        return max_label, normalized_scores