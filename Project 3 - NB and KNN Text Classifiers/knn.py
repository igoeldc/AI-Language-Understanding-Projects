import numpy as np
from scipy.sparse import csr_matrix
from textclassifier import TextClassifier

class KNN(TextClassifier):
    def __init__(self, train_size=80, k=15):
        super().__init__(train_size)
        self.k = max(min(k, 20), 5)
        self.vectorized_training_labels = None
        self.vectorized_training_matrix = None
        self.word_to_index = None
        
    # Create word to index mapping
    def vocab_index(self):
        self.word_to_index = {}
        for index, word in enumerate(self.vocabulary):
            self.word_to_index[word] = index
    
    def normalize_matrix(self, matrix):
        # Compute L2 norm for each row
        norms = np.sqrt((matrix.multiply(matrix)).sum(axis=1))
        
        # Handle zero rows
        norms[norms == 0] = 1
        
        # Normalize each row
        normalized = matrix.multiply(1/norms)
        
        return normalized
    
    def text_to_sparse_vector(self, text):
        # Convert text to sparse vector format
        words = self.preprocess_text(text)
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if word in self.word_to_index:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create sparse vector
        indices = []
        values = []
        for word, count in word_counts.items():
            indices.append(self.word_to_index[word])
            values.append(count)
        
        return csr_matrix(
            (values, (np.zeros(len(indices)), indices)),
            shape=(1, len(self.vocabulary))
        )
    
    def train(self):
        self.vocab_index()
        
        # Convert training set to sparse matrix format
        rows = []
        self.vectorized_training_labels = []
        
        for text, label in self.training_data:
            rows.append(self.text_to_sparse_vector(text))
            self.vectorized_training_labels.append(label)
        
        # Combine data into a sparse matrix
        self.vectorized_training_matrix = csr_matrix(np.vstack([row.toarray() for row in rows]))
        
        # Normalize the matrix
        self.vectorized_training_matrix = self.normalize_matrix(self.vectorized_training_matrix)
    
    def classify(self, text):
        # Convert input text to sparse vector and normalize
        test_vector = self.text_to_sparse_vector(text)
        test_vector = self.normalize_matrix(test_vector)
        
        # Calculate cosine similarities with training set
        similarities = self.vectorized_training_matrix.dot(test_vector.T).toarray().flatten()
        
        # Get indices of k nearest neighbors
        nearest_indices = np.argpartition(similarities, -self.k)[-self.k:]
        
        # Get their labels and similarities
        k_nearest = []
        for idx in nearest_indices:
            similarity = similarities[idx]
            label = self.vectorized_training_labels[idx]
            k_nearest.append((similarity, label))
        
        # Sort by similarity
        k_nearest.sort(reverse=True)
        
        # Count votes for each label
        vote_counts = {}
        for _, label in k_nearest:
            if label not in vote_counts:
                vote_counts[label] = 0
            vote_counts[label] += 1
        
        # Find label with most votes
        predicted_class = None
        max_votes = -1
        for label, votes in vote_counts.items():
            if votes > max_votes:
                max_votes = votes
                predicted_class = label
        
        return predicted_class