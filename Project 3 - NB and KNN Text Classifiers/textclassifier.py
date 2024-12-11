import csv, re

# General text data methods
class TextClassifier:
    def __init__(self, train_size=80):
        self.train_size = min(max(train_size, 50), 80) # Training size has to be between 50 - 80
        self.vocabulary = set() # Create vocabulary
        self.training_data = [] # Text data
        self.test_data = [] # Labels

    # Clean and tokenize text
    def preprocess_text(self, text):
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Split into words and remove empty strings
        return [word for word in text.split() if word]

    # Load and split data from multiple CSV files (train, valid, test) into training and test sets
    def load_data(self, *filepaths): # Unpacks a list of multiple files to generate the dataset
        data = []
        
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                
                # Skip header row
                next(csv_reader)
                
                # Convert CSV rows to (text, label) format, first two columns are not relevant
                data.extend([(row[2], row[3]) for row in csv_reader])
            
            # Calculate and split indices
            total_samples = len(data)
            train_end = int(total_samples * (self.train_size / 100))
            test_start = int(total_samples * 0.8)

            self.training_data = data[:train_end]
            self.test_data = data[test_start:]

            # Build vocabulary from training data
            for text, _ in self.training_data:
                words = self.preprocess_text(text)
                self.vocabulary.update(words)