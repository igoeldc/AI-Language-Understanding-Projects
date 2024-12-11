import sys
from knn import KNN
from naivebayes import NaiveBayes

# Calculate classification metrics
def calculate_metrics(true_labels, predicted_labels):
    # Get unique labels and pick first as positive class
    unique_labels = list(set(true_labels))
    unique_labels.sort()
    positive_class = unique_labels[0]
    
    tp = tn = fp = fn = 0
    
    # Add 1 for depending if it is TP, TN, FP, or FN
    for true, pred in zip(true_labels, predicted_labels):
        if true == pred == positive_class:
            tp += 1
        elif true == pred != positive_class:
            tn += 1
        elif true != positive_class and pred == positive_class:
            fp += 1
        elif true == positive_class and pred != positive_class:
            fn += 1
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f_score = 2 * (precision * sensitivity) / (precision + sensitivity) \
              if (precision + sensitivity) > 0 else 0
    
    return {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'accuracy': accuracy,
        'f_score': f_score
    }

# Relabel True/False to Male/Female
def mf(tf):
    if tf.lower() == "true":
        return "Male"
    else:
        return "Female"

def main():
    # Read command line arguments
    if len(sys.argv) == 3:
        if int(sys.argv[1]) in [0, 1]:
            algo = int(sys.argv[1])
        else:
            algo = 0

        if (50 <= int(sys.argv[2]) <= 80):
            train_size = int(sys.argv[2])
        else:
            train_size = 80        
    else:
        algo = 0
        train_size = 80
    
    # Choose appropriate classifier
    if algo == 0:
        classifier = NaiveBayes(train_size)
        algo_name = "Naive Bayes"
    else:
        classifier = KNN(train_size)
        algo_name = "K Nearest Neighbors"
    
    print(f"Goel, Ishaan, A20507191 solution:")
    print(f"Training set size: {train_size} %")
    print(f"Classifier type: {algo_name}\n")
    
    # Load and preprocess data
    print("Training classifier...")
    classifier.load_data('tweet_dataset/train.csv','tweet_dataset/valid.csv','tweet_dataset/test.csv')
    classifier.train()
    
    # Test classifier
    print("Testing classifier...")
    true_labels = [label for _, label in classifier.test_data]
    predicted_labels = []
    for text, _ in classifier.test_data:
        if algo == 0:
            pred, _ = classifier.classify(text)
        else:
            pred = classifier.classify(text)
        predicted_labels.append(pred)
    
    # Calculate and display metrics
    metrics = calculate_metrics(true_labels, predicted_labels)
    print("\nTest results / metrics:")
    print(f"Number of true positives: {metrics['true_positives']}")
    print(f"Number of true negatives: {metrics['true_negatives']}")
    print(f"Number of false positives: {metrics['false_positives']}")
    print(f"Number of false negatives: {metrics['false_negatives']}")
    print(f"Sensitivity (recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Negative predictive value: {metrics['npv']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F-score: {metrics['f_score']:.4f}\n")
    
    # Manual classification
    while True:
        print("Enter your sentence/document:")
        sentence = input("\nSentence/document S: ")
        
        if algo == 0:
            predicted_class, probabilities = classifier.classify(sentence)
            print(f"\nwas classified as {mf(predicted_class)} ({predicted_class}).")
            for class_label, prob in probabilities.items():
                print(f"P({mf(class_label)} ({class_label}) | S) = {prob:.4f}")
        else:
            predicted_class = classifier.classify(sentence)
            print(f"\nwas classified as {mf(predicted_class)} ({predicted_class}).")
        
        choice = input("\nDo you want to enter another sentence [Y/N]? ").strip().upper()
        if choice != 'Y':
            break

if __name__ == "__main__":
    main()