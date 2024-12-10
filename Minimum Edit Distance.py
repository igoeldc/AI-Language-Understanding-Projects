import sys
import pandas as pd
import re
import time

# Load words.csv and EDweights.csv into dataframes
def load_data(words_file, weights_file):
    try:
        words = pd.read_csv(words_file, header=None)[0].tolist()
        weights = pd.read_csv(weights_file, index_col=0)
        return words, weights
    except FileNotFoundError:
        print("Error: At least one data file not found in the current directory.")
        sys.exit(1)

def clean_string(input_string):
    # Convert to lowercase
    input_string = input_string.lower()
    
    # Use regex to remove anything that is not a-z
    cleaned_string = re.sub(r'[^a-z]', '', input_string)
    
    return cleaned_string

# Minimum Edit Distance function
def minimum_edit_distance(string1, string2, use_weights, weights):
    string1 = clean_string(string1)
    string2 = clean_string(str(string2))
    m, n = len(string1), len(string2)
    
    # Initialize memo
    memo = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        memo[i][0] = i
    for j in range(n + 1):
        memo[0][j] = j
    
    # Run algorithm on the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if string1[i-1] == string2[j-1]:
                memo[i][j] = memo[i-1][j-1]
            else:
                deletion = memo[i-1][j] + 1
                insertion = memo[i][j-1] + 1
                if not use_weights:
                    substitution = memo[i-1][j-1] + 2
                else:
                    substitution = memo[i-1][j-1] + weights.loc[string1[i-1], string2[j-1]]
                memo[i][j] = min(deletion, insertion, substitution)

    return memo[m][n]

def find_suggestions(misspelled_word, word_list, use_weights, weights):
    suggestions = []
    min_distance = float('inf')

    for word in word_list:
        distance = minimum_edit_distance(misspelled_word, word, use_weights, weights)
        if distance < min_distance: # if there is a word with a smaller distance, then make a new list
            min_distance = distance
            suggestions = [word]
        elif distance == min_distance: # otherwise if it matches the current distance, add it to the list
            suggestions.append(word)

    return sorted(suggestions)

def main():
    if len(sys.argv) != 3: # makes sure that the user enters the correct number of arguments
        print("Error: Incorrect number of arguments.")
        print("Usage: python CS481_P01_A20507191.py WEIGHTS MISSPELLED_WORD")
        sys.exit(1)

    weights_arg = int(sys.argv[1])
    misspelled_word = sys.argv[2].lower()

    use_weights = weights_arg == 1 # if the user enters anything that is out of range, it will default to 0

    words, weights = load_data("words.csv", "EDweights.csv")

    start_time = time.time()
    suggestions = find_suggestions(misspelled_word, words, use_weights, weights)
    end_time = time.time()

    processing_time = end_time - start_time

    print("Goel, Ishaan, A20507191 solution:")
    print(f"Weights: {weights_arg}")
    print(f"Misspelled word: {misspelled_word}")
    print(f"\nProcessing time: {processing_time:.6f} seconds")
    print("\nMinimum edit distance suggested word(s):")
    for word in suggestions:
        print(word)

if __name__ == "__main__":
    main()