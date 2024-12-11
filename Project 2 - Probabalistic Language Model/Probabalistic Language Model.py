import nltk
from nltk import bigrams, ConditionalFreqDist

# download the corpus and stopwords from nltk
# nltk.download('brown')
# nltk.download('stopwords')

# save the corpus and stopwords
corpus = nltk.corpus.brown

# get top three most likely next words from bigram probabilities
def top_three_words(word, bigram_prob):
    if word not in bigram_prob:
        return []
    
    # get all words the subsequent words
    next_words = bigram_prob[word].items()
    
    # Sort by probability (frequency) and get top 3
    sorted_words = sorted(next_words, key=lambda x: x[1], reverse=True)[:3]
    
    total_words = sum(bigram_prob[word].values())
    
    return [(w, c / total_words) for w, c in sorted_words]

words = [w.lower() for w in corpus.words() if w.lower()]
bigram_prob = ConditionalFreqDist(bigrams(words))

sentence = []
while True:
    # initial word
    if not sentence:
        word = input("Enter the first word (or 'QUIT' to exit): ").lower()
    else:
        word = sentence[-1]
    
    if word == 'quit':
        break
    
    if word not in bigram_prob:
        print(f"'{word}' not found in the corpus.")
        choice = input("Enter '1' to try again or '2' to QUIT: ")
        if choice == '2': # if user enters anything other than 1 or 2, 1 will be selected automatically
            break
        continue
    
    # appends word if sentence is not empty
    if not sentence:
        sentence.append(word)
    
    # calculates top three most likely next words
    top_three = top_three_words(word, bigram_prob)
    
    # gives user choice for next word or end sentence
    print(f"\nCurrent sentence: {' '.join(sentence)}")
    print("\nWhich word should follow:")
    for i, (next_word, prob) in enumerate(top_three, 1):
        print(f"{i}. {next_word} P({word} {next_word}) = {prob:.4f}")
    print("4. QUIT")
    
    choice = input("Enter your choice (1-4): ")
    if choice == '4':
        break
    elif choice in ['1', '2', '3']:
        sentence.append(top_three[int(choice) - 1][0])
    else:
        print("Invalid choice. Assuming choice 1.") # input validation, defaults to 1
        sentence.append(top_three[0][0])

print(f"\nFinal sentence: {' '.join(sentence)}")
