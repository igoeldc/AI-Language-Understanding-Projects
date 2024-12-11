import nltk
from nltk import bigrams, ConditionalFreqDist

# download and save the corpus from nltk
# nltk.download('brown')
corpus = nltk.corpus.brown

# calculate sentence probability
def calc_sent_prob(sentence):
    # lowercase and tokenize sentence
    tokenize = sentence.lower().split()
    
    # sentence bigrams
    sentence_bigrams = list(bigrams(tokenize))
    
    # all bigram probabilities
    bigram_prob = ConditionalFreqDist((w1, w2) for w1, w2 in bigrams(corpus.words()))
    
    # calculate sentence probability
    sentence_prob = 0.25  # starting probability
    bigram_probs = []
    
    for w1, w2 in sentence_bigrams:
        bigram_count = bigram_prob[w1][w2]
        w1_count = bigram_prob[w1].N()
        
        # w1 w2 bigram probability
        if w1_count == 0:
            prob = 0
        else:
            prob = bigram_count / w1_count
        
        bigram_probs.append((f"{w1} {w2}", prob))
        sentence_prob *= prob
    
    # end sentence probability
    sentence_prob *= 0.25
    
    return sentence_prob, bigram_probs

sentence = input("Enter a sentence: ")
probability, bigram_probs = calc_sent_prob(sentence)

# display results
print(f"\nSentence: {sentence}")
print("\nBigram probabilities:")
print(f"<s> {sentence[0]}: 0.25")
for bigram, prob in bigram_probs:
    print(f"{bigram}: {prob}")
print(f"{sentence[-1]} <s>: 0.25")
print(f"\nFinal sentence probability: {probability}")