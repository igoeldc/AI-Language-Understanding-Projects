import nltk
from nltk import FreqDist
import matplotlib.pyplot as plt

# download the corpora and stopwords from nltk
# nltk.download('brown')
# nltk.download('reuters')
# nltk.download('stopwords')

# save the corpora and stopwords
bcorpus = nltk.corpus.brown
rcorpus = nltk.corpus.reuters
stopwords = nltk.corpus.stopwords.words('english')

# process the corpora 
def process_corpus(corpus):
    # take the unique words (lowercased) from the corpus that are not stop words
    words = [w for w in corpus.words() if w.lower not in stopwords]
    
    # create the distribution from the words
    fdist = FreqDist(words)
    
    return fdist

bdist = process_corpus(bcorpus)
rdist = process_corpus(rcorpus)

# create frequency distributions for both corpora
topn = 10

print(bdist.most_common(topn))
print(rdist.most_common(topn))

# log log plots
def distrubution_plot(fdist, corpus, nrank):
    # find the top n ranked words
    top_n = fdist.most_common(nrank)
    
    # extract ranks and frequencies
    ranks = list(range(1, nrank+1))
    freqs = [freq for word, freq in top_n]
    
    # create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ranks, freqs, alpha=0.5)
    
    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set labels and title
    ax.set_title(f'Log-Log Plot of Word Frequencies in {corpus} Corpus')
    ax.set_xlabel('Log(Rank)')
    ax.set_ylabel('Log(Frequency)')
    
    # Display the plot
    plt.show()

nrank = 1000
# distrubution_plot(bdist, 'Brown', nrank)
# distrubution_plot(rdist, 'Reuters', nrank)

# unigram probabilities
def unigram_prob(word, fdist, total_words):
    count = fdist[word]
    prob = count / total_words
    return count, prob

def unigram_calc(corpus, fdist):
    total_words = sum(fdist.values())
    
    tech_count, tech_prob = unigram_prob(technical_word, fdist, total_words)
    non_tech_count, non_tech_prob = unigram_prob(non_technical_word, fdist, total_words)
    
    print(f"\n{corpus} Corpus:")
    print(f"'{technical_word}': count = {tech_count}, probability = {tech_prob}")
    print(f"'{non_technical_word}': count = {non_tech_count}, probability = {non_tech_prob}")

technical_word = "neutron"
non_technical_word = "person"

unigram_calc("Brown", bdist)
unigram_calc("Reuters", rdist)