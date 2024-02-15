import nltk
import requests
from nltk.stem import PorterStemmer
import nltk.downloader
nltk.download('punkt')
nltk.download('wordnet')
import string
from collections import defaultdict
import re
from nltk.stem import WordNetLemmatizer
import math



# Read and append fiction text
with open("wizardOfOz.txt", "r") as file:
    WizardOfOzText = file.read()
with open("peterPan.txt", "r") as file:
    peterPanText = file.read()
with open("aliceInWonderland.txt", "r") as file:
    AliceInWonderlandText = file.read()
fictionText = WizardOfOzText + " " + peterPanText + " " + AliceInWonderlandText

# Read and append nonfiction text
with open("benFranklin.txt", "r") as file:
    benFranklinText = file.read()
with open("hellenKeller.txt", "r") as file:
    hellenKellerText = file.read()
with open("napoleon.txt", "r") as file:
    napoleonText = file.read()
nonFictionText = benFranklinText + " " + hellenKellerText + " " + napoleonText


# Get list of stopWords NLTK's Gist (Code provided in issues)
stopwords_list = requests.get(
    "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopWords = set(stopwords_list.decode().splitlines())


# Normalize text (remove stop words, punctuation, and lowercase with an option of stemming )
def normalizeText(text, lemma=False):
    normalizedText = ""
    tokenCount = 0
    tokens = defaultdict(int)
    lemmatizer = WordNetLemmatizer() if lemma else None

    for word in nltk.word_tokenize(text):
        word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
        word = word.lower()
        if not word.strip() or word in stopWords:  # Skip empty strings and stopwords
            continue

        if lemmatizer:
            word = lemmatizer.lemmatize(word)

        tokens[word] += 1
        tokenCount += 1
        # Add to a normalized for topic modeling.
        normalizedText += word + " "


    # Sort by frequency with help from GPT
    sortedTokens = dict(sorted(tokens.items(), key=lambda item: item[1], reverse=True))
    # Output total count and Bag of words tokens
    return [tokenCount, sortedTokens, normalizedText]


# Calculate probability with laplace smoothing
def calculateProbabilities(tokenCount, sortedTokens):
    probabilities = {}
    for word, count in sortedTokens.items():
        # In denominator, add length of vocabulary to account for adding one to each vocab
        probabilities[word] = (count + 1) / (tokenCount + len(sortedTokens)) 
    return probabilities


# Computer the log likelihood ratio using the two sets of probabilities
def computeLLR(fictionProb, nonFictionProb):
    llr = {}
    allWords = set(fictionProb.keys()).union(nonFictionProb.keys())
    for word in allWords:
        pwc = fictionProb.get(word, 1 / (fictionTokenCount + len(allWords)))  # Probability in fiction
        pwcCo = nonFictionProb.get(word, 1 / (nonFictionTokenCount + len(allWords)))  # Probability in non-fiction
        llr[word] = math.log(pwc / pwcCo)  # LLR computation
    return llr


# MAIN FUNCTION
lemmatize = input('do you want to lemmatize words? (y/n)')
lemmatize = True if lemmatize.lower() == "y" else False


# Normalize text
fictionTokenCount, fictionBOW, normalizedFiction = normalizeText(fictionText, lemmatize)
nonFictionTokenCount, nonFictionBOW, normalizedNonFiction = normalizeText(nonFictionText, lemmatize)
# Computer probs with laplace smoothing
fictionProbabilities = calculateProbabilities(fictionTokenCount, fictionBOW)
nonFictionProbabilities = calculateProbabilities(nonFictionTokenCount, nonFictionBOW)
# Find LLR for each word in the combined vocabulary
llrValues = computeLLR(fictionProbabilities, nonFictionProbabilities)

# Highest score = most non-fiction word, lowest = most fiction word
sortedLlrFiction = sorted(llrValues.items(), key=lambda item: item[1], reverse=True)
sortedLlrNonFiction = sorted(llrValues.items(), key=lambda item: item[1])

# Display top 10 words for fiction
print("Words most indicative of fiction:")
for word, llr in sortedLlrFiction[:50]:
    print(f"{word}: {llr}")

# Display top 10 words for non-fiction (words with the lowest LLR values)
print("\nWords most indicative of non fiction:")
for word, llr in sortedLlrNonFiction[:50]:
    print(f"{word}: {llr}")





# Topic modeling portion
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models
import pandas as pd

# Assuming fictionText and nonFictionText are your preprocessed texts
texts = [normalizedFiction.split(), normalizedNonFiction.split()]  # List of lists of tokens

# Create a Dictionary and Corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Run LDA
numTopics = 2  # adjust as needed
ldaModel = LdaModel(corpus, num_topics=numTopics, id2word=dictionary, passes=15)

# Extract Topics
topics = ldaModel.print_topics(num_words=10)

# Display Topics
for topic in topics:
    print(topic)
