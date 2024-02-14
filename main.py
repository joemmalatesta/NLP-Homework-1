import nltk
from nltk.stem import PorterStemmer
import string
import matplotlib.pyplot as plt
from collections import defaultdict

# Read text files
with open("wizardOfOz.txt", "r") as file:
    wizard_of_oz_text = file.read()

with open("aliceInWonderland.txt", "r") as file:
    alice_in_wonderland_text = file.read()

# Get list of stopWords NLTK's Gist (Code provided in issues)
stopwords_list = nltk.corpus.stopwords.words('english')
stopWords = set(stopwords_list)


def normalizeText(text, lowerCase=False, removeStopWords=False, stemming=False, removePunctuation=False):
    tokenCount = 0
    tokens = defaultdict(int)  # Use defaultdict to simplify counting
    for word in nltk.word_tokenize(text):
        # Lowercase all words in the text.
        if lowerCase:
            word = word.lower()

        # Optional punctuation removal
        if removePunctuation:
            word = word.translate(str.maketrans("", "", string.punctuation))

        # Skip over stopWords if selected. Use lower because that's what the dataset is in
        if removeStopWords and word.lower() in stopWords:
            continue

        # Optional stemming
        if stemming:
            stemmer = PorterStemmer()
            word = stemmer.stem(word)

        tokens[word] += 1
        tokenCount += 1
    # Sort the dictionary by values in descending order - From ChatGPT
    sortedTokens = dict(
        sorted(tokens.items(), key=lambda item: item[1], reverse=True))
    return [tokenCount, sortedTokens]


def compute_log_likelihood_ratio(word, word_count_category, total_count_category, word_count_other_category, total_count_other_category):
    # Calculate probabilities
    p_w_given_c = word_count_category / total_count_category
    p_w_given_other = word_count_other_category / total_count_other_category
    # Avoid division by zero and add-one smoothing
    p_w_given_c_smoothed = (word_count_category + 1) / \
        (total_count_category + len(sortedTokens_alice_in_wonderland))
    p_w_given_other_smoothed = (word_count_other_category + 1) / \
        (total_count_other_category + len(sortedTokens_wizard_of_oz))
    # Calculate log likelihood ratio
    llr = (p_w_given_c_smoothed / p_w_given_other_smoothed) - \
        (p_w_given_other / p_w_given_other)
    return llr


# If name = main whatever. MAIN FUNCTION
lowercaseText = input('Would you like to lowercase text? (y/n)')
if lowercaseText.lower() == 'y':
    lowercaseText = True
else:
    lowercaseText = False

removeStopWords = input('Would you like to remove stop words? (y/n)')
if removeStopWords.lower() == 'y':
    removeStopWords = True
else:
    removeStopWords = False

stemming = input('Would you like to stem words? (y/n)')
if stemming.lower() == 'y':
    stemming = True
else:
    stemming = False

removePunctuation = input('Would you like to remove punctuation? (y/n)')
if removePunctuation.lower() == 'y':
    removePunctuation = True
else:
    removePunctuation = False


tokenCount_wizard_of_oz, sortedTokens_wizard_of_oz = normalizeText(wizard_of_oz_text,
                                                                   lowerCase=lowercaseText,
                                                                   removeStopWords=removeStopWords,
                                                                   stemming=stemming,
                                                                   removePunctuation=removePunctuation)

tokenCount_alice_in_wonderland, sortedTokens_alice_in_wonderland = normalizeText(alice_in_wonderland_text,
                                                                                 lowerCase=lowercaseText,
                                                                                 removeStopWords=removeStopWords,
                                                                                 stemming=stemming,
                                                                                 removePunctuation=removePunctuation)

# Naive Bayes Analysis
log_likelihood_ratios_alice = {}
log_likelihood_ratios_wizard = {}

for word in sortedTokens_alice_in_wonderland:
    if word in sortedTokens_wizard_of_oz:
        word_count_alice = sortedTokens_alice_in_wonderland[word]
        word_count_wizard = sortedTokens_wizard_of_oz[word]
        log_likelihood_ratio_alice = compute_log_likelihood_ratio(word,
                                                                  word_count_alice,
                                                                  tokenCount_alice_in_wonderland,
                                                                  word_count_wizard,
                                                                  tokenCount_wizard_of_oz)
        log_likelihood_ratios_alice[word] = log_likelihood_ratio_alice

for word in sortedTokens_wizard_of_oz:
    if word in sortedTokens_alice_in_wonderland:
        word_count_alice = sortedTokens_alice_in_wonderland[word]
        word_count_wizard = sortedTokens_wizard_of_oz[word]
        log_likelihood_ratio_wizard = compute_log_likelihood_ratio(word,
                                                                   word_count_wizard,
                                                                   tokenCount_wizard_of_oz,
                                                                   word_count_alice,
                                                                   tokenCount_alice_in_wonderland)
        log_likelihood_ratios_wizard[word] = log_likelihood_ratio_wizard

# Output top 10 words with highest log likelihood ratio
print("\nTop 10 words most associated with Alice in Wonderland:")
for word, llr in sorted(log_likelihood_ratios_alice.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{word}: {llr}")

print("\nTop 10 words most associated with The Wizard of Oz:")
for word, llr in sorted(log_likelihood_ratios_wizard.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{word}: {llr}")
