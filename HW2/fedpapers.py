import numpy as np
import json
from sklearn.feature_extraction import text

x = open('fedpapers_split.txt').read()
papers = json.loads(x)

papersH = papers[0]  # papers by Hamilton
papersM = papers[1]  # papers by Madison
papersD = papers[2]  # disputed papers

nH, nM, nD = len(papersH), len(papersM), len(papersD)
print('Number of papers by Hamilton: ', nH)
print('Number of papers by Madison: ', nM)
print('Number of disputed papers: ', nD)

# This allows you to ignore certain common words in English
# You may want to experiment by choosing the second option or your own
# list of stop words, but be sure to keep 'HAMILTON' and 'MADISON' in
# this list at a minimum, as their names appear in the text of the papers
# and leaving them in could lead to unpredictable results
#stop_words = text.ENGLISH_STOP_WORDS.union({'HAMILTON','MADISON'})

stop_words = {'HAMILTON', 'MADISON'}

# Form bag of words model using words used at least 10 times
vectorizer = text.CountVectorizer(stop_words=stop_words, min_df=10)
X = vectorizer.fit_transform(papersH+papersM+papersD).toarray()

# X is a matrix of shape 80 x 1307 : each row represents a paper, for which we have the frequency of apparition for 1307
# vocabulary words

# Uncomment this line to see the full list of words remaining after filtering out
# stop words and words used less than min_df times
#print(vectorizer.vocabulary_)  # Gives the number of apparition in all papers (summed) for each word of 1307 words list

# Split word counts into separate matrices
XH, XM, XD = X[:nH,:], X[nH:nH+nM,:], X[nH+nM:,:]

# Apply Laplace smoothing : See slide 15 in the fourth set of slides
XH += 1
XM += 1
XD += 1

# Estimate probability of each word in vocabulary being used by Hamilton
fH = (XH.sum(axis=0))/XH.sum()

# Estimate probability of each word in vocabulary being used by Madison
fM = (XM.sum(axis=0))/XM.sum()

# Compute ratio of these probabilities
fratio = fH/fM

# Compute prior probabilities 
piH = nH/(nH+nM)
piM = nM/(nH+nM)

hamilton_papers = 0
madison_papers = 0

for xd in XD:  # Iterate over disputed documents
    # Compute likelihood ratio for Naive Bayes model for the current disputed document
    LR = (piH/piM)*np.prod(fratio ** xd)

    if LR > 1:  # shouldn't we compare to 1 ? As stated in slide 9 of lecture 2
        print('Hamilton')
        hamilton_papers += 1
    else:
        print('Madison')
        madison_papers += 1

print('Number of disputed documents classified as authored by Hamilton: ', hamilton_papers)
print('Number of disputed documents classified as authored by Madison: ', madison_papers)

