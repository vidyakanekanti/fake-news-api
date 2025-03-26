from nltk.corpus import stopwords
from nltk.corpus import wordnet

print("Stopwords count:", len(stopwords.words('english')))
print("WordNet Synsets example:", wordnet.synsets('computer'))
