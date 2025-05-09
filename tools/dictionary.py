import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def define(term: str) -> str:
    """Return the first WordNet definition for the term."""
    syns = wordnet.synsets(term)
    if not syns:
        return "No definition found."
    return syns[0].definition()