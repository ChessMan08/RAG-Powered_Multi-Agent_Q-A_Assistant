import nltk
from nltk.corpus import wordnet

def define(term: str) -> str:
    """Return the first WordNet definition."""
    syns = wordnet.synsets(term)
    return syns[0].definition() if syns else "No definition found."
