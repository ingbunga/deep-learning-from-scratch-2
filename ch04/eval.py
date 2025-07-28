import sys
sys.path.append('..')
from common.util import most_similar, analogy
import pickle

plk_file = 'cbow_params.pkl'

with open(plk_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

    queries = ['you', 'year', 'car', 'toyota']
    for query in queries:
        most_similar(query, word_to_id, id_to_word, word_vecs)
        print()

    analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
    analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
    analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
    analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)