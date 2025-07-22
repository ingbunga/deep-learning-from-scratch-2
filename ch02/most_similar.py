import sys
sys.path.append('..')
from common.util import most_similar, preprocess, create_co_matrix

txt = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(txt)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C)