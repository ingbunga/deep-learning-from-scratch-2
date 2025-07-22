import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, most_similar, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)
print('동시 발생 행렬')
print(C)
print('-' * 50)
print('PPMI 행렬')
print(W)