import sys
sys.path.append('..')
from common.np import *
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size: int, hidden_size: int, window_size: int, corpus):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 계층 생성
        self.in_layers: list[Embedding] = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 배열에 모은다.
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for in_layer in layers:
            self.params += in_layer.params
            self.grads += in_layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장
        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray):
        h = np.zeros((contexts.shape[0], self.word_vecs.shape[1]), dtype='f')
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h /= len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout /= len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
