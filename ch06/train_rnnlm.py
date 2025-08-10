import sys
sys.path.append('..')
from common import config
# GPU에서 실행할 경우 config.GPU를 True로 설정하세요
# ============================================
config.GPU = True 
# ============================================
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm




# 하이퍼 파라미터
batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25


if __name__ == '__main__':
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    # 모델 생성
    model = Rnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # 기울기 클리핑을 적용하여 학습
    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
    trainer.plot(ylim=(0, 500))

    # 테스트 데이터로 평가
    model.reset_state()
    test_perplexity = eval_perplexity(model, corpus_test)
    print('Test Perplexity: ', test_perplexity)

    # 매개변수 저장
    model.save_params()