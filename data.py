import os
import sys
import random
import json

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

def load_pairs_data():
    '''
    从pair文件中读取文件
    '''
    res = []
    for f in ['train', 'valid', 'test']:
        pairs_data = []
        filename = 'iqa.' + f + '.tokenlized.pair.json'
        file_path = os.path.join('data', filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                data = json.load(source_file)
                for i in range(0, len(data), 11):
                    pairs_data.append(data[i])
                    rand_neg = random.randint(1,10) + i
                    pairs_data.append(data[rand_neg])
                    # print(data[i]['qid'], data[i]['label'],data[rand_neg]['qid'],data[rand_neg]['label'])

                res.append(pairs_data)
    return res

_train_data, _test_data, _valid_data = load_pairs_data()

'''
build vocab data with more placeholder
'''
def load_pairs_vocab():
    '''
    从pair中读取字典
    '''
    vocab_file = os.path.join('data', 'iqa.vocab.json')
    with open(vocab_file, 'r', encoding='utf-8', errors='ignore') as source_file:
        vocab = json.load(source_file)

    return vocab

vocab_data = load_pairs_vocab()
print("keys", vocab_data.keys())
vocab_size = len(vocab_data['word2id'].keys())
VOCAB_PAD_ID = vocab_size+1
VOCAB_GO_ID = vocab_size+2
vocab_data['word2id']['<PAD>'] = VOCAB_PAD_ID
vocab_data['word2id']['<GO>'] = VOCAB_GO_ID
vocab_data['id2word'][VOCAB_PAD_ID] = '<PAD>'
vocab_data['id2word'][VOCAB_GO_ID] = '<GO>'


def _get_corpus_metrics():
    '''
    max length of questions
    '''
    for cat, data in zip(["valid", "test", "train"], [_valid_data, _test_data, _train_data]):
        max_len_question = 0
        total_len_question = 0
        max_len_utterance = 0
        total_len_utterance = 0
        for x in data:
            total_len_question += len(x['question']) 
            total_len_utterance += len(x['utterance'])
            if len(x['question']) > max_len_question: 
                max_len_question = len(x['question'])
            if len(x['utterance']) > max_len_utterance: 
                max_len_utterance = len(x['utterance'])
        print('max len of %s question : %d, average: %d' % (cat, max_len_question, total_len_question/len(data)))
        print('max len of %s utterance: %d, average: %d' % (cat, max_len_utterance, total_len_utterance/len(data)))
    # max length of answers


class BatchIter():
    '''
    Load data with mini-batch
    '''
    def __init__(self, data = None, batch_size = 100):
        assert data is not None, "data should not be None."
        self.batch_size = batch_size
        self.data = data

    def next(self):
        random.shuffle(self.data)
        index = 0
        total_num = len(self.data)
        while index <= total_num:
            yield self.data[index:index + self.batch_size]
            index += self.batch_size

def padding(lis, pad, size):
    '''
    right adjust a list object
    '''
    if size > len(lis):
        lis += [pad] * (size - len(lis))
    else:
        lis = lis[0:size]
    return lis

def pack_question_n_utterance(q, u, q_length = 20, u_length = 99):
    '''
    combine question and utterance as input data for feed-forward network
    '''
    assert len(q) > 0 and len(u) > 0, "question and utterance must not be empty"
    q = padding(q, VOCAB_PAD_ID, q_length)
    u = padding(u, VOCAB_PAD_ID, u_length)
    assert len(q) == q_length, "question should be pad to q_length"
    assert len(u) == u_length, "utterance should be pad to u_length"
    return q + [VOCAB_GO_ID] + u

def __resolve_input_data(data, batch_size, question_max_length = 20, utterance_max_length = 99):
    '''
    resolve input data
    '''
    batch_iter = BatchIter(data = data, batch_size = batch_size)
    
    for mini_batch in batch_iter.next():
        result = []
        for o in mini_batch:
            x = pack_question_n_utterance(o['question'], o['utterance'], question_max_length, utterance_max_length)
            y_ = o['label']
            assert len(x) == utterance_max_length + question_max_length + 1, "Wrong length afer padding"
            assert VOCAB_GO_ID in x, "<GO> must be in input x"
            assert len(y_) == 2, "desired output."
            result.append([x, y_])
        if len(result) > 0:
            # print('data in batch:%d' % len(mini_batch))
            yield result
        else:
            raise StopIteration

# export data

def load_train(batch_size = 100, question_max_length = 20, utterance_max_length = 99):
    '''
    load train data
    '''
    return __resolve_input_data(_train_data, batch_size, question_max_length, utterance_max_length)

def load_test(question_max_length = 20, utterance_max_length = 99):
    '''
    load test data
    '''
    result = []
    for o in _test_data:
        x = pack_question_n_utterance(o['question'], o['utterance'], question_max_length, utterance_max_length)
        y_ = o['label']
        assert len(x) == utterance_max_length + question_max_length + 1, "Wrong length afer padding"
        assert VOCAB_GO_ID in x, "<GO> must be in input x"
        assert len(y_) == 2, "desired output."
        result.append((x, y_))
    return result

def load_valid(batch_size = 100, question_max_length = 20, utterance_max_length = 99):
    '''
    load valid data
    '''
    return __resolve_input_data(_valid_data, batch_size, question_max_length, utterance_max_length)

def test_batch():
    data = load_test()
    print("DATA Structure:", data[:2])
    print("test length", len(data))
    print("VOCAB_PAD_ID", VOCAB_PAD_ID)
    print("VOCAB_GO_ID", VOCAB_GO_ID)

if __name__ == '__main__':
    test_batch()


