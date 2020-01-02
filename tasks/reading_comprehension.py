import json, os
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
import codecs, re
from rouge import Rouge
import argparse

def boolean_string(s):
    if s not in {'False', 'True', 'false' ,'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True,help='BERT配置文件路径')
parser.add_argument('--checkpoint_path',type=str, required=True, help='BERT权重路径')
parser.add_argument('--dict_path', type=str, required=True, help='词表路径')
parser.add_argument('--albert', default=False, type=boolean_string, required=False, help='是否使用Albert')

parser.add_argument('--train_data_path',type=str, required=True, help='训练集路径')
parser.add_argument('--val_data_path',type=str, required=True, help='验证集路径')
parser.add_argument('--sample_path',type=str, required=False, help='语料样例路径')

parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch_size')
parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
parser.add_argument('--topk', default=2, type=int, required=False, help='解码TopK')
parser.add_argument('--max_p_len', default=256, type=int, required=False, help='最大阅读材料长度')
parser.add_argument('--max_q_len', default=32, type=int, required=False, help='最大问题长度')
parser.add_argument('--max_a_len', default=32, type=int, required=False, help='最大答案长度')

args = parser.parse_args()
print('args:\n' + args.__repr__())

max_p_len = args.max_p_len
max_q_len = args.max_q_len
max_a_len = args.max_a_len
max_qa_len = max_q_len + max_a_len

# bert配置
config_path = args.config_path
checkpoint_path = args.checkpoint_path
dict_path = args.dict_path

train_data = json.load(open(args.train_data_path))


# 加载并精简词表，建立分词器
_token_dict = load_vocab(dict_path)  # 读取词典
token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t, _ in sorted(_token_dict.items(), key=lambda s: s[1]):
    if t not in token_dict:
        if len(t) == 3 and (Tokenizer._is_cjk_character(t[-1])
                            or Tokenizer._is_punctuation(t[-1])):
            continue
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 建立分词器

class data_generator:
    """数据生成器
    """
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
        for i in idxs:
            D = self.data[i]
            question = D['question']
            answers = [p['answer'] for p in D['passages'] if p['answer']]
            passage = np.random.choice(D['passages'])['passage']
            passage = re.sub(u' |、|；|，', ',', passage)
            final_answer = ''
            for answer in answers:
                if all([a in passage[:max_p_len - 2] for a in answer.split(' ')]):
                    final_answer = answer.replace(' ', ',')
                    break
            qa_token_ids, qa_segment_ids = tokenizer.encode(
                question, final_answer, max_length=max_qa_len + 1)
            p_token_ids, p_segment_ids = tokenizer.encode(passage,
                                                          max_length=max_p_len)
            token_ids = p_token_ids + qa_token_ids[1:]
            segment_ids = p_segment_ids + qa_segment_ids[1:]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d

model = build_bert_model(
    config_path,
    checkpoint_path,
    application='seq2seq',
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
    albert=args.albert,
)

# 交叉熵作为loss，并mask掉输入部分的预测
y_in = model.input[0][:, 1:]  # 目标tokens
y_mask = model.input[1][:, 1:]
y = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


def get_ngram_set(x, n):
    """生成ngram合集，返回结果格式是:
    {(n-1)-gram: set([n-gram的第n个字集合])}
    """
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i: i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result


def gen_answer(question, passages, topk=2, mode='extractive'):
    """beam search解码来生成答案
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索。
    passages为多篇章组成的list，从多篇文章中自动决策出最优的答案，
    如果没答案，则返回空字符串。
    mode是extractive时，按照抽取式执行，即答案必须是原篇章的一个片段。
    """
    token_ids, segment_ids = [], []
    for passage in passages:
        passage = re.sub(u' |、|；|，', ',', passage)
        p_token_ids = tokenizer.encode(passage, max_length=max_p_len)[0]
        q_token_ids = tokenizer.encode(question, max_length=max_q_len + 1)[0]
        token_ids.append(p_token_ids + q_token_ids[1:])
        segment_ids.append([0] * len(token_ids[-1]))
    target_ids = [[] for _ in range(topk)]  # 候选答案id
    target_scores = [0] * topk  # 候选答案分数
    for i in range(max_a_len):  # 强制要求输出不超过max_q_len字
        _target_ids, _segment_ids = [], []
        # 篇章与候选答案组合
        for tids, sids in zip(token_ids, segment_ids):
            for t in target_ids:
                _target_ids.append(tids + t)
                _segment_ids.append(sids + [1] * len(t))
        _padded_target_ids = sequence_padding(_target_ids)
        _padded_segment_ids = sequence_padding(_segment_ids)
        _probas = model.predict([_padded_target_ids, _padded_segment_ids
                                 ])[..., 3:]  # 直接忽略[PAD], [UNK], [CLS]
        _probas = [
            _probas[j, len(ids) - 1] for j, ids in enumerate(_target_ids)
        ]
        _probas = np.array(_probas).reshape((len(token_ids), topk, -1))
        if i == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果开始[SEP]为最大值，那说明该篇章没有答案
            _probas_argmax = _probas[:, 0].argmax(axis=1)
            _available_idxs = np.where(_probas_argmax != 0)[0]
            if len(_available_idxs) == 0:
                return ''
            else:
                _probas = _probas[_available_idxs]
                token_ids = [token_ids[j] for j in _available_idxs]
                segment_ids = [segment_ids[j] for j in _available_idxs]
        if mode == 'extractive':
            # 如果是抽取式，那么答案必须是篇章的一个片段
            # 那么将非篇章片段的概率值全部置0
            _zeros = np.zeros_like(_probas)
            _ngrams = {}
            for p_token_ids in token_ids:
                for k, v in get_ngram_set(p_token_ids, i + 1).items():
                    _ngrams[k] = _ngrams.get(k, set()) | v
            for j, t in enumerate(target_ids):
                _available_idxs = _ngrams.get(tuple(t), set())
                _available_idxs.add(token_dict['[SEP]'])
                _available_idxs = [k - 3 for k in _available_idxs]
                _zeros[:, j, _available_idxs] = _probas[:, j, _available_idxs]
            _probas = _zeros
        _probas = (_probas**2).sum(0) / (_probas.sum(0) + 1)  # 某种平均投票方式
        _log_probas = np.log(_probas + 1e-6)  # 取对数，方便计算
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
        target_ids = [_candidate_ids[k] for k in _topk_arg]
        target_scores = [_candidate_scores[k] for k in _topk_arg]
        best_one = np.argmax(target_scores)
        if target_ids[best_one][-1] == 3:
            return tokenizer.decode(target_ids[best_one])
    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])

example = json.load(open(args.sample_path))

def just_show():
    for d in example:
        q_text = d['question']
        p_texts = [p['passage'] for p in d['passages']]
        p_answer = [p['answer'] for p in d['passages'] if len(p['answer'])>=1][0]
        g_answer = gen_answer(q_text, p_texts, 1, None)
        print("问: "+ q_text+" | 答: "+g_answer)

rouge = Rouge()

class Evaluate(keras.callbacks.Callback):
    def __init__(self,val_path):
        self.data = json.load(open(val_path))

    def on_epoch_end(self, epoch, logs=None):
        just_show()
        rouge_1_scores = []
        rouge_2_scores =  []
        rouge_l_scores = []
        for d in self.data:
            q_text = d['question']
            p_texts = [p['passage'] for p in d['passages']]
            p_answer = [p['answer'] for p in d['passages'] if len(p['answer'])>=1]
            if len(p_answer)>0:
                p_answer = p_answer[0]
            else:
                p_answer = '-'
            g_answer = gen_answer(q_text, p_texts, 1, None)
            token_p_answer = " ".join([str(t) for t in tokenizer.encode(p_answer)[0]])
            token_g_answer = " ".join([str(t) for t in tokenizer.encode(g_answer)[0]])
            rouge_score = rouge.get_scores(token_g_answer,token_p_answer)
            rouge_1_scores.append(rouge_score[0]['rouge-1']['f'])
            rouge_2_scores.append(rouge_score[0]['rouge-2']['f'])
            rouge_l_scores.append(rouge_score[0]['rouge-l']['f'])
        print('rouge-1:',np.mean(rouge_1_scores))
        print('rouge-2:',np.mean(rouge_2_scores))
        print('rouge-L:',np.mean(rouge_l_scores))


evaluator = Evaluate(args.val_data_path)

train_D = data_generator(train_data, args.batch_size)

model.fit_generator(train_D.forfit(),
                    steps_per_epoch=len(train_D),
                    callbacks=[evaluator],
                    epochs=args.epochs,)