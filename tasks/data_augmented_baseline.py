#! -*- coding: utf-8 -*-

from __future__ import print_function
import re
import numpy as np
import json
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import uniout  # 打印中文
from keras.layers import Input, Embedding, Reshape
from keras.models import Model
import pandas as pd
import random
import argparse

def save_data(file,data,label):
    fp = open(file,'a+', encoding='utf-8')
    for l in data:
        s1 = l[0]
        s2 = l[1]
        fp.write('{\"sentence1\":\"' + s1 + '\",\"sentence2\":\"' + s2 +'\",\"label\":\"' + str(label) +  '\"}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True,help='BERT配置文件路径')
    parser.add_argument('--checkpoint_path',type=str, required=True, help='BERT权重路径')
    parser.add_argument('--dict_path', type=str, required=True, help='词表路径')
    parser.add_argument('--train_data_path',type=str, required=True, help='训练集路径')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch_size')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
    parser.add_argument('--topk1', default=25, type=int, required=False, help='最大长度')
    parser.add_argument('--topk2', default=2, type=int, required=False, help='最大长度')
    parser.add_argument('--max_seq_len', default=256, type=int, required=False, help='最大长度')  
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    maxlen = args.max_seq_len
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    dict_path = args.dict_path
    batch_size = args.batch_size
    epochs = args.epochs
    num_classes = 2
    lr = args.lr

    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    dict_path = args.dict_path
    train_data = args.train_data_path

    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)

    train_df = pd.read_csv(train_data,sep='\t', header = None)
    train_df.columns = ['s1','s2','label']

    class data_generator(DataGenerator):
        """数据生成器
        """
        def __iter__(self, r=False):
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            batch_token_ids, batch_segment_ids, batch_labels = [], [], []
            for i in idxs:
                line = self.data.loc[i]
                if (random.random() < 0.5):
                    s1 = line['s1'].replace('***','*')
                    s2 = line['s2'].replace('***','*')
                else:
                    s2 = line['s1'].replace('***','*')
                    s1 = line['s2'].replace('***','*')                
                token_ids, segment_ids = tokenizer.encode(s1, s2, max_length=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([line['label']])
                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids, batch_labels], None
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    class CrossEntropy(Loss):
        """交叉熵作为loss，并mask掉padding部分
        """
        def compute_loss(self, inputs, mask=None):
            y_true, y_pred = inputs
            if mask[1] is None:
                y_mask = 1.0
            else:
                y_mask = K.cast(mask[1], K.floatx())[:, 1:]
            y_true = y_true[:, 1:]  # 目标token_ids
            y_pred = y_pred[:, :-1]  # 预测序列，错开一位
            loss = K.sparse_categorical_crossentropy(y_true, y_pred)
            loss = K.sum(loss * y_mask) / K.sum(y_mask)
            return loss

    c_in = Input(shape=(1,))
    c = Embedding(num_classes, maxlen)(c_in)
    c = Reshape((maxlen,))(c)

    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='lm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        layer_norm_cond=c,
        additional_input_layers=c_in,
    )
    output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])
    model = Model(model.inputs, output)
    model.compile(optimizer=Adam(lr))
    model.summary()

    def random_generate(c=0, n=2, s1_topk=5):
        """随机采样生成句子对
        每次从最高概率的topk个token中随机采样一个
        """
        label_ids = [[c] for _ in range(n)]
        target_ids = [[2] for _ in range(n)]
        sep_index = [0 for _ in range(n)]
        R = []
        for i in range(64):
            segment_ids = []
            for t,index in zip(target_ids,sep_index):
                if index > 0:
                    segment_ids.append([0] * index+ [1] * (len(t) - index))
                else:
                    segment_ids.append([0] * len(t))
            # 下面直接忽略[PAD], [UNK], [CLS]
            _probas = model.predict([target_ids, segment_ids, label_ids])[:, -1, 3:]
            for j, p in enumerate(_probas):
                p_arg_topk = p.argsort()[::-1][:s1_topk]
                #if 0 in p_arg_topk:
                #    target_ids[j].append(3)
                #else:
                p_topk = p[p_arg_topk]
                p = p_topk / sum(p_topk)
                idx = np.random.choice(len(p), p=p)
                target_ids[j].append(p_arg_topk[idx] + 3)
                    
                if p_arg_topk[idx] + 3 == 3 and sep_index[j] == 0:
                    sep_index[j] = i
        for tokens in target_ids:
            tokens.append(3)
            cls_index = tokens.index(3)
            R.append(tokenizer.decode(tokens[:cls_index]))
            #sentences.sort(key = lambda i:len(i),reverse=True) 
        return R

    def gen_sent(s, label, topk=2):
        """beam search解码
        每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
        """
        label_ids = [[label] for _ in range(topk)]
        token_ids, segment_ids = tokenizer.encode(s)
        target_ids = [[] for _ in range(topk)]  # 候选答案id
        target_scores = [0] * topk  # 候选答案分数
        for i in range(64):  # 强制要求输出不超过max_output_len字
            _target_ids = [token_ids + t for t in target_ids]
            _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
            _probas = model.predict([_target_ids, _segment_ids, label_ids
                                    ])[:, -1, 3:]  # 直接忽略[PAD], [UNK], [CLS]
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

    def gen_sen_pair(label,n,s1_topk,s2_topk):
        s1_pair = random_generate(label,n,s1_topk)
        output = []
        for line in s1_pair:
            s2 = gen_sent(line,label,s2_topk)
            output.append([line,s2])
        return output

    class Evaluate(keras.callbacks.Callback):
        def __init__(self):
            self.lowest = 1e10

        def on_epoch_end(self, epoch, logs=None):
            # 保存最优
            if logs['loss'] <= self.lowest:
                self.lowest = logs['loss']
                model.save_weights('./best_model.weights')
            print("正样本:")
            print(gen_sen_pair(1,2,topk1,topk2))
            print("负样本:")
            print(gen_sen_pair(0,2,topk1,topk2))


    train_generator = data_generator(train_df, batch_size)
    evaluator = Evaluate()
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator]) 



if __name__ == "__main__":
    main()
