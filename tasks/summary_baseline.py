import numpy as np
import pandas as pd
from tqdm import tqdm
import os, json, codecs
from collections import Counter
from bert4keras.bert import build_bert_model
from bert4keras.utils import Tokenizer, load_vocab
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from rouge import Rouge
import keras
import math
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True,help='BERT配置文件路径')
parser.add_argument('--checkpoint_path',type=str, required=True, help='BERT权重路径')
parser.add_argument('--dict_path', type=str, required=True, help='词表路径')
parser.add_argument('--albert', default=False, type=bool, required=False, help='是否使用Albert')

parser.add_argument('--train_data_path',type=str, required=True, help='训练集路径')
parser.add_argument('--val_data_path',type=str, required=True, help='验证集路径')
parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch_size')
parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
parser.add_argument('--topk', default=2, type=int, required=False, help='解码TopK')
parser.add_argument('--max_input_len', default=256, type=int, required=False, help='最大输入长度')
parser.add_argument('--max_output_len', default=32, type=int, required=False, help='最大输出长度')

args = parser.parse_args()
print('args:\n' + args.__repr__())

def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])



class DataGenerator(keras.utils.Sequence):

    # 对于所有数据输入，每个 epoch 取 dataSize 个数据
    # data 为 pandas iterator
    def __init__(self, data_path  ,batch_size=8):
        print("init")
        self.data_path = data_path
        data = pd.read_csv(data_path,
                sep = '\t',  
                header=None,
                )
        self.batch_size = batch_size
        self.dataItor = data
        self.data = data.dropna().sample(frac=1)

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.floor(len(self.data) / (self.batch_size))-1

    def __getitem__(self, index):
        # 生成每个batch数据
        batch = self.data[index*self.batch_size:(index+1)*self.batch_size]
        
        # 生成数据
        x,y = self.data_generation(batch,index,len(self.data))
        return [x,y], None

    def on_epoch_end(self):
        #在每一次epoch结束进行一次随机

        self.data = self.data.sample(frac=1)
    def data_generation(self, batch,index,lenth):
        batch_x = []
        batch_y = []
        for a,b in batch.iterrows():
            content_len = len(b[1])
            title_len = len(b[0])
            if(content_len + title_len > max_input_len):
                content = b[1][:max_input_len - title_len]
            else:
                content = b[1]
            x, s = tokenizer.encode(content, b[0])
            batch_x.append(x)
            batch_y.append(s)
        return padding(batch_x),padding(batch_y)



def get_model(config_path,checkpoint_path,keep_words,albert=False,lr = 1e-5):
    model = build_bert_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        application='seq2seq',
        keep_words=keep_words,
        albert=albert
    )


    y_in = model.input[0][:, 1:] # 目标tokens
    y_mask = model.input[1][:, 1:]
    y = model.output[:, :-1] # 预测tokens，预测与目标错开一位

    # 交叉熵作为loss，并mask掉输入部分的预测
    cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(lr))
    return model


def gen_sent(s, topk=2):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:max_input_len])
    target_ids = [[] for _ in range(topk)]  # 候选答案id
    target_scores = [0] * topk  # 候选答案分数
    for i in range(max_output_len):  # 强制要求输出不超过max_output_len字
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        _probas = model.predict([_target_ids, _segment_ids
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

class Evaluate(keras.callbacks.Callback):
    def __init__(self, val_data_path, topk):
        self.data=pd.read_csv(val_data_path,sep = '\t',header=None,)
        self.lowest = 1e10
        self.topk = topk

    def on_epoch_end(self, epoch, logs=None):

        for i in tqdm(range(len(self.data))):
            rouge_l_scores = []
            content = self.data.loc[i][1]
            title = self.data.loc[i][0]
            generated_title = gen_sent(content, self.topk)
            token_title = " ".join([str(t) for t in tokenizer.encode(title)[0]])
            token_gen_title = " ".join([str(t) for t in tokenizer.encode(generated_title)[0]])
            rouge_score = rouge.get_scores(token_gen_title,token_title)
            rouge_l_scores.append(rouge_score[0]['rouge-l']['f'])
        print("rouge-l scores: ",np.mean(rouge_l_scores))


config_path = args.config_path
checkpoint_path = args.checkpoint_path
dict_path = args.dict_path


min_count = 0
max_input_len = args.max_input_len
max_output_len = args.max_output_len
batch_size = args.batch_size
epochs = args.epochs
topk = args.topk

train_data_path = args.train_data_path
val_data_path = args.val_data_path

df_train = pd.read_csv(train_data_path,sep = '\t',header = None)
df_val = pd.read_csv(val_data_path,sep = '\t',header = None)


chars = {}
for df in [df_train,df_val]:
    for a,b in tqdm(df.dropna().iterrows()):
        for w in b[0]: # 纯文本，不用分词
            chars[w] = chars.get(w,0) + 1
            
        for w in b[1]: # 纯文本，不用分词
            chars[w] = chars.get(w,0) + 1
                    
                    
chars = [(i, j) for i, j in chars.items() if j >= min_count]
chars = sorted(chars, key=lambda c: - c[1])
chars = [c[0] for c in chars]


_token_dict = load_vocab(dict_path) # 读取词典
token_dict, keep_words = {}, []

for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.append(_token_dict[c])

for c in chars:
    if c in _token_dict:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])


tokenizer = Tokenizer(token_dict) # 建立分词器

rouge = Rouge()        
model = get_model(config_path, checkpoint_path, keep_words, args.albert, args.lr)

evaluator = Evaluate(val_data_path, topk)

model.fit_generator(
    DataGenerator(train_data_path,batch_size),
    epochs=epochs,
    callbacks=[evaluator]
)


