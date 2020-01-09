from tqdm import tqdm
import numpy as np
import os,json,glob,codecs
import tensorflow as tf
import keras
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path',type=str, required=True, help='训练数据存放的根路径')
parser.add_argument('--seq2seq_config_path', default="./", type=str, required=False,help='seq2seq配置文件路径')

parser.add_argument('--min_count', default=32, type=int, required=False, help='字符出现的最小次数')
parser.add_argument('--maxlen', default=400, type=int, required=False, help='文章的最大长度')
parser.add_argument('--char_size', default=128, type=int, required=False, help='字符embedding大小')
parser.add_argument('--topk', default=3, type=int, required=False, help='解码TopK')
parser.add_argument('--z_dim', default=128, type=int, required=False, help='隐藏层单元个数')
parser.add_argument('--epochs', default=100, type=int, required=False, help='训练循环次数')
parser.add_argument('--batch_size', default=64, type=int, required=False, help='训练batch大小')

args = parser.parse_args()
print('args:\n' + args.__repr__())


train_data_path = args.train_data_path
seq2seq_config_path = args.seq2seq_config_path

min_count = args.min_count
maxlen = args.maxlen
batch_size = args.batch_size
epochs = args.epochs
char_size = args.char_size
z_dim = args.z_dim
topk = args.topk


# 训练样本。THUCNews数据集，每个样本保存为一个txt。
txts = glob.glob(train_data_path + '/*/*.txt')
def _batch_texts():
    texts = []
    for txt in txts:
        text = codecs.open(txt, encoding='utf-8').read()
        texts.append(text)
        if len(texts) == 100:
            yield texts
            texts = []
    if texts:
        yield texts # 返回最后一批数据（可能不足100）

if os.path.exists(seq2seq_config_path + 'seq2seq_config.json'):
    chars,id2char,char2id = json.load(open(seq2seq_config_path + 'seq2seq_config.json'))
    id2char = {int(i):j for i,j in id2char.items()}
else:
    chars = {}
    for text in tqdm(_batch_texts()):
        for t in text:
            title, content = t.split('\n')[0].strip(), ''.join(t.split('\n')[1:]).replace('\u3000', '')
            for w in content:
                chars[w] = chars.get(w,0) + 1
            for w in title:
                chars[w] = chars.get(w,0) + 1
    chars = {i:j for i,j in chars.items() if j >= min_count}
    # 0: mask
    # 1: unk
    # 2: start
    # 3: end
    id2char = {i+4:j for i,j in enumerate(chars)}
    char2id = {j:i for i,j in id2char.items()}
    json.dump([chars,id2char,char2id], open(seq2seq_config_path + 'seq2seq_config.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end: # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen-2]]
        ids = [2] + ids + [3]
    else: # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids

def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])

def padding(x):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]

def data_generator():
    # 数据生成器
    X,Y = [],[]
    while True:
        for text in _batch_texts():
            for t in text:
                title, content = t.split('\n')[0].strip(), ''.join(t.split('\n')[1:]).replace('\u3000', '')
                X.append(str2id(content))
                # title 在首尾统一加上 start / end
                Y.append(str2id(title, start_end=True))
                if len(X) == batch_size:
                    X = np.array(padding(X))
                    Y = np.array(padding(Y))
                    yield [X,Y], None
                    X,Y = [],[]
                
def to_one_hot(x):
    """输出一个词表大小的向量，来标记该词是否在文章出现过
    """
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars)+4) # 经过onehot，会在x最后增加一个维度，val=len(chars)+4
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x

class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape)-1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')
    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs

class OurBidirectional(Layer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name # 给层重命名
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0]) # 将维度降至1维，因为 tf.reverse_sequence 的 seq_lengths 参数要求传入1维的list
        seq_len = K.cast(seq_len, 'int32')
        return tf.reverse_sequence(x, seq_len, seq_dim=1)
    def call(self, inputs):
        x, mask = inputs # 输入是一个list
        x_forward = self.forward_layer(x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.backward_layer(x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], -1)
        if K.ndim(x) == 3:
            return x * mask
        else:
            return x
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.forward_layer.units * 2,)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)

class SelfModulatedLayerNormalization(Layer):
    """模仿Self-Modulated Batch Normalization，
    只不过将Batch Normalization改为Layer Normalization
    """
    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden # 32
    def build(self, input_shape): # build 定义权重的地方
        super(SelfModulatedLayerNormalization, self).build(input_shape) # 相当于设 self.built = True
        output_dim = input_shape[0][-1] # 128
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = Dense(output_dim)
        self.gamma_dense_1 = Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = Dense(output_dim)
    def call(self, inputs): # call 编写层的功能逻辑的地方
        inputs, cond = inputs
        inputs = self.layernorm(inputs)
        beta = self.beta_dense_1(cond)   
        beta = self.beta_dense_2(beta)   
        gamma = self.gamma_dense_1(cond) 
        gamma = self.gamma_dense_2(gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return inputs * (gamma + 1) + beta
    def compute_output_shape(self, input_shape): 
        return input_shape[0]


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = tf.einsum('ijkl,ijml->ijkm', qw, kw) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('ijkl,ijlm->ijkm', a, vw)
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)



# 搭建seq2seq模型
x_in = Input(shape=(None,))
y_in = Input(shape=(None,))
x, y = x_in, y_in

# x_mask shape: (batch, seq_len, 1)
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

# 输出一个词表大小的向量，来标记该词是否在文章出现过
x_one_hot = Lambda(to_one_hot)([x, x_mask])
x_prior = ScaleShift()(x_one_hot) # 学习输出的先验分布（标题的字词很可能在文章出现过）不改变shape

embedding = Embedding(len(chars)+4, char_size)
x = embedding(x)
y = embedding(y) # 对目标 title 做embedding

# encoder (1) 双层双向LSTM
x = LayerNormalization()(x) # 层归一化
x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x = LayerNormalization()(x)
x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x_max = Lambda(seq_maxpool)([x, x_mask]) # 做 max_pooling

# decoder（1）双层单向LSTM
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max]) # 对y进行层归一化，同时让y与x_max交互
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])

# attention 交互
xy = Attention(8, 16)([y, x, x, x_mask])
xy = Concatenate()([y, xy])

# 输出分类
xy = Dense(char_size)(xy)
# ReLU是将所有的负值都设为零，相反，LeakyReLU是给所有负值赋予一个非零斜率
xy = LeakyReLU(0.2)(xy) # from keras.layers import LeakyReLU
xy = Dense(len(chars) + 4)(xy)
xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior]) # 与先验结果取平均
xy = Activation('softmax')(xy)

# 交叉熵作为 loss，但mask掉padding部分
cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

model = Model([x_in, y_in], xy)
model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-3))


def gen_sent(s, topk=3, maxlen=64):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s)] * topk) # 输入转id
    yid = np.array([[2]] * topk) # 解码均以<start>开头，这里<start>的id为2
    scores = [0] * topk # 候选答案分数
    for i in range(maxlen): # 强制要求输出不超过maxlen字
        proba = model.predict([xid, yid])[:, i, 3:] # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6) # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:,-topk:] # 每一项选出topk
        _yid = [] # 暂存的候选目标序列
        _scores = [] # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]+3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(topk):
                for k in range(topk): # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]+3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:] # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = np.array(_yid)
        scores = np.array(_scores)
        best_one = np.argmax(scores)
        if yid[best_one][-1] == 3:
            return id2str(yid[best_one])
    # 如果maxlen字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])


def show():
	s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医 。'
	s2 = u'在海底捞就餐过程中，在锅底吃出烟头，在与商家沟通过程中，店员觉得顾客音量大，在店面影响不好，报警让警方过来处理。目前双方仍在协商中。'
	s3 = u'1月5日，作为全球新年后的第一场国际田联金标赛事，2020年厦门马拉松鸣枪开跑，在3万跑者前进的步伐中迎来18岁“成人礼”。'
	print(gen_sent(s1, topk))
    print(gen_sent(s2, topk))
    print(gen_sent(s3, topk))

class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        show()
        # 保存最有结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
            
evaluator = Evaluate()

# 开始训练
model.fit_generator(data_generator(),
                    steps_per_epoch=1000,
                    epochs=epochs,
                    callbacks=[evaluator])