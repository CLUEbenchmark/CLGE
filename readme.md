# LGEC
Language Generation Evaluation for Chinese 中文生成任务基准测评

为中文生成任务提供预训练模型和任务的排行榜。

## 模型介绍

BERT 类模型可以直接用于各种下游 NLP 任务，例如文本分类、命名实体识别和机器阅读理解等。
但是对于文本生成类（NLG）任务，BERT 在预训练时使用双向语言模型与单向的文本生成目标不一致，
导致 BERT 在文本生成的表现并不好。

因此，将 BERT 用于文本生成需要一些改动，以下是一些研究者的尝试：

1. [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)

UNILM 通过对 BERT 的注意力矩阵进行 mask 实现单向输出的语言模型。

![image](docs/images/unilm.png)

2. [Pretraining-Based Natural Language Generation for Text Summarization](https://arxiv.org/abs/1902.09243 )

以两阶段的方式使用两个 BERT 进行编码和解码实现 Encoder-Decoder 架构。

![image](docs/images/pbnlg.png)

## 评价指标

1. Rouge-L

Rouge-L 根据生成文本和参考文本的最长公共子序列（LCS）得出测评分数，计算方法如下：



## 数据集介绍

1. CSL 中长文本摘要生成 

[运行结果](docs/csl.md)

数据量：训练集(10,000)，验证集(1,000)

|         模型          | 验证集（val) |               训练参数              |
| :-------------------: | :----------: |  :--------------------------------: |
|      ALBERT-tiny      |    54.45     |  batch_size=8, length=256, epoch=5, lr=1e-5  |
|       BERT-base       |    66.69     | batch_size=8, length=256, epoch=5, lr=1e-5  |
|     BERT-wwm-ext      |    66.78     |  batch_size=8, length=256, epoch=10, lr=1e-5 |
|    RoBERTa-wwm-ext    |    66.91     |  batch_size=4, length=256, epoch=10, lr=1e-5 |
|   RoBERTa-wwm-large   |    -     |  batch_size=4, length=256, epoch=10, lr=1e-5 |


2. LCSTS 短文本摘要生成

[运行结果](docs/lcsts.md)

数据量：训练集(10,000)，验证集(1,000)

|         模型          | 验证集（val) |               训练参数              |
| :-------------------: | :----------: |  :--------------------------------: |
|      ALBERT-tiny      |    31.49     |  batch_size=16, length=128, epoch=10, lr=1e-5  |
|       BERT-base       |    35.59     | batch_size=16, length=128, epoch=5, lr=1e-5  |
|     BERT-wwm-ext      |    35.33     |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|    RoBERTa-wwm-ext    |    36.11     |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|   RoBERTa-wwm-large   |    -     |  batch_size=8, length=128, epoch=10, lr=1e-5 |

