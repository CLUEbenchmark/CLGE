# CLGE
Chinese Language Generation Evaluation 中文生成任务基准测评

为中文生成任务提供预训练模型和任务的排行榜。

## 运行方法
```
1、克隆项目 
   git clone https://github.com/CLUEbenchmark/CLGE.git
2、进入到相应任务的目录
   例如运行 csl 任务
       cd CLGE/tasks/csl  
3、运行对应任务的脚本: 会自动安装依赖和下载模型并开始运行。
       sh run_bert_base.sh
   如运行 sh run_bert_base.sh 会开始 csl 任务在 BERT_base 上的训练

```

## 评价指标

1. Rouge-L

Rouge-L 根据生成文本和参考文本的最长公共子序列（LCS）得出测评分数。
由于 python rouge 只适用于英文，使用 BERT Tokenizer 将中文转换为 id 再计算得分。


## 数据集介绍

1. CSL 中长文本摘要生成 

取自中文科技文献数据(CSL)，选取 10k 条计算机相关领域论文及其标题作为训练集。
 
[运行结果](docs/csl.md)

数据量：训练集(10,000)，验证集(1,000)

|         模型          | 验证集（val) |               训练参数              |
| :-------------------: | :----------: |  :--------------------------------: |
|      ALBERT-tiny      |    54.45     |  batch_size=8, length=256, epoch=5, lr=1e-5  |
|       BERT-base       |    66.69     | batch_size=8, length=256, epoch=5, lr=1e-5  |
|     BERT-wwm-ext      |    66.78     |  batch_size=8, length=256, epoch=10, lr=1e-5 |
|    RoBERTa-wwm-ext    |    66.91     |  batch_size=8, length=256, epoch=10, lr=1e-5 |
|   RoBERTa-wwm-large   |    68.10     |  batch_size=4, length=256, epoch=10, lr=1e-5 |


2. LCSTS 短文本摘要生成

微博短文本及其作者给出的摘要，选取了数据集中 PART_II 部分作为训练集。

[运行结果](docs/lcsts.md)

数据量：训练集(10,000)，验证集(1,000)

|         模型          | 验证集（val) |               训练参数              |
| :-------------------: | :----------: |  :--------------------------------: |
|      ALBERT-tiny      |    31.49     |  batch_size=16, length=128, epoch=10, lr=1e-5  |
|       BERT-base       |    35.59     | batch_size=16, length=128, epoch=5, lr=1e-5  |
|     BERT-wwm-ext      |    35.33     |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|    RoBERTa-wwm-ext    |    36.11     |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|   RoBERTa-wwm-large   |    -     |  batch_size=8, length=128, epoch=10, lr=1e-5 |

