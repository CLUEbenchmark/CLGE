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
```
数据量：训练集(10,000)，验证集(1,000)，测试集(1,000)
示例：
{
    title: 基于活跃时间分组的软件众包工人选择机制
    content: 针对现有的软件众包工人选择机制对工人间协同开发考虑不足的问题,在竞标模式的基础上提出一种基于活跃时间分组的软件众包工人选择机制。首先,基于活跃时间将众包工人划分为多个协同开发组;然后,根据组内工人开发能力和协同因子计算协同工作组权重;最后,选定权重最大的协同工作组为最优工作组,并根据模块复杂度为每个任务模块从该组内选择最适合的工人。实验结果表明,该机制相比能力优先选择方法在工人平均能力上仅有0. 57%的差距,同时因为保证了工人间的协同而使项目风险平均降低了32%,能有效指导需多人协同进行的众包软件任务的工人选择。
}
```
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
```
数据量：训练集(8,666)，验证集(1,000)，测试集(1,000)
示例：
{
    abstract: 新《环保法》元旦起实施企业违法可拘留责任人
    content: 正式实施的新《环保法》对企业惩治力度大大加强，受到罚款处罚，被责令改正，拒不改正的，依法作出处罚决定的行政机关可以自责令更改之日的次日起，按照原处罚数额按日连续处罚。企业违法可拘留责任人。
}
```
|         模型          | 验证集（val) |               训练参数              |
| :-------------------: | :----------: |  :--------------------------------: |
|      ALBERT-tiny      |    31.49     |  batch_size=16, length=128, epoch=10, lr=1e-5  |
|       BERT-base       |    35.59     | batch_size=16, length=128, epoch=5, lr=1e-5  |
|     BERT-wwm-ext      |    35.33     |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|    RoBERTa-wwm-ext    |    36.11     |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|   RoBERTa-wwm-large   |    -     |  batch_size=8, length=128, epoch=10, lr=1e-5 |

