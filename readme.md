# CLGE
Chinese Language Generation Evaluation 中文生成任务基准测评

为中文生成任务提供数据集、基准(预训练)模型和排行榜。


## 一键运行

```
1、克隆项目 
   git clone --depth=1 https://github.com/CLUEbenchmark/CLGE.git
2、进入到相应任务的目录
   例如运行 csl 任务
       cd CLGE/tasks/csl  
3、运行对应任务的脚本: 会自动安装依赖、下载模型和数据集并开始运行。
       sh run_bert_base.sh
   如运行 sh run_bert_base.sh 会开始 csl 任务在 BERT_base 上的训练
```

## 测评指标

**1. Rouge-1**

rouge-1 比较生成文本和参考文本之间的重叠词（字）数量

**2. Rouge-2**

rouge-2 比较生成文本和参考文本之间的 2-gram 重叠的数量


**3. Rouge-L**

rouge-l 根据生成文本和参考文本之间的最长公共子序列得出

**4. BLEU**

[Bilingual Evaluation Understudy](https://www.aclweb.org/anthology/P02-1040/)



## 数据集介绍

### **1. CSL 中长文本摘要生成**

[中文科学文献数据(CSL)](https://github.com/P01son6415/chinese-scientific-literature-dataset)，选取 10k 条计算机相关领域论文及其标题作为训练集。

```
数据量：训练集(10,000)，验证集(1,000)，测试集(1,000)
示例：
{
    title: 基于活跃时间分组的软件众包工人选择机制
    content: 针对现有的软件众包工人选择机制对工人间协同开发考虑不足的问题,在竞标模式的基础上提出一种基于活跃时间分组的软件众包工人选择机制。首先,基于活跃时间将众包工人划分为多个协同开发组;然后,根据组内工人开发能力和协同因子计算协同工作组权重;最后,选定权重最大的协同工作组为最优工作组,并根据模块复杂度为每个任务模块从该组内选择最适合的工人。实验结果表明,该机制相比能力优先选择方法在工人平均能力上仅有0. 57%的差距,同时因为保证了工人间的协同而使项目风险平均降低了32%,能有效指导需多人协同进行的众包软件任务的工人选择。
}
```

[运行结果](docs/csl.md)

|         模型          | Rouge-L | Rouge-1 | Rouge-2 | BLEU |             训练参数              |
| :-------------------: | :------: |:---: |:---: |:---: |  :--------------------------------: |
|      ALBERT-tiny      |  48.11  | 52.75 | 37.96 | 21.63 |  batch_size=8, length=256, epoch=10, lr=1e-5  |
|       BERT-base       |  59.76  | 63.83 | 51.29 | 41.45 |  batch_size=8, length=256, epoch=10, lr=1e-5  |
|     BERT-wwm-ext      |  59.40  | 63.44 | 51.00 | 41.19 |  batch_size=8, length=256, epoch=10, lr=1e-5 |
|    RoBERTa-wwm-ext    |  -  | - | - | - |  batch_size=8, length=256, epoch=10, lr=1e-5 |
|   RoBERTa-wwm-large   |  -  | - | - | - |  batch_size=4, length=256, epoch=10, lr=1e-5 |
|     LSTM-seq2seq      |  41.80  | 46.48 | 30.48 | 22.00 |  batch_size=64, length=256, epoch=20, lr=1e-3 |


### **2. LCSTS 短文本摘要生成**

https://arxiv.org/abs/1506.05865

微博短文本及其作者给出的摘要，选取了数据集中 PART_II 部分作为训练集。

```
数据量：训练集(8,666)，验证集(1,000)，测试集(1,000)
示例：
{
    abstract: 新《环保法》元旦起实施企业违法可拘留责任人
    content: 正式实施的新《环保法》对企业惩治力度大大加强，受到罚款处罚，被责令改正，拒不改正的，依法作出处罚决定的行政机关可以自责令更改之日的次日起，按照原处罚数额按日连续处罚。企业违法可拘留责任人。
}
```

[运行结果](docs/lcsts.md)

|         模型          | Rouge-L | Rouge-1 | Rouge-2 | BLEU |               训练参数              |
| :-------------------: | :------: |:---: |:---: |:---: |  :--------------------------------: |
|      ALBERT-tiny      |  22.58  | 26.43 | 13.06 | 06.10 |  batch_size=16, length=128, epoch=10, lr=1e-5  |
|       BERT-base       |  -  | - | - | - |  batch_size=16, length=128, epoch=10, lr=1e-5  |
|     BERT-wwm-ext      |  -  | - | - | - |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|    RoBERTa-wwm-ext    |  -  | - | - | - |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|   RoBERTa-wwm-large   |  -  | - | - | - |  batch_size=8, length=128, epoch=8, lr=1e-5  |
|     LSTM-Seq2Seq      |  11.20  | 13.46 | 04.12 | 03.35 |  batch_size=64, length=128, epoch=20, lr=1e-3 |



### **3. WebQA 阅读理解问答**

百度 WebQA 问答数据集，取自科学空间的精简版本，只保留了阅读材料、问题和答案。

```
数据量：训练集(10,000)，验证集(1,000)，测试集(1,000)
示例：
{
    'question': '《齐民要术》的作者贾思勰是那个时代的人？',
    {
        'answer': '', 
        'passage': '在古代农学发展史上，贾思勰所著《齐民要术》是古代农学体系形成的标志。'
    },
    {
        'answer': '北魏',
        'passage': '答：1、中国古代农业百科全书《齐民要术》是北魏时期的中国杰出农学家贾思勰所著的一部综合性农书，也是世界农学史上最早的专著之一。'
     },
}
```

|         模型          | Rouge-L | Rouge-1 | Rouge-2 | BLEU |               训练参数              |
| :-------------------: | :------: |:---: |:---: |:---: |  :--------------------------------: |
|      ALBERT-tiny      |  -  | - | - | - |  batch_size=16, length=128, epoch=10, lr=1e-5  |
|       BERT-base       |  -  | - | - | - |  batch_size=16, length=128, epoch=5, lr=1e-5  |
|     BERT-wwm-ext      |  -  | - | - | - |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|    RoBERTa-wwm-ext    |  -  | - | - | - |  batch_size=16, length=128, epoch=5, lr=1e-5 |
|   RoBERTa-wwm-large   |  -  | - | - | - |  batch_size=8, length=128, epoch=8, lr=1e-5  |


### 4. CSL 关键词生成
Coming soom...

### 5. AFQMC 同义句生成
Coming soom...

## Contribution
Share your data set with community or make a contribution today! Just send email to chineseGLUE#163.com,

or join QQ group: 836811304

## Reference

[1] bert4keras：https://github.com/bojone/bert4keras

[2] 玩转Keras之seq2seq自动生成标题：[https://kexue.fm/archives/5861#seq2seq%E6%8F%90%E5%8D%87](https://kexue.fm/archives/5861#seq2seq提升)

