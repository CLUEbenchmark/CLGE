# CLGE
Chinese Language Generation Evaluation 中文生成任务基准测评

为中文生成任务提供数据集、基准(预训练)模型和排行榜。


## 一键运行

```
1、克隆项目 
   git clone --depth=1 https://github.com/CLUEbenchmark/CLGE.git
2、下载任务数据集并解压至 CLGEdataset，进入到相应任务的目录
   例如运行 csl 任务
       cd CLGE/tasks/csl  
3、运行对应任务的脚本: 会自动安装依赖、下载模型并开始运行。
       sh run_bert_base.sh
   如运行 sh run_bert_base.sh 会开始 csl 任务在 BERT_base 上的训练
```

## 文本生成方法

**1. LSTM-seq2seq**

参考：苏剑林. (2018, Sep 01). 《玩转Keras之seq2seq自动生成标题 》[Blog post]. Retrieved from https://spaces.ac.cn/archives/5861

**2. BERT-UNILM 方案**

参考：苏剑林. (2019, Sep 18). 《从语言模型到Seq2Seq：Transformer如戏，全靠Mask 》[Blog post]. Retrieved from https://spaces.ac.cn/archives/6933

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
[百度网盘](https://pan.baidu.com/s/1FFG_s8z47r6e7EqoRtfIrw) 提取码: u6mc

[中文科学文献数据(CSL)](https://github.com/P01son6415/chinese-scientific-literature-dataset)，选取 10k 条计算机相关领域论文及其标题作为训练集。

```
数据量：训练集(3,000)，验证集(500)，测试集(500)
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
|    RoBERTa-wwm-ext    |  58.99  | 63.23 | 50.74 | 41.31 |  batch_size=8, length=256, epoch=10, lr=1e-5 |
|   RoBERTa-wwm-large   |  -  | - | - | - |  batch_size=4, length=256, epoch=10, lr=1e-5 |
|     LSTM-seq2seq      |  41.80  | 46.48 | 30.48 | 22.00 |  batch_size=64, length=256, epoch=20, lr=1e-3 |


## Contribution
Share your data set with community or make a contribution today! Just send email to chineseGLUE#163.com,

or join QQ group: 836811304

## Reference

[1] bert4keras：https://github.com/bojone/bert4keras

[2] 玩转Keras之seq2seq自动生成标题：[https://kexue.fm/archives/5861#seq2seq%E6%8F%90%E5%8D%87](https://kexue.fm/archives/5861#seq2seq提升)

