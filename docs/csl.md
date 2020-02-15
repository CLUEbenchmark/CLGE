
## CSL 运行结果

在每一轮训练结束后会输出 example.tsv 的预测结果，
本页面展示的是各模型在最后一轮训练的表现。

1. BERT_base 

![image](../docs/images/csl/bert_base.png)

2. BERT_wwm_ext 

![image](../docs/images/csl/bert_wwm_ext.png)


3. ALBERT_tiny

![image](../docs/images/csl/albert_tiny.png)

4. RoBERTa_wwm_ext

![image](../docs/images/csl/roberta_wwm_ext.png)

5. RoBERTa_wwm_large

```$xslt
- 2002s - loss: 0.1047
生成标题: 城市交通场景下基于车辆节点认知交互的路由算法
生成标题: 近场信源自适应近场迭代自适应算法
生成标题: 基于活跃时间分组的软件众包工人选择机制
生成标题: 基于弱变异准则的测试用例约简方法
生成标题: 一种改进的全色锐化算法及其在ms图像中的应用
rouge-l scores:  0.6810021277286873
```

6. LSTM_Seq2Seq
```$xslt
- 86s 86ms/step - loss: 0.1826 - val_loss: 13.1699
生成标题:  车载网环境下路由算法
生成标题:  基于迭代自适应方法的控制参数预测
生成标题:  基于活动分组的软件选择机制的研究与设计论
生成标题:  基于测变异变异常包结构的测试用例约简方法
生成标题:  基于稀疏表示的递归算法
rouge-l scores:  0.40666066996461137
#实验峰值为43,但是示例只展示完整录得的实验结果

```
