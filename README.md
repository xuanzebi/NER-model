## Cyber data

|      数据集      | Train   | Test   | Dev    |
| :--------------: | ------- | ------ | ------ |
|  **Sentences**   | 38200   | 4776   | 4777   |
| 全为O的Sentences | 10043   | 1243   | 1248   |
|    **Chars**     | 2210397 | 278482 | 270178 |
|   **Entities**   | 113592  | 14623  | 13805  |
| VUL_ID(漏洞编号) | 265     | 30     | 25     |
|    SW(软件名)    | 5397    | 647    | 719    |
|   ORG(组织名)    | 14557   | 1861   | 1727   |
|    PER(人名)     | 9944    | 1355   | 1291   |
|    LOC(地名)     | 18958   | 2467   | 2290   |
|   RT(相关术语)   | 64471   | 8263   | 7753   |



| 句子长度 | Train | Test | Dev  |
| -------- | ----- | ---- | ---- |
| mean     | 58    | 58   | 57   |
| min      | 3     | 2    | 2    |
| max      | 5569  | 3246 | 1380 |
| 98%      | 198   | 194  | 201  |

#### Problem,Todo

- 在计算P、R、F1时，验证集测试集不截断√、**截断**两种。（实体级别的F1√） （token级别的F1√）。
- 分batch，使用每一个batch的最大长度
- 字信息√、词信息、字词信息混合           基于字信息的模型一般优于基于词信息的RNN模型，主要是基于词的模型有两个问题，OOV和数据稀疏（词表过大）
- 使用pytorch的pad_packed_sequence和pack_padded_sequence进行实验对比。
- torchtext、allennlp、fast.ai
- 不同的词向量的对比效果，是否冻结词向量
- 构建词表时不加入测试集
- 将预测结果保存
- 不同的dropout rate, 0.3 0.5的往往效果比较好
- BERT  微调和不微调两种
- CNN 相关模型



##### Motivation

1、如何获取更加高效的词、字信息表示。 不使用BERT的情况下。

2、设计能提出安全特征的网络结构。

##### 实验

1、BiLSTM-CRF 基于字符

2、BERT

3、lattice lstm

4、双指针 重要

5、CNN 作为encoder提取 位置局部信息，来融合到RNN或者BERT中   

6、RNN之后接一层CNN



#### 记录

1、Tencent词向量  “中文，《”  替换成英文的

```
词表大小 4660
unk words 数量为9,unk 的word（set）数量为9
Embedding:
     pretrain word:8824330, vocab: 4660, prefect match:4625, case_match:26, oov:9, oov%:0.0019313304721030042
['</PAD>', '</UNK>', '，', '《', 'I-SW', 'I-PER', '\ufeff海', 'I-RT', '\u202c']
Found embeddings for 99.85% of vocab
Found embeddings for 26 of case match
Found embeddings for 0 of title match
Found embeddings for  96.33% of all text
[('\ufeff海', 1), ('\u202c', 1), ('I-RT', 2), ('I-PER', 3), ('I-SW', 10), ('《', 1472), ('，', 96230)]
```

2、 cc.zh.300.vec

```
unk words 数量为210,unk 的word（set）数量为210
Embedding:
     pretrain word:2000000, vocab: 4660, prefect match:4450, case_match:0, oov:210, oov%:0.045064377682403435

Found embeddings for 95.53% of vocab
Found embeddings for 0 of case match
Found embeddings for 0 of title match
Found embeddings for  99.98% of all text
```

3、Baidu Encyclopedia 百度百科 Word + Character + Ngram

https://github.com/Embedding/Chinese-Word-Vectors

```

```
