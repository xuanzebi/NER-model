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

##### Problem,Todo

- 在计算P、R、F1时，验证集测试集不截断、截断两种。（实体级别的F1） （token级别的F1）。
- 分batch，使用每一个batch的最大长度
- 字信息、词信息、字词信息混合 
- 使用pytorch的pad_packed_sequence和pack_padded_sequence进行实验对比。
- torchtext、allennlp、fast.ai



##### Motivation

1、如何获取更加高效的词、字信息表示。 不使用BERT的情况下。

2、设计能提出安全特征的网络结构。

3、