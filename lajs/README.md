# CAIL2022 —— 类案检索

该项目为 **CAIL2022——类案检索** 的代码和模型提交说明

## 任务介绍

该任务为面向中国刑事案件的类案检索。具体地，给定若干个查询案例（query），每一个查询案例各自对应一个大小为100的候选案例（candidate）池，要求从候选案例池中筛选出与查询案例相关的类案。类案相似程度划分为四级（从最相关：3 到完全不相关：0），判定标准详见[类案标注文档](https://docs.qq.com/doc/DU1FTbWZtcnpBVnhx)。每个查询案例最终的提交形式为对应的100个候选案例的排序列表，预测越相似的案例排名越靠前。

## 数据集说明

本任务所使用的数据集来自于裁判文书网公开的裁判文书。其中初赛阶段全部数据、复赛阶段训练集、封测阶段训练集均使用公开的中文类案检索数据集[LeCaRD](https://github.com/myx666/LeCaRD)。以初赛阶段测试数据集为例，文件结构如下：

```
input
├── candidates
│   ├── 111
│   ├── 222
│   ├── 333
│   ├── 444
│   └── 555
└── query.json

6 directories, 1 file
```

其中，input是输入文件根目录，包含了两个部分：`query.json`和`candidates/`。如果是训练集，在根目录下还会有一个label文件：`label_top30_dict.json`。
`query.json`包括了该阶段所有的query，每个query均以字典格式进行存储。下面是一个query的示例：

```
{"path": "ba1a0b37-3271-487a-a00e-e16abdca7d83/005da2e9359b1d71ae503d98fba4d3f31b1.json", "ridx": 1325, "q": "2016年12月15日12时许，被害人郑某在台江区交通路工商银行自助ATM取款机上取款后，离开时忘记将遗留在ATM机中的其所有的卡号为62×××73的银行卡取走。后被告人江忠取钱时发现该卡处于已输入密码的交易状态下，遂分三笔取走卡内存款合计人民币（币种，下同）6500元。案发后，被告人江忠返还被害人郑某6500元并取得谅解。", "crime": ["诈骗罪", "信用卡诈骗罪"]}
```

query的各个字段含义如下：
- **path**：查询案例对应的判决书在原始数据集中的位置（在本次比赛中不重要，可以忽略）
- **ridx**：每个查询案例唯一的ID
- **q**：查询案例的内容（只包含案情描述部分）
- **crime**：查询案例涉及的罪名

`candidates/`下有若干个子文件夹，每个子文件夹包含了一个query的全部100个candidates；子文件夹名称对应了其所属query的**ridx**。这100个candidate分别以字典的格式单独存储在json文件中，下面是一个candidate的示例：

```
{"ajId":"dee49560-26b8-441b-81a0-6ea9696e92a8","ajName":"程某某走私、贩卖、运输、制造毒品一案","ajjbqk":" 公诉机关指控，2018年3月1日下午3时许，被告人程某某在本市东西湖区某某路某某工业园某某宾馆门口以人民币300元的价格向吸毒人员张某贩卖毒品甲基苯丙胺片剂5颗......","pjjg":" 一、被告人程某某犯贩卖毒品罪，判处有期徒刑十个月......","qw":"湖北省武汉市东西湖区人民法院 刑事判决书 （2018）鄂0112刑初298号 公诉机关武汉市东西湖区人民检察院。 被告人程某某......","writId": "0198ec7627d2c78f51e5e7e3862b6c19e42", "writName": "程某某走私、贩卖、运输、制造毒品一审刑事判决书"}
```

candidate的各个字段含义如下：
- ajId：候选案例的ID（可忽略）
- ajName：案例的名称
- ajjbqk：案件基本情况
- cpfxgc：裁判分析过程
- pjjg：法院的判决结果
- qw：判决书的全文内容
- writID：判决书的ID（可忽略）
- writName是判决书的名称（可忽略）

一些注意事项：
- 查询案例的ID（ridx）可能为正整数（例如1325）或者负整数（例如-991），但是本次比赛中并不加以区分，只需要看作唯一对应的ID序号，其数值不具有任何含义。
- 根据组委会要求，初赛阶段仅使用25%的LeCaRD数据作为训练集和测试集；复赛阶段将使用LeCaRD全集作为训练接；复赛阶段和封闭评测阶段都将使用不公开的数据进行评测，但是数据结构、数据类型与前两个阶段保持一致。
- LeCaRD数据集的论文原文。如果您在CAIL评测中或者评测后引用LeCaRD数据集发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了LeCaRD数据集”，并按如下格式引用：
```
@inproceedings{ma2021lecard,
  title={LeCaRD: a legal case retrieval dataset for Chinese law system},
  author={Ma, Yixiao and Shao, Yunqiu and Wu, Yueyue and Liu, Yiqun and Zhang, Ruizhe and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2342--2348},
  year={2021}
}
```

## 提交的文件格式
本次比赛只需要提交检索结果文件，**必须**命名为`prediction.json`，结果以字典格式存储。下面是一个`prediction.json`的示例（数字仅供示意）：

```
{ "111": [12, 7, 87, ... ], "222": [8765, 543, 777, ... ], "-32": [99, 342, 30, ...] ... }
```

请注意：提交结果的字典务必包括**全部**query的`ridx`作为key，并且由于本次类案检索任务的评测指标是NDCG@30，所以每个key下对应的列表长度至少为30。提交格式的错误将会直接影响评测结果！

在`baseline/`文件夹下，有一个简单的bm25模型作为参考；在初赛数据集上，该模型的NDCG@30为0.7903。

## 评测指标

类案检索任务的评测指标为[NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)@30，也即结果列表前30位candidate的NDCG（Normalized Discounted Cumulative Gain）值。复赛阶段至多可提交三个模型，以最高NDCG@30分数作为复赛阶段成绩；封闭评测阶段。参赛者可自主从复赛阶段提交的三个模型中指定任意一个模型为最终模型，其在封测数据集上的NDCG@30分数计为封测阶段成绩。

最终队伍分数 = 0.3 * 复赛成绩 + 0.7 * 封测成绩

## 常见问题Q&A（持续更新）：