# CAIL 2022 论辩理解赛道 第一阶段数据集

## 数据说明

本赛道第一阶段所下发的数据集包括训练集 `train_*.jsonl` 和测试集 `test_*.jsonl`，其中训练集样本均来自涉及故意伤害罪的（刑事）案件，测试集则来自涉及海事海商纠纷的（民事）案件。所有数据均以 JSON Line 格式存储，每个样本一个 JSON 字符串；文件中出现的 Unicode 字符（中文等）以转义方式存储，可用 `pandas` 或 `datasets` 包直接读取。

数据集中包含的文件有：

- `train_text.jsonl`：包含了裁判文书所有对于辩诉双方辩护全文的数据，共 $16577$ 条。每条数据包含的字段内容如下：
  - `sentence_id`：句子 ID
  - `text_id`：裁判文书 ID
  - `category`：刑事、民事案件分类
  - `chapter`：刑事罪名或民事案由所在章节
  - `crime`：具体的刑事罪名或民事案由
  - `position`：诉方（sc）与辩方（bc）标志
  - `sentence`：句子文本

- `train_entry.jsonl`：包含了 $4183$ 对裁判文书中的互动论点对，每条数据包含的字段内容如下：
  - `id`：论点对 ID
  - `text_id`：裁判文书 ID
  - `category`：刑事、民事案件分类
  - `chapter`：刑事罪名或民事案由所在章节
  - `crime`：具体的刑事罪名或民事案由
  - `sc`：诉方论点
  - `bc_x`（$x=1,2,3,4,5$）：候选辩方论点，共五句
  - `answer`：正确辩方论点编号

- `test_text.jsonl`：同下发数据中的 `train_text.jsonl` 格式完全一致，共 $2639$ 条数据；

- `test_entry.jsonl`：同下发数据中的 `train_entry.jsonl` 格式基本一致，包含了 $859$ 对裁判文书中的互动论点对，但缺少相应的 `answer` 标签。

## 提交文件格式

**2022-08-29 更新：根据评测平台设置，请提交仅含 predict.jsonl 的 .zip 压缩文件！**

在第一阶段中，请提交经过模型预测的、对应 `test_entry.jsonl` 中样本的 `answer` 标签，以 JSON Line 文件格式存储，命名为 `predict.jsonl`。其中，文件每行包含单个样本的预测 JSON 字符串，单样本预测结果的 JSON 格式可参考 `predict_standard.json`。如果提交结果不符合规定的格式，将可能导致评测失败或准确率计算错误，影响评测成绩。
