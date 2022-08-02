# CAIL 2022 论辩理解赛道 基线模型

## 模型简介

本赛道的基线模型基于 BERT 实现，并融合了共有词汇量的启发式预测。

本基线模型仅供各位选手参考。

## 运行方式

### 训练

将训练集文件 `train_*.jsonl` 存储于 `data` 目录内，运行 `train.py` 即可在训练集上精调 BERT 模型。精调后的 BERT 模型参数存储于新建的 `model` 目录内。

### 测试

将测试集文件 `test_*.jsonl` 存储于 `data` 目录内，并确保 `model` 目录中存放了精调后的 BERT 模型参数（可以通过上述训练过程得到）。运行 `main.py` 即可在测试集上预测结果，并在新建的 `output` 目录下生成符合提交要求的 `predict.jsonl` 文件。

## 运行要求

请使用 Python 3.7 或以上版本运行训练与测试脚本，并参考 `requirements.txt` 配置 Python 环境。简单而言，脚本至少需要以下依赖项：

- `python>=3.7`
- `jieba`
- `pytorch`
- `transformers`
