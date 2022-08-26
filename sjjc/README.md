# CAIL2022 —— 事件检测

该项目为 **CAIL2022——事件检测** 赛道的代码和提交说明

报名地址 [[link]](http://cail.cipsc.org.cn/task1.html?raceID=1&cail_tag=2022)， 数据集下载 [[link]](https://cloud.tsinghua.edu.cn/d/6e911ff1286d47db8016/)，CAIL2022官网[[link]](http://cail.cipsc.org.cn/index.html)

选手交流QQ群：237633234

## 数据说明

本任务所使用的数据集来自于论文 [LEVEN: A Large-Scale Chinese Legal Event Detection Dataset](https://aclanthology.org/2022.findings-acl.17.pdf)。

关于任务的详细信息，请参考 [thunlp/LEVEN](https://github.com/thunlp/LEVEN) ：
 - 数据集的背景
 - 数据集的标注手册
 - Baseline代码实现
 - 事件检测在 [判决预测](http://cail.cipsc.org.cn/task_summit.html?raceID=2&cail_tag=2018) 和 [类案检索](http://cail.cipsc.org.cn/task3.html?raceID=3&cail_tag=2022) 两个Legal AI 下游任务中的应用
 	- 实现细节参考[论文](https://aclanthology.org/2022.findings-acl.17.pdf) 5.5节
 	- 代码参考 [thunlp/LEVEN/Downstreams](https://github.com/thunlp/LEVEN/tree/main/Downstreams)

数据集格式信息如下：

`train.jsonl` 为训练集，格式如下

```json
{
    "title": "李某波犯猥亵儿童罪一审刑事判决书",    //标题
    "id": "a6f3b705d93e441dbd3a29365e854193",  //文档id
    "crime": "猥亵儿童罪",                      //罪名
    "case_no": "（2014）梅兴法刑初字第344号",    //案号
    "content": [ //文档的内容
    		{
    		 "sentence":"...", //句子内容
    		 "tokens": ["...", "..."] //分词后的句子
			}
	],
	"events":[ //事件标注
        {
            "id": '0fd7970c76d64c5d9ac1c015609c028b', //事件id
            "type": '租用/借用',                       //事件类型
            "type_id": 22,                          //事件类型id（1-108）
            "mention":[ //事件提及
            	{
              		"trigger_word": "租住", //触发词
              		"sent_id": 0, //句子id
              		"offset": [41, 42], //句子中的位置
					"id": "2db165c25298aefb682cba50c9327e4f", //这个事件提及的id
              	}
             ]
        }
    ],
	"negative_triggers":[//触发词负样本
        {
            "trigger_word": "出生",
            "sent_id": 0,
            "offset": [21, 22],
			"id": "66571b43dcf9461cb7ce979875fc9287",
        }
    ]
}
```

`test_stage1.jsonl` 为第一阶段的测试集，格式与 `train.jsonl` 一致，但并未提供事件标注，选手需要为`candidates` 字段中的每一个candidate预测事件类型，评测方式参考 [evaluate.py](https://github.com/thunlp/LEVEN/blob/main/evaluate.py) 

```json
{
    "title": "姚均飞强奸罪一审刑事判决书",           //标题
    "id": "9720823b46ea4efebb52539f2016d8b8",    //文档id
    "crime": "强奸罪",                           //罪名
    "case_no": "（2018）渝0154刑初280号",          //案号
    "content": [ //内容
    		{
    		 "sentence":"...", //句子id
    		 "tokens": ["...", "..."]  //分词后的句子
			}
	],
	"candidates":[ //事件触发词的候选集合，选手需要判断每个触发词所对应的事件类型
        {
            "trigger_word": "认识",
            "sent_id": 0,
            "offset": [28, 29],
			"id": "f3f93191743a4c63966f5c48f8f6383c",  //候选词id
        }
    ]
}
```

# Baseline模型

我们提供了基于BERT的baseline模型 [Baseline](https://github.com/thunlp/LEVEN/tree/main/Baselines/BERT%2BCRF)

运行`run_train.sh` 进行训练

运行`run_infer.sh` 进行预测并生成提交格式

## 提交的文件格式及组织形式

针对三个阶段，选手需要分别提交 `results1.jsonl`  `results2.jsonl` `results3.jsonl` 的压缩后的zip文件

`results1.jsonl` 的每一行是一个json，字段如下

```json
{
    "id": "40f9f0f02256402baab5a9c344c95c8f",    ////文档id
    "predictions": [
        {
            "id": "260f26bd1cf24c98bb18b85cea5ffcfc",   //候选词id
            "type_id": 0        //预测事件类型id (0-108)
        },
        {
            "id": "df6f2afc6f7a433dbbe2f54afac48bad",
            "type_id": 0
        },
     		...
     ]
}
```

