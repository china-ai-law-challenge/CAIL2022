###数据来源

刑事判决书涉毒类案件中的情节描述段落，共1750条。

### 数据用途

用于命名实体识别和关系抽取任务。

### 数据格式

json文件存储，格式参考NYT数据集格式，包括articleId（文章编号），sentID（段落编号），sentText（段落文本），entityMentions（实体列表），relationMentions（关系列表）等字段，其中entityMentions包括start（实体左边界）、end（实体右边界）、text（实体内容）、label（实体类型）；relationMentions包括e1start（头实体左边界）、e1text（头实体内容）、e2start（尾实体左边界）、e2text（尾实体内容）、label（关系类型）。