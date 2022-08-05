# -*-coding:utf-8-*-

# lead-3 baseline
import json


def cut_sentences(content):
    '''
    分句
    @param content: 未切分文本
    @type content: str
    @return: sentence list
    @rtype: list
    '''
    end_flag = ['?', '!', '.', '？', '！', '：', '。', '…']

    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        tmp_char += char

        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break

        if char in end_flag:
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char)
                tmp_char = ''

    return sentences


def gen_lead3_data(input_path, output_path):
    '''
    lead-3
    @param input_path: 测试集文件
    @type input_path: str
    @param output_path: 摘要文件
    @type output_path: str
    @return:
    @rtype:
    '''
    with open(input_path, "r", encoding="utf8") as jsonf:
        a = jsonf.readlines()
        lead3 = {}
        for index in range(len(a)):
            b = json.loads(a[index].strip())
            sentences = cut_sentences(b['text'])
            for senti in range(len(sentences)):
                text0 = sentences[senti]
                text1 = sentences[senti + 1]
                text2 = sentences[senti + 2]
                break
            if text0 == "" and text1 == "" and text2 == "":
                text0 = sentences[11]
                text1 = sentences[12]
                text2 = sentences[13]
            caselead3 = text0 + text1 + text2
            lead3["id"] = b["id"]
            lead3["summary"] = caselead3
            new = json.dumps(lead3, ensure_ascii=False)
            with open(output_path, "a", encoding="utf8") as jsonw:
                jsonw.write(new + "\n")
    return


if __name__ == "__main__":
    # get lead_3 results
    input_path = 'evaluate/evaluate.jsonl'
    output_path = 'data/result.jsonl'
    gen_lead3_data(input_path, output_path)
