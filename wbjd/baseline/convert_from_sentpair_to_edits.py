# coding:utf-8

import sys
import Levenshtein
import json

src_path = sys.argv[1]
tgt_path = sys.argv[2]
sid_path = sys.argv[3]
output_path = sys.argv[4]

with open(src_path) as f_src, open(tgt_path) as f_tgt, open(sid_path) as f_sid:
    lines_src = f_src.readlines()
    lines_tgt = f_tgt.readlines()
    lines_sid = f_sid.readlines()
    assert len(lines_src) == len(lines_tgt) == len(lines_sid)

    for i in range(len(lines_src)):
        src_line = lines_src[i].strip().replace(' ', '')
        tgt_line = lines_tgt[i].strip().replace(' ', '')
        sid = eval(lines_sid[i].strip())['pid']
        edits = Levenshtein.opcodes(src_line, tgt_line)
        result = {'pid':sid,'target':[]}

        for edit in edits:
            if "。" in tgt_line[edit[3]:edit[4]]: # rm 。
                continue
            if edit[0] == "insert":
                result['target'].append({"pos": str(edit[1]), "ori": "", "edit": tgt_line[edit[3]:edit[4]], "type": "miss"})
            elif edit[0] == "replace":
                result['target'].append(
                    {"pos": str(edit[1]), "ori": src_line[edit[1]:edit[2]], "edit": tgt_line[edit[3]:edit[4]], "type": "spell"})
            elif edit[0] == "delete":
                result['target'].append({"pos": str(edit[1]), "ori": src_line[edit[1]:edit[2]], "edit": "", "type": "redund"})

        with open(output_path, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
        f.close()


