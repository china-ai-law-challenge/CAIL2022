# coding:utf-8

import sys
import tokenization
from tqdm import tqdm

tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)

for line in tqdm(sys.stdin):
    line = eval(line.strip())['source']
    items = line.strip()

    line = tokenization.convert_to_unicode(items)
    if not line:
        print()
        continue

    tokens = tokenizer.tokenize(line)
    print(' '.join(tokens))
