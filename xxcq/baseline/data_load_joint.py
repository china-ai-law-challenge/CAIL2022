import os
import numpy as np
import random
from utils import relnameToIdx, relIdxToName

VOCAB = ('<PAD>', 'O', 'I', 'B')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, TrgLen=None, TrgRels=None,
                        TrgPointers=None,SrcLen=None,nerTag=None,spanTag=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.TrgLen = TrgLen
        self.TrgRels=TrgRels
        self.TrgPointers=TrgPointers
        self.SrcLen = SrcLen
        self.nerTag = nerTag
        self.spanTag = spanTag


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, src_words,input_mask, segment_ids, TrgLen=None, TrgRels=None,
                        TrgPointers=None,SrcLen=None,nerTag=None,spanTag=None):
        self.input_ids = input_ids
        self.SrcWords = src_words
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.TrgLen = TrgLen
        self.TrgRels=TrgRels
        self.TrgPointers=TrgPointers
        self.SrcLen = SrcLen
        self.nerTag = nerTag
        self.spanTag = spanTag
# In[7]:


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError() 

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class relationsTextProcessor(DataProcessor):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None    
    
    def get_train_examples(self, filename,tagname,nername,datatype):
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
        alltext, tag_label, nertag, spantag = self._read_files(os.path.join(self.data_dir, filename),\
                                                         os.path.join(self.data_dir, tagname),\
                                                         os.path.join(self.data_dir, nername), datatype)
        # tag_label = self._tag_exchange(tag_label)
        return self._create_examples(alltext, tag_label, nertag, spantag, "train")
        
    def get_dev_examples(self, filename,tagname,nername,datatype):
        """See base class."""
        alltext, tag_label, nertag, spantag = self._read_files(os.path.join(self.data_dir, filename),\
                                                         os.path.join(self.data_dir, tagname),\
                                                         os.path.join(self.data_dir, nername), datatype)
        # tag_label = self._tag_exchange(tag_label)
        return self._create_examples(alltext, tag_label, nertag, spantag, "dev")
    
    def get_test_examples(self, filename,tagname,nername,datatype):
        alltext, tag_label, nertag, spantag = self._read_files(os.path.join(self.data_dir, filename),\
                                                         os.path.join(self.data_dir, tagname),\
                                                         os.path.join(self.data_dir, nername), datatype)
        # tag_label = self._tag_exchange(tag_label)
        return self._create_examples(alltext, tag_label, nertag, spantag, "test")

    # def get_labels(self,tagname):
    #     """See base class."""
    #     if self.labels == None:
    #         self.labels = list(pd.read_csv(os.path.join(self.data_dir, tagname),header=None)[0].values)
    #     return self.labels

    def _create_examples(self, text, label, nertags, spantags, set_type, labels_available=True):  #本来在这里加入other作为text_b
        """Creates examples for the training and dev sets."""
        examples = []
        # print(len(text),len(label),text[0])
        # with open('./data_for_joint/lexicon_simple.txt','r',encoding='utf-8') as fin:
        #     dic=fin.read().split()
        #     tb='$'.join(dic)    
        for i in range(len(text)):
            guid = "%s-%s" % (set_type, i)
            text_a = text[i]
            text_b=None #tb
            # text_b = other[i]
            if labels_available:
                trg_rels,trg_pointers = label[i]
                nertag=nertags[i]
                spantag=spantags[i]
                # flag=0
                # for trg_pointer in trg_pointers:
                #     if trg_pointer[0]>300 or trg_pointer[1]>300 or trg_pointer[2]>300 or trg_pointer[3]>300:
                #         flag=1
                #         break
                # if flag==1:
                #     continue
            else:
                labels = []
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, TrgLen=len(trg_pointers), TrgRels=trg_rels,
                        TrgPointers=trg_pointers,SrcLen=len(text_a),nerTag=nertag, spanTag=spantag))
        return examples
    
    def _read_files(self, path, tag_path, nertpath, datatype):
        fin = open(path, 'r', encoding='utf-8')
        tag_label = self._read_tag(tag_path, datatype)
        nerTag = self._read_nerTtag(nertpath,datatype)
        spanpath=nertpath.strip('ner')+'lexiconf'
        spanTag = self._read_spantag(spanpath,datatype)
        alltext = []

        line = fin.readline()
        while line:
            if len(line)>512:
                line=line[:512]
            alltext.append(line)
            line = fin.readline()
        fin.close()
        
        return alltext, tag_label, nerTag, spanTag
    
    def _read_json3(self, path, tag_path, n_gram=3):  #除了返回原始文本还返回了上下文
        fin = open(path, 'r', encoding='utf-8')
        tag_dic, tagname_dic = self._read_tag(tag_path)

        alltext = []
        tag_label = []
        othertext = []

        line = fin.readline()
        while line:
            d = json.loads(line)
            for isent in range(len(d)):
                alltext.append(str(d[isent]['sentence']))
                tag_label.append(self._getlabel(d[isent], tag_dic))
                if isent > (n_gram-1):
                    other = ""
                    for j in range(isent-n_gram, isent):
                        other += str(d[j]['sentence'])        
                    if isent < len(d) - n_gram:  
                        for j in range(isent+1, isent+1+n_gram):
                            other += str(d[j]['sentence'])
                    else:
                        for j in range(isent+1, len(d)):
                            other += str(d[j]['sentence'])
                    othertext.append(other)
                else:
                    other = ""
                    for j in range(isent):
                        other += str(d[j]['sentence'])
                    if isent < len(d) - n_gram:
                        for j in range(isent+1, isent+1+n_gram):
                            other += str(d[j]['sentence'])
                    else:
                        for j in range(isent+1, len(d)):
                            other += str(d[j]['sentence'])
                    othertext.append(other)
            line = fin.readline()
        fin.close()
        
        return alltext, tag_label, othertext

    
    def _read_nerTtag(self, nerpath,datatype):
        f = open(nerpath, 'r', encoding='utf-8-sig')
        nertags=[]
        line = f.readline()
        
        while line:
            linelist=line.split()
            if len(linelist)>512:
                linelist=linelist[:512]
            linel=[tag2idx[i] for i in linelist]
            nertags.append(linel)
            line = f.readline()
        f.close()
        return nertags#trg_rels,trg_pointers

    def _read_spantag(self, spanpath,datatype):
        f = open(spanpath, 'r', encoding='utf-8-sig')
        spantags=[]
        line = f.readline()
        
        while line:
            linelist=line.split()
            # linel=[tag2idx[i] for i in linelist]
            spantags.append(linelist)
            line = f.readline()
        f.close()
        return spantags#trg_rels,trg_pointers
    

    def _read_tag(self, tags_path,datatype):
        f = open(tags_path, 'r', encoding='utf-8-sig')
        tags=[]
        line = f.readline()
        while line:
            trg_line = line.strip()
            trg_rels = []
            trg_pointers = []
            parts = trg_line.split('|')
            if datatype == 1:
                random.shuffle(parts)
                    
            for part in parts:
                elements = part.strip().split()
                trg_rels.append(relnameToIdx[elements[4]])
                trg_pointers.append((int(elements[0]), int(elements[1]), int(elements[2]), int(elements[3])))
            tags.append((trg_rels,trg_pointers))
            # tagname_dic[len(tag_dic)] = line.strip()
            # tag_dic[line.strip()] = len(tag_dic)
            line = f.readline()
        f.close()
        
        return tags#trg_rels,trg_pointers
    
    def _tag_exchange(self, tag_label):
        tag_code = []
        for tag in tag_label:
            if tag == []:
                c = np.zeros(shape=(20, ), dtype=int)
                tag_code.append(c)
            else:
                c = np.zeros(shape=(20, ), dtype=int)
                for i in tag:
                    c[i] = 1
                tag_code.append(c)
        tag_code = np.array(tag_code)
        return tag_code



def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        # print(list(example.text_a),len(example.text_a),len(example.nerTag),max_seq_length)
        tokens_a = list(example.text_a.strip())#tokenizer.tokenize(example.text_a)
        tokens_b = None
        nertag=example.nerTag
        if example.text_b:
            tokens_b = list(example.text_b)#tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                nertag = nertag[:(max_seq_length - 2)]
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        nertag = [tag2idx['<PAD>']] + nertag + [tag2idx['<PAD>']]
        # print(len(nertag),tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        nertag += padding

        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(nertag) == max_seq_length
        # print(len(tokens_a),len(input_ids),len(example.nerTag))
        # labels_ids = []
        # for label in example.labels:
        #     labels_ids.append(float(label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              src_words=tokens_a,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              TrgLen=example.TrgLen, 
                              TrgRels=example.TrgRels,
                              TrgPointers=example.TrgPointers,
                              SrcLen=len(tokens_a),
                              nerTag=nertag,
                              spanTag=example.spanTag))  #example.SrcLen
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


