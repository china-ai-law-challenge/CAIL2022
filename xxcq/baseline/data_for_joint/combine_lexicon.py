with open('cut_lexicon.txt','r',encoding='utf-8')as f1,open('lexicon_simple.txt','r',encoding='utf-8')as f2,\
    open('lexicon.txt','w',encoding='utf-8')as fout:
    list1=f1.read().split()
    list2=f2.read().split()
    list3=list(set(list1+list2))
    for i in list3:
        fout.write(i+'\n')