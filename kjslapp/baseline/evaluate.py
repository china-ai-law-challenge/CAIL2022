import argparse
import pandas as pd
import sys
from sklearn.metrics import f1_score
import os
e_min = 0.0000001
def rationale_acc(selected_id_a, Case_a_rational, selected_id_b, Case_b_rational, pre_relation, relations):
    right_a = 0
    for i in selected_id_a:
        for j in Case_a_rational:
            if i == j:
                right_a += 1
    if len(selected_id_a) == 0:
        P_a = 0
    else:
        P_a = right_a/len(selected_id_a)
    
    if len(Case_a_rational)==0:
        R_a = 0
    else:
        R_a = right_a/len(Case_a_rational)

    if P_a + R_a <= e_min:
        F_a = 0
    else:
        F_a = 2*P_a*R_a/(P_a+R_a)

    right_b = 0
    for i in selected_id_b:
        for j in Case_b_rational:
            if i == j:
                right_b += 1
    if len(selected_id_b) == 0:
        P_b = 0
    else:
        P_b = right_b/len(selected_id_b)
    
    if len(Case_b_rational)==0:
        R_b = 0
    else:
        R_b = right_b/len(Case_b_rational)

    if P_b+R_b <= e_min:
        F_b = 0
    else:
        F_b = 2*P_b*R_b/(P_b+R_b)

    right_c = 0
    for i in pre_relation:
        for j in relations:
            if i == j:
                right_c += 1
    if len(pre_relation) == 0:
        P_c = 0
    else:
        P_c = right_c / len(pre_relation)
    if len(relations) == 0:
        R_c = 0
    else:
        R_c = right_c / len(relations)
    if P_c + R_c <= e_min:
        F_c = 0
    else:
        F_c = 2 * P_c * R_c / (P_c + R_c)
    return (F_a + F_b + F_c)/3


base_dir, summit_file = sys.argv[1], sys.argv[2]
gold_file = os.path.join(base_dir, "competition_stage_1_test_truth.txt") # competition_stage_2_test_truth.txt competition_stage_3_test_truth.txt

gold_data = pd.read_csv(gold_file, sep='\t')
pred_data = pd.read_csv(summit_file, sep='\t')
pred_data.set_index(["id"], inplace=True)
pred_labels = []
truth_labels = []
relations_list = []
for i in range(len(gold_data)):
    id = gold_data.loc[i]['id']
    Case_a_rational = eval(gold_data.loc[i]['Case_A_rationales'])
    Case_b_rational = eval(gold_data.loc[i]['Case_B_rationales'])
    relations = eval(gold_data.loc[i]['relation'])
    truth_labels.append(gold_data.loc[i]['label'])
    selected_id_a = eval(pred_data.loc[id]['Case_A_rationales'])
    selected_id_b = eval(pred_data.loc[id]['Case_B_rationales'])
    pre_relation = eval(pred_data.loc[id]['relation'])
    pred_labels.append(pred_data.loc[id]['label'])
    relations_score = rationale_acc(selected_id_a, Case_a_rational, selected_id_b, Case_b_rational, pre_relation, relations)
    relations_list.append(relations_score)
f1_macro = f1_score(truth_labels, pred_labels, average='macro')
print(0.5*f1_macro+0.5*(sum(relations_list)/len(relations_list)))













