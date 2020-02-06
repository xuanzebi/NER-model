import json
import sys
package_dir_b = "/opt/hyp/NER/NER-model"
sys.path.insert(0, package_dir_b)

from util.util import compute_spans_bieos,compute_f1
data = json.load(open('/opt/hyp/NER/NER-model/saved_models/test_msra/result/test_pred_entity.txt', encoding='utf-8'))

ans = 0
true_entity = []
pred_entity = []
c = 0
d = 0

for i,(text,true,pred) in enumerate(data):
    a = text.split(' ')
    b = true.split(' ')
    c = pred.split(' ')
    if len(a) == len(b) == len(c):
        if len(a) < 200:
            print(len(a))
            c+=1
    else:
        # print(data[i])
        print(len(a),len(b),len(c))
        d+=1

print(c,d)