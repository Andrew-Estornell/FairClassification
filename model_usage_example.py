
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle as pkl

for name in ['Models/adult_trainedModelInfo', 'Models/Data_1980_trainedModelInfo', 'Models/student-mat_trainedModelInfo', 'Models/lawschool2_trainedModelInfo']:
    print(name)
    d = pkl.load(open(name+'.pickle', 'rb'))

    for split in d.keys():
        d[split]['models'] = {clf_name: clf for clf_name, clf in d[split]['models'].items() if ('gamma' in clf_name and 'GB' not in clf_name) or 'gamma' not in clf_name}
        print(d[split]['models'].keys())

    with open(name+'(1).pickle', 'wb') as handle:
        pkl.dump(d, handle, protocol=pkl.HIGHEST_PROTOCOL)

