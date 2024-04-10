import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TranscationEncoder
dataset = [['milk','onion','nutmug''beans','eggs','yogurt'],
           ['dill','onion','nutmug''beans','eggs','yogurt'],
           ['milk','apple','kb','eggs'],
           ['milk','unicorn','corn','kb'],
           ['corn','onoion','onoion','kb','icecream','eggs']]
te = TranscationEncoder()
te_ary = te.fit(dataset).transform(dataset)
df=pd.DataFrame(te_ary,columns=te.columns_)
print(df)
rules = fpgrowth(df,min_support = 0.7,use_colnames = True)
print(rules)
