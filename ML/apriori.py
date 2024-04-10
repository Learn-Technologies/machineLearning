import numpy as np
import pandas as pd
from apyori import apriori

##stored_data = pd.read_csv('')

##print(stored_data)
##print(stored_data.shape)

##stored_data = [['beer','cheese'],['beer','nuts']]

stored_data = [[1,2,3,4],[1,2,4],[1,5,6],[1,4,5],[2,4,5]]
association_rules = apriori(stored_data,min_support=0.5,min_confidence = 0.70,min_length=3,min_lift=1.2)
association_r = list(association_rules)

print(len(association_r))
print(association_r[0])
##print(association_r[1])
##print(association_r[2])
