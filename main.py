import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

import string_to_features as stf

nrows = 1
fillchar:str = ' '

df = pd.read_csv('person_202111231235.csv' ,nrows=nrows, dtype={"surname":str,"firstname":str,"middlename":str})
#print(df.head())

df['surnamelen'] = df["surname"].str.len()
stf.set_df_column_features(df,"surname")
df['firstnamelen'] = df["firstname"].str.len()
stf.set_df_column_features(df,"firstname")
df['middlename'] = df["middlename"].str.len()
stf.set_df_column_features(df,"middlename")


# df['letter12'] = df["surnameX"].str.slice(0,2).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['letter23'] = df["surnameX"].str.slice(2,4).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['letter34'] = df["surnameX"].str.slice(4,6).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['letter67'] = df["surnameX"].str.slice(9,11).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['letter78'] = df["surnameX"].str.slice(11,13).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['letter1011'] = df["surnameX"].str.slice(17,20).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['letter1112'] = df["surnameX"].str.slice(21,24).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['nletter12'] = df["firstnameX"].str.slice(0,2).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['nletter23'] = df["firstnameX"].str.slice(2,4).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['nletter34'] = df["firstnameX"].str.slice(4,6).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['nletter67'] = df["firstnameX"].str.slice(9,11).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['nletter78'] = df["firstnameX"].str.slice(11,13).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['nletter1011'] = df["firstnameX"].str.slice(17,20).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['nletter1112'] = df["firstnameX"].str.slice(21,24).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['oletter12'] = df["middlenameX"].str.slice(0,2).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['oletter23'] = df["middlenameX"].str.slice(2,4).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['oletter34'] = df["middlenameX"].str.slice(4,6).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['oletter67'] = df["middlenameX"].str.slice(9,11).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['oletter78'] = df["middlenameX"].str.slice(11,13).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['oletter1011'] = df["middlenameX"].str.slice(17,20).transform(lambda x: 32767*ord(x[0])+ord(x[1]))
# df['oletter1112'] = df["middlenameX"].str.slice(21,24).transform(lambda x: 32767*ord(x[0])+ord(x[1]))


df['surname'] = df['surname'].str.pad(20,'right',' ')
df['firstname'] = df['firstname'].str.pad(20,'right',' ')
df['middlename'] = df['middlename'].str.pad(20,'right',' ')


print(df.head())
exit()


featuresToConsider = ['surnamelen','letter12','letter23','letter34','letter67','letter78','letter1011','letter1112','nletter12','nletter23','nletter23','nletter34','nletter67','nletter78','nletter1011','nletter1112','letter1112','oletter12','oletter23','oletter23','oletter34','oletter67','oletter78','oletter1011','oletter1112']
# featuresToConsider = ['lettersvector03']

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.0001),max_features= 1.0)
model.fit(df[featuresToConsider])


df['scores']=model.decision_function(df[featuresToConsider])
df['anomaly']=model.predict(df[featuresToConsider])
df.head()

anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)

df.loc[df['anomaly'] == -1].to_csv('person_anomaly_result',)