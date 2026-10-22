import pandas as pd, numpy as np
df=pd.read_pickle('extracted.pkl')
fam_fix={'R0125':'Wrapper-meta','R0136':'Hybrid','R0157':'Hybrid','R0162':'Embedded','R0204':'Embedded','R0213':'Transformation','R0244':'Embedded','R0247':'Hybrid','R0268':'Wrapper-meta','R0364':'Transformation','R0386':'Hybrid','R0442':'Hybrid','R0534':'Filter','R0653':'Hybrid','R0743':'Embedded','R0789':'Transformation','R0797':'Wrapper-meta','R0817':'Hybrid','R0821':'Filter','R0861':'Embedded','R0885':'Embedded','R0889':'Wrapper-meta','R0915':'Hybrid','R0919':'Wrapper-meta','R0925':'Filter','R0931':'Wrapper-meta','R0932':'Wrapper-meta','R1045':'Embedded','R1055':'Hybrid','R1059':'Transformation','R1102':'Embedded','R1290':'Hybrid','R1447':'Wrapper-meta','R1475':'Other','R1507':'Wrapper-seq','R1527':'Wrapper-meta','R1262':'Other'}
for k,v in fam_fix.items(): df.loc[df.rec_id==k,'Primary FS family']=v
inc=df[df.eligibility=='INCLUDE'].copy().reset_index(drop=True)
inc.insert(0,'ID',[f'S{i:03d}' for i in range(1,len(inc)+1)])
Y=lambda c: (inc[c]=='Y')
# QA
qa=pd.DataFrame({'ID':inc.ID})
qa['QA1']=np.where(inc['Primary FS family']!='Other', np.where(inc._params=='Y',1,0.5),0)
qa['QA2']=np.where(Y('FS fitted on training data only'),1,0)
qa['QA3']=inc['Validation protocol'].map({'k-fold CV':1,'Single hold-out':0.5,'Not stated':0})
qa['QA4']=np.where(Y('Baseline without FS'),1,np.where(inc._cmpfs=='Y',0.5,0))
qa['QA5']=(Y('FAR reported').astype(int)+Y('Macro/per-class F1 reported').astype(int))/2
c=Y('FS time reported').astype(int)+Y('Inference latency reported').astype(int)+Y('Memory reported').astype(int)+Y('Energy reported').astype(int)
qa['QA6']=np.where(c>=2,1,np.where(c==1,0.5,0))
qa['QA7']=np.where((inc['No. of datasets']>=2)|Y('Cross-dataset test'),1,0)
qa['QA8']=np.where(Y('Code/data available'),1,0)
qa['Total']=qa[[f'QA{i}' for i in range(1,9)]].sum(1)
print(qa.Total.describe()); print(qa.Total.value_counts().sort_index())
inc.to_pickle('inc.pkl'); qa.to_pickle('qa.pkl')
