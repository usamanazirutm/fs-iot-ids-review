import pandas as pd, re
d=pd.read_csv('/mnt/user-data/uploads/merged_records.csv')
d['abstract']=d['abstract'].fillna(''); d['title']=d['title'].fillna('')
T=d.title.str.lower(); A=d.abstract.str.lower(); TA=T+' '+A
iot=r"\b(iot|iiot|iomt|iov|internet of (medical )?things|internet of vehicles|industrial internet)\b"
ids=r"(intrusion|attack|anomal|botnet|ddos|malware|threat|malicious)"
fs=r"(feature[s]? selection|feature[s]? reduction|selected features|dimensionality reduction|feature[s]? subset|feature optimi[sz]ation|feature engineering|feature ranking|optimal features|relevant features|feature elimination)"
opt=r"(metaheuristic|meta-heuristic|swarm|pso|particle swarm|grey wolf|gray wolf|gwo|whale|genetic algorithm|\bga\b|optimi[sz]er|optimization algorithm|harris hawks|firefly|bat algorithm|cuckoo|ant colony|jaya|sine cosine|salp|equilibrium optimi|chimp|dragonfly|bee colony|evolutionary)"
rev=r"(survey|review|overview|state[- ]of[- ]the[- ]art|taxonomy|systematic|bibliometric|tutorial|comparative study of existing)"
emp=r"(nsl|kdd|unsw|cic|bot-?iot|ton[_ -]?iot|n-?baiot|edge-?iiot|iotid|x-?iiotid|dataset|accuracy|f1|precision|detection rate|%)"
reason=[]
for i in range(len(d)):
    t,a,ta=T[i],A[i],TA[i]; ty=str(d.type[i]).lower(); v=str(d.venue[i]).lower(); vt=str(d.venue_type[i]).lower()
    if re.search(rev,t) or 'review' in ty: r='EC4 review/survey'
    elif 'arxiv' in v or vt=='repository' or 'preprint' in ty or 'book' in ty or 'thesis' in t: r='EC4 non-peer-reviewed/book'
    elif not re.search(iot,ta): r='EC1 not IoT'
    elif not re.search(ids,ta): r='EC1 not IDS'
    elif not (re.search(fs,t) or (re.search(opt,t) and re.search(fs,a))): r='EC2 FS not a primary contribution'
    elif a and not re.search(emp,a): r='EC3 no empirical evaluation'
    else: r='INCLUDE'
    reason.append(r)
d['ta_decision']=reason
print(d.ta_decision.value_counts())
inc=d[d.ta_decision=='INCLUDE']
print(inc.year.value_counts().sort_index()); print('oa',(inc.oa_url.fillna('')!='').sum(),'doi',(inc.doi.fillna('')!='').sum(),'noabs',(inc.abstract=='').sum())
d.to_csv('screened.csv',index=False)
