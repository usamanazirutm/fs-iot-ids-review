import pandas as pd, gzip, json, re, numpy as np
meta=pd.read_csv('screened.csv').set_index('rec_id')
rows=[json.loads(x) for x in gzip.open('/mnt/user-data/uploads/fulltext_jsonl.gz','rt',encoding='utf-8')]

def body_of(t):
    tl=t.lower()
    # cut references
    m=list(re.finditer(r'\n\s*(references|bibliography|reference list)\s*\n',tl))
    if m: tl=tl[:m[-1].start()]
    # cut related work
    a=re.search(r'\n\s*(\d+\.?\s*|[ivx]+\.\s*)?(related works?|literature review|literature survey|background and related work|review of literature|previous work|prior work)\s*\n',tl)
    if a:
        b=re.search(r'\n\s*(\d+\.?\s*|[ivx]+\.\s*)?(proposed|methodology|materials and methods|methods|system model|research method|the proposed|framework|approach|preliminar)[^\n]{0,60}\n',tl[a.end():])
        if b and b.start()<60000: tl=tl[:a.start()]+tl[a.end()+b.start():]
    tl=re.sub(r'\s*\n\s*',' ',tl); tl=re.sub(r'\s*-\s*','-',tl); tl=re.sub(r' +',' ',tl)
    return tl

R=lambda p,s: bool(re.search(p,s))
def body_of_orig(t):
    m=list(re.finditer(r'\n\s*(References|REFERENCES|Bibliography)\s*\n',t))
    return t[:m[-1].start()] if m else t
def near(p,s,w=80): return bool(re.search(r'(?:'+p+r')[^.]{0,'+str(w)+r'}(feature|attribute)|(feature|attribute)[^.]{0,'+str(w)+r'}(?:'+p+')',s))
META=r"(particle swarm|\bpso\b|grey wolf|gray wolf|\bgwo\b|whale optimi|\bwoa\b|genetic algorithm|\bga\b-based|\bnsga|harris hawk|firefly|bat algorithm|cuckoo|ant colony|\baco\b|jaya|sine cosine|salp swarm|equilibrium optimi|chimp|dragonfly|bee colony|marine predator|slime mould|aquila|seagull|tunicate|sparrow|coati|osprey|pelican|zebra|dolphin|crow search|moth|flower pollination|simulated annealing|differential evolution|tabu search|arithmetic optimi|reptile search|snake optimi|metaheuristic|meta-heuristic|swarm intelligence|bio-inspired|nature-inspired|evolutionary algorithm|white shark|golden jackal|gazelle|walrus|hippopotamus|horse herd|butterfly|squirrel|lion optimi|wild horse|african vulture|bald eagle|honey badger|mayfly|elephant herding|grasshopper|krill herd|cat swarm|kookaburra|chameleon|jellyfish|black widow|mantis|beluga|archimedes|gorilla|dung beetle|hiking optimi|bonobo|tasmanian|nutcracker)"
FILT=r"(mutual information|information gain|chi-square|chi square|chi2|\bχ2|pearson|correlation[- ]based|correlation coefficient|anova|f-test|fisher score|relieff?|gain ratio|mrmr|minimum redundancy|variance threshold|symmetric uncertainty|select ?k ?best|univariate)"
WRAP=r"(recursive feature elimination|\brfe\b|sequential forward|sequential backward|forward selection|backward elimination|sequential feature selection|\bsfs\b|\bsbs\b|boruta|exhaustive search)"
EMB=r"(lasso|l1[- ]regulari|feature importance|gini importance|importance score|tree-based feature|embedded feature|embedded method|selectfrommodel|permutation importance)"
TRANS=r"(principal component analysis|\bpca\b|linear discriminant analysis|\blda\b|autoencoder|t-sne|truncated svd|\bica\b)"
DS={'NSL-KDD':r'nsl[- _]?kdd','KDD Cup 99':r'kdd[ -]?(cup)?[ -]?(19)?99','UNSW-NB15':r'unsw[- _]?nb[- _]?15','CICIDS2017':r'cic[- _]?ids[- _]?2017','CSE-CIC-IDS2018':r'(cse[- _]?)?cic[- _]?ids[- _]?2018','CIC-DDoS2019':r'cic[- _]?ddos[- _]?2019','Bot-IoT':r'bot[- _]?iot','TON_IoT':r'ton[- _]?iot','N-BaIoT':r'n[- _]?baiot','Edge-IIoTset':r'edge[- _]?iiot','CICIoT2023':r'cic[- _]?iot[- _]?2023','IoTID20':r'iotid[- _]?20','X-IIoTID':r'x[- _]?iiotid','MQTT-IoT-IDS2020':r'mqtt[- _]?iot[- _]?ids','WSN-DS':r'wsn[- _]?ds','AWID':r'\bawid','IoT-23':r'iot[- _]?23\b','CICIoMT2024':r'ciciomt','WUSTL-IIoT':r'wustl','MedBIoT':r'medbiot','Kitsune':r'kitsune','CICIoV2024':r'ciciov','Car-Hacking':r'car[- ]hacking','NF-ToN-IoT/NetFlow':r'\bnf-(ton|bot|unsw|cse)','DS2OS':r'ds2os','RT-IoT2022':r'rt[-_ ]?iot[-_ ]?2022','SWaT':r'\bswat\b|secure water treatment','CIC-DIAD2024':r'cic[-_ ]?diad','Kaggle (unspecified)':r'kaggle'}
CLF={'RF':r'random forest','SVM':r'support vector|\bsvm\b','DT':r'decision tree|\bj48\b|c4\.5','kNN':r'k-nearest|nearest neighbo|\bknn\b|\bk-nn\b','NB':r'naive bayes|naïve bayes','LR':r'logistic regression','XGBoost':r'xgboost','LightGBM':r'lightgbm|light gbm','CatBoost':r'catboost','AdaBoost':r'adaboost','MLP/ANN/DNN':r'multilayer perceptron|multi-layer perceptron|\bmlp\b|artificial neural|\bann\b|deep neural|\bdnn\b','CNN':r'convolutional|\bcnn\b','LSTM/GRU/RNN':r'\blstm\b|\bgru\b|recurrent','Transformer/Attention':r'transformer|attention mechanism|self-attention','Autoencoder':r'autoencoder'}
DL=['MLP/ANN/DNN','CNN','LSTM/GRU/RNN','Transformer/Attention','Autoencoder']
LEGACY={'NSL-KDD','KDD Cup 99','CICIDS2017','CSE-CIC-IDS2018','CIC-DDoS2019','AWID'}

def nums_acc(s):
    v=[float(x) for x in re.findall(r'accuracy[^.;]{0,90}?(\d{2,3}(?:\.\d+)?)\s?%',s)]
    v+= [float(x)*100 for x in re.findall(r'accuracy[^.;]{0,60}?(0\.\d{2,4})\b',s)]
    v=[x for x in v if 50<=x<=100]
    return max(v) if v else np.nan

def feat_counts(s):
    for p in [r'(?:from|reduced from|out of|of the)\s+(\d{1,4})\s+(?:to|features? to|down to)\s+(\d{1,3})\b',
              r'(\d{1,3})\s+(?:features|attributes)\s+(?:out of|from)\s+(?:the\s+)?(?:original\s+|total\s+)?(\d{1,4})',]:
        m=re.search(p,s)
        if m:
            a,b=int(m.group(1)),int(m.group(2))
            o,se=(a,b) if 'to' in p.split('(\\d')[1][:30] else (b,a)
            o,se=max(a,b),min(a,b)
            if 0<se<o<=2000: return o,se
    m=re.search(r'(?:selected|selects|selecting|optimal|best|top)\s+(\d{1,3})\s+(?:optimal |best |relevant |most relevant |important |key )?(?:features|attributes)',s)
    if m and 0<int(m.group(1))<300: return np.nan,int(m.group(1))
    return np.nan,np.nan

out=[]
for r in rows:
    rid=r['rec_id']; full=r['text']; tl=full.lower(); b=body_of(full)
    m=meta.loc[rid]; ta=(str(m.title)+' '+str(m.abstract)).lower()
    iot_n=len(re.findall(r'\b(iot|iiot|iomt|internet of things)\b',b))
    fs_n=len(re.findall(r'feature[s]? selection|selected features|feature reduction|feature subset|dimensionality reduction',b))
    dc={k:len(re.findall(p,b)) for k,p in DS.items()}; mx=max(dc.values())
    ds=[k for k,v in dc.items() if v>0 and v>=max(3,0.4*mx)] or [k for k,v in dc.items() if v>0 and v==mx]
    ft=dict(Filter=R(FILT,b),Seq=R(WRAP,b),Meta=R(META,b),Emb=R(EMB,b),Trans=R(TRANS,b))
    hyb=R(r'hybrid feature selection|two[- ]stage feature selection|multi[- ]stage feature|combin\w* (filter|wrapper)|filter[- ]wrapper|ensemble feature selection',b) or sum([ft['Filter'],ft['Seq'] or ft['Meta'],ft['Emb']])>=2 and R(r'hybrid',ta)
    # primary family from title+abstract
    if R(r'hybrid feature selection|two[- ](stage|phase|tier|layer) feature selection|multi[- ]stage feature|ensemble feature selection|dual[- ]layer feature',ta): prim='Hybrid'
    elif near(META,ta): prim='Wrapper-meta'
    elif near(WRAP,ta): prim='Wrapper-seq'
    elif near(EMB,ta) or R(r'(xgboost|random forest|tree|lightgbm|gini)[- ]based feature',ta): prim='Embedded'
    elif near(FILT,ta): prim='Filter'
    elif near(TRANS,ta): prim='Transformation'
    elif ft['Meta'] and near(META,b): prim='Wrapper-meta'
    elif ft['Seq']: prim='Wrapper-seq'
    elif ft['Emb']: prim='Embedded'
    elif ft['Filter']: prim='Filter'
    elif ft['Trans']: prim='Transformation'
    else: prim='Other'
    clfs=[k for k,p in CLF.items() if len(re.findall(p,b))>=2]
    dl_used=any(len(re.findall(CLF[k],b))>=3 or R(CLF[k],ta) for k in DL)
    if R(r'\b(\d+)[- ]fold|k[- ]fold|cross[- ]validation',b): val='k-fold CV'
    elif R(r'\b(70|75|80|60|67|90)\s?[:/%-]\s?(30|25|20|40|33|10)\b|train[- ]test split|hold[- ]out|training and testing sets?|split into training',b): val='Single hold-out'
    else: val='Not stated'
    train_only='Y' if R(r'(only|solely|exclusively) (on|to|from) the training|on the training (set|data) only|fitted on the training|applied (only )?to the training (set|data)|to avoid (data )?leakage|prevent (data )?leakage',b) else 'Unclear'
    rep='Y' if R(r'independent runs|(\d{1,3}) (independent )?runs|run \d+ times|averaged? over \d+|mean (and|±) standard deviation|± ?\d',b) else 'N'
    base='Y' if R(r'without (applying )?feature selection|all (the )?features|full feature set|original feature set|before feature selection|entire feature set|complete feature set',b) else 'N'
    acc=nums_acc(ta); acc=acc if not np.isnan(acc) else nums_acc(b)
    bo=body_of_orig(full)
    far='Y' if (len(re.findall(r'false alarm|false[- ]positive rate',b))+len(re.findall(r'\b(FAR|FPR)\b',bo)))>=2 else 'N'
    f1=R(r'f1|f-measure|f-score',b); mac='Y' if f1 and R(r'macro|per[- ]class|class[- ]wise|each class|for each attack',b) else 'N'
    stat='Y' if R(r'wilcoxon|friedman|t-test|mann[- ]whitney|statistical(ly)? significan|p[- ]value',b) else 'N'
    fst='Y' if R(r'(selection|execution|computation(al)?|running|training) time|time complexity|time taken',b) else 'N'
    lat='Y' if R(r'(prediction|inference|detection|testing|classification) time|latency',b) else 'N'
    mem='Y' if R(r'memory (usage|consumption|footprint|requirement)|ram usage|model size',b) else 'N'
    en='Y' if R(r'(energy|power) consumption (of|was|is|for|per|in)|\bjoules?\b|\bmj\b|\bmwh?\b',b) else 'N'
    edge='Y' if R(r'(deployed|implemented|tested|evaluated|run|executed|measured) on (an? |the )?(raspberry|jetson|esp32|arduino|microcontroller)',b) else 'N'
    drift='Y' if R(r'concept drift|data drift|incremental learning|online learning',b) else 'N'
    adv='Y' if R(r'adversarial (attack|example|sample|perturbation|robustness|training)s?',b) and R(r'(robust\w*|accuracy|performance) (under|against|to) (the )?adversarial|adversarial (examples|samples) (were|are) generated|fgsm|\bpgd\b|carlini|deepfool|evasion attack',b) else 'N'
    xai='Y' if R(r'\bshap\b|\blime\b|explainab',b) else 'N'
    xsel='Y' if xai=='Y' and R(r'shap[- ]based feature selection|(shap|explainab\w*)[^.]{0,60}(select|rank)\w* (the )?features|features? (selected|ranked) (using|by|with) (shap|xai)',b) else 'N'
    code='Y' if R(r'github\.com|zenodo|code (is|will be) (publicly )?available|source code (is|are) available',b) else 'N'
    cross='Y' if R(r'cross[- ]dataset|cross[- ]domain|trained on [^.]{0,60} (and )?tested on [^.]{0,60}(dataset)|unseen dataset',b) else 'N'
    cls='Both' if R(r'binary',b) and R(r'multi[- ]?class',b) else ('Multi-class' if R(r'multi[- ]?class',b) else ('Binary' if R(r'binary',b) else ''))
    o,se=feat_counts(b)
    params='Y' if R(r'population size|number of (iterations|agents|particles|wolves|generations)|max(imum)? iterations|threshold (value|of)|parameter settings?|\bk ?= ?\d',b) else 'N'
    hw='Y' if R(r'\b(intel|core i\d|ryzen|nvidia|gpu|xeon)\b|\d+ ?gb (of )?(ram|memory)|\bram\b',b) else 'N'
    imb='Y' if R(r'imbalanc|smote|oversampl|undersampl|class weight',b) else 'N'
    prep='Y' if R(r'normali[sz]|standardi[sz]|min[- ]max|preprocess|pre-process|duplicate',b) else 'N'
    sellist='Y' if R(r'(the )?selected features (are|include|were|is)|list of (the )?selected features|selected features (are )?(shown|listed|presented) in',b) else 'N'
    cmpfs='Y' if sum([ft['Filter'],ft['Seq'],ft['Meta'],ft['Emb'],ft['Trans']])>=2 else 'N'
    # eligibility
    if iot_n<2: elig='EC1 not IoT (full text)'
    elif fs_n<2: elig='EC2 no FS step (full text)'
    elif not R(r'\b(intrusion|attack|botnet|ddos|malware|anomal)',str(m.title).lower()+' '+b[:3000]): elig='EC1 not IDS (full text)'
    elif not ds and not R(r'dataset',b): elig='EC3 no identifiable dataset'
    elif not R(r'accuracy|precision|recall|f1|detection rate',b): elig='EC3 no quantitative results'
    else: elig='INCLUDE'
    out.append(dict(rec_id=rid,eligibility=elig,iot_mentions=iot_n,fs_mentions=fs_n,
      Authors=m.authors,Year=int(m.year),Title=m.title,Venue=m.venue,
      **{'Venue type':'Conference' if 'conference' in str(m.type).lower() or str(m.venue_type)=='conference' or R(r'conference|proceedings|symposium|workshop',str(m.venue).lower()) else 'Journal'},
      DOI=m.doi,Filter='Y' if ft['Filter'] else 'N',**{'Wrapper (sequential)':'Y' if ft['Seq'] else 'N','Wrapper (metaheuristic)':'Y' if ft['Meta'] else 'N'},
      Embedded='Y' if ft['Emb'] else 'N',Hybrid='Y' if hyb else 'N',**{'Transformation (PCA/AE)':'Y' if ft['Trans'] else 'N','Primary FS family':prim,
      'Classifier(s)':', '.join(clfs),'Deep learning used':'Y' if dl_used else 'N','Dataset(s)':', '.join(ds) if ds else 'Private/unspecified','No. of datasets':max(1,len(ds)),
      'Legacy/non-IoT dataset only':'Y' if set(ds)<=LEGACY else 'N','Cross-dataset test':cross,'Class setting':cls,'Original features':o,'Selected features':se,
      'Validation protocol':val,'FS fitted on training data only':train_only,'Repeated runs (stochastic FS)':rep,'Baseline without FS':base,
      'Best reported accuracy (%)':acc,'FAR reported':far,'Macro/per-class F1 reported':mac,'Statistical test':stat,'FS time reported':fst,
      'Inference latency reported':lat,'Memory reported':mem,'Energy reported':en,'Edge/IoT hardware used':edge,'Concept drift addressed':drift,
      'Adversarial robustness tested':adv,'XAI used':xai,'XAI used for selection':xsel,'Code/data available':code},
      _params=params,_hw=hw,_imb=imb,_prep=prep,_sellist=sellist,_cmpfs=cmpfs))
df=pd.DataFrame(out)
print(df.eligibility.value_counts())
ov={'R0775':'Retrieved but unreadable','R1635':'Not retrieved (wrong file)','R1396':'EC1 not IDS (full text)','R1574':'EC4 retracted','R1236':'EC5 non-English'}
for k,v in ov.items(): df.loc[df.rec_id==k,'eligibility']=v
df.to_pickle('extracted.pkl')
inc=df[df.eligibility=='INCLUDE']
for c in ['Primary FS family','Validation protocol','Venue type']: print(inc[c].value_counts())
print(inc[['Original features','Selected features']].notna().sum())
for c in ['FAR reported','Inference latency reported','Edge/IoT hardware used','Code/data available','FS fitted on training data only','Cross-dataset test','Concept drift addressed','Adversarial robustness tested','XAI used','Repeated runs (stochastic FS)','Statistical test']:
    print(c,(inc[c]=='Y').sum())
print(inc['No. of datasets'].value_counts().sort_index())
