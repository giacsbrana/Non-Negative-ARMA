import math as mm
import numpy as np
import scipy
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.optimize import minimize  

urls = [
    "https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/M5_insample{}.txt".format(i) for i in range(1, 6)
] + ["https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/M5_outsample.txt"]
aa = [pd.read_csv(url, sep='\s+', header=None).values for url in urls]
sales = np.vstack(aa[:5])
outsample = aa[5]
weights=pd.read_csv('https://raw.githubusercontent.com/giacsbrana/Non-Negative-ARMA/refs/heads/main/M5weights.csv')

listsum= lambda x: [sum([u[g] for u in x]) for g in range(len(x[0]))]


L12={f'{n}':{'item': weights['Agg_Level_1'][n],'dept':weights['Agg_Level_2'][n],'data':sales[n].tolist()+outsample[n].tolist(),'w':float(weights['weight'][n]),'arma':[]} 
     for n in range(30490)}


mean=lambda x: sum(x)/len(x)
var=lambda x: sum([i**2 for i in x])/len(x)-mean(x)**2
diff=lambda x:[x[i]-x[i-1] for i in range(1,len(x))]
MeanAbsD=lambda x:mean([abs(x[i]-x[i-1]) for i in range(1,len(x))])
qnorm=lambda x: scipy.stats.norm.ppf(x).tolist()
zero=lambda x: x if x>=0 else 0

def NNarma(y,steps):
    p1=.5; p2=.67; p3=.95; p4=.99; 
    s=7; cma=['NA']*len(y)
    for g in range(len(y)-s+1):
        ss=sum([h/s for h in y[g:(g+s)]])
        if ss!=0: cma[g+3]=ss
        else: cma[g+3]='NA'
    residuals=[y[i]/cma[i] if cma[i]!="NA" else "NA" for i in range(len(cma))]
    sfactors=[mean([residuals[h] for h in range(w,len(y)-s+w,s) if residuals[h]!='NA']) for w in range(s)]    
    sfactors=[(u*s)/sum(sfactors) for u in sfactors]
    if min(sfactors)>0:
        sfactout=(sfactors*(len(y)))[len(y):(len(y)+steps)];div=(sfactors*(len(y)))[:len(y)]
        y=[y[f]/div[f] for f in range(len(y))]
    def ARMAg(para):
        phi=1-mm.exp(-abs(para[0]));theta=-phi*(1-mm.exp(-abs(para[1])));co=abs(para[2])
        m=[];v=[];m+=[0];v+=[y[0]];K=phi+theta;like=0
        for t in range(1,len(y)):
            m+=[phi*m[t-1]+K*v[t-1]]
            v+=[y[t]-co-m[t]]
            like+=v[t]*v[t]
        return like    
    res = minimize(ARMAg, [1,1,1]).x; res=res.tolist();
    phi=1-mm.exp(-abs(res[0]));theta=-phi*(1-mm.exp(-abs(res[1])));co=abs(res[2])
    m=[];v=[];m+=[0];v+=[y[0]];K=phi+theta;
    for t in range(1,len(y)):
        m+=[phi*m[t-1]+K*v[t-1]]
        v+=[y[t]-co-m[t]]
    mf=[phi*m[-1]+K*v[-1]];
    for s in range(1,steps): mf+=[phi*mf[s-1]]
    fo=[f+co for f in  mf]; fout=var(v)    
    if min(sfactors)>0: fo=[fo[f]*sfactout[f] for f in range(len(fo))] 
    Interv=[];Interv+=[fout];
    for j in range(2,steps+1): Interv+=[Interv[-1]+K**2*phi**(2*(j-2))*fout]
    lower=[[fo[i]]+[zero(fo[i]-qnorm((1+u)/2)*mm.sqrt(Interv[i])) for u in [p1,p2,p3,p4]] for i in range(len(fo))]
    upper=[[fo[i]]+[zero(fo[i]+qnorm((1+u)/2)*mm.sqrt(Interv[i])) for u in [p1,p2,p3,p4]] for i in range(len(fo))]
  
    return {'mean':fo,'lower':lower,'upper':upper}

def spl(y,act,metodo,steps):
    u=[0.75,0.835,0.975,0.995]
    A1=[[u[g]*(act[s]-metodo['upper'][s][g+1]) if metodo['upper'][s][g+1]<=act[s] else (1-u[g])*(metodo['upper'][s][g+1]-act[s]) for s in range(steps)] for g in range(4)]
    u=[0.25,0.165,0.025,0.005]
    A2=[[u[g]*(act[s]-metodo['lower'][s][g+1]) if metodo['lower'][s][g+1]<=act[s] else (1-u[g])*(metodo['lower'][s][g+1]-act[s]) for s in range(steps)] for g in range(4)]
    A3=[[.5*(act[s]-metodo['upper'][s][0]) if metodo['upper'][s][0]<=act[s] else .5*(metodo['upper'][s][0]-act[s]) for s in range(steps)]]
    SPL=A1+A2+A3
    return mean([mean(SPL[k])/MeanAbsD(y) for k in range(len(SPL))])

    
WRMSSE12=0
SPL12=0
J=0;ALL=[]
steps=28

for good in range(len(L12)): 
  y=L12[f'{good}']['data'][:-steps]
  act=L12[f'{good}']['data'][-steps:];
  a=0;w=L12[f'{good}']['w']
  while y[a]==0:a+=1 
  y=y[a:]
  method=NNarma(y,steps); L12[f'{good}']['arma']=method
  err=[act[g]-method['mean'][g] for g in  range(len(method['mean']))]
  SPL12+=spl(y,act,method,steps)*w
  WRMSSE12+=mm.sqrt(mean([e**2 for e in err])/mean([g**2 for g in diff(y)]))*w
  ALL+=[[WRMSSE12,SPL12]]
  print(good)
  
print(WRMSSE12,SPL12)

G=[L12[f'{i}']['item'] for i in range(3049)]


L11={f'{k1*3049+k2}':{'item': g,'dept':s,'data':listsum([L12[i]['data'] for i in L12 if L12[i]['item']==g  and L12[i]['dept'][0]==s]),
                      'w':sum([L12[i]['w'] for i in L12 if L12[i]['item']==g and L12[i]['dept'][0]==s]),
                      'arma':listsum([L12[i]['arma']['mean'] for i in L12 if L12[i]['item']==g  and L12[i]['dept'][0]==s])} 
     for k1,s in enumerate(['C','T','W'])
     for k2,g in enumerate(G)
    }


WRMSSE11bu=0
WRMSSE11td=0
SPL11=0
J=0;ALL=[]
steps=28

for good in range(len(L11)): 
  y=L11[f'{good}']['data'][:-steps]
  act=L11[f'{good}']['data'][-steps:];
  a=0;w=L11[f'{good}']['w']
  while y[a]==0:a+=1 
  y=y[a:]
  method=NNarma(y,steps)
  errTD=[act[g]-method['mean'][g] for g in  range(len(method['mean']))]
  errBU=[act[g]-L11[f'{good}']['arma'][g] for g in  range(len(act))]
  SPL11+=spl(y,act,method,steps)*w
  WRMSSE11bu+=mm.sqrt(mean([e**2 for e in errBU])/mean([g**2 for g in diff(y)]))*w
  WRMSSE11td+=mm.sqrt(mean([e**2 for e in errTD])/mean([g**2 for g in diff(y)]))*w
  ALL+=[[WRMSSE11td,WRMSSE11bu,SPL11]]
  print(good)
  
print(WRMSSE11bu,WRMSSE11td,SPL11)


L10={f'{k2}':{'item': g,'data':listsum([L12[i]['data'] for i in L12 if L12[i]['item']==g]),
              'w':sum([L12[i]['w'] for i in L12 if L12[i]['item']==g]),
              'arma':listsum([L12[i]['arma']['mean'] for i in L12 if L12[i]['item']==g])} 
     for k2,g in enumerate(G)
    }

WRMSSE10td=0
WRMSSE10bu=0
SPL10=0
J=0;ALL=[]
steps=28

for good in range(len(L10)): 
  y=L10[f'{good}']['data'][:-steps]
  act=L10[f'{good}']['data'][-steps:];
  a=0;w=L10[f'{good}']['w']
  while y[a]==0:a+=1 
  y=y[a:]
  method=NNarma(y,steps)
  errTD=[act[g]-method['mean'][g] for g in  range(len(method['mean']))]
  errBU=[act[g]-L10[f'{good}']['arma'][g] for g in  range(len(method['mean']))]
  SPL10+=spl(y,act,method,steps)*w
  WRMSSE10td+=mm.sqrt(mean([e**2 for e in errTD])/mean([g**2 for g in diff(y)]))*w
  WRMSSE10bu+=mm.sqrt(mean([e**2 for e in errBU])/mean([g**2 for g in diff(y)]))*w
  ALL+=[[WRMSSE10td,WRMSSE10bu,SPL10]]
  print(good)
  
print(WRMSSE10bu,WRMSSE10td,SPL10)

