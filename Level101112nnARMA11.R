
#########################################################################################################
### This code allows testing the performance of the non-negative ARMA(1,1) using M5 competition dataset #
#########################################################################################################
## If you have any questions please contact me at: giacomo.sbrana@neoma-bs.fr ###########################
#########################################################################################################
 

aa1=as.matrix(read.table(url("https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/M5_insample1.txt")))
aa2=as.matrix(read.table(url("https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/M5_insample2.txt")))
aa3=as.matrix(read.table(url("https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/M5_insample3.txt")))
aa4=as.matrix(read.table(url("https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/M5_insample4.txt")))
aa5=as.matrix(read.table(url("https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/M5_insample5.txt")))
outsample=as.matrix(read.table(url("https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/M5_outsample.txt")))
W=read.csv(url("https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/WeightsM5.csv"),header = TRUE)
sales=rbind(aa1,aa2,aa3,aa4,aa5)
who=read.csv(url("https://raw.githubusercontent.com/giacsbrana/M5_dataset/main/ItemDeptCatStoreState.csv"))
steps=28

ArmaNoNegative<-function(y,steps){prob1=.5;prob2=.67;prob3=.95;prob4=.99;yy=y

s=7;w<-rep(1/s,s); cma<-matrix(NA,length(y),1); e=length(yy[yy==0])/length(yy);
for(g in 1:(length(y)-s+1)){cma[g+3]<-sum(w*y[g:(g+s-1)])};
residuals<-y/cma;   sfactors<-c();for(seas in 1:s){
  sfactors[seas]<-mean(na.omit(residuals[seq(seas,length(y)-s+seas,by=s)]))}
sfactors<-sfactors*s/sum(sfactors)
if(min(sfactors)>0){
  sfactout<-rep(sfactors,length(y)+steps)[(length(y)+1):(length(y)+steps)]
  y<-y/rep(sfactors,ceiling(length(y)/s))[1:length(y)]}

ARMAg=function(para){phi=1-exp(-abs(para[1]));theta=-phi*(1-exp(-abs(para[2])));co=abs(para[3])
v<-m<-c();m[1]=0;v[1]=y[1];f=var(y);like=0;K=phi+theta
for(t in 2:length(y)){
  m[t]=phi*m[t-1]+K*v[t-1]
  v[t]=y[t]-co-m[t]
}
sum(v^2)
}

res=optim(c(3,2 ,1),ARMAg);phi=1-exp(-abs(res[[1]][1]));theta=-phi*(1-exp(-abs(res[[1]][2])));co=abs(res[[1]][3])
v<-m<-c(); m[1]=0;v[1]=y[1];K=phi+theta
for(t in 2:length(y)){
  m[t]=phi*m[t-1]+K*v[t-1]
  v[t]=y[t]-co-m[t]
}

mf<-rep(0,steps);mf[1]=phi*m[length(y)]+K*v[length(y)];for(s in 2:steps){mf[s]=phi*mf[s-1]}
fo=(mf+co);#if(sum(tail(yy,steps))==0){fo=rep(0,steps)};
fout=var(v)
if(min(sfactors)>0){fo=fo*sfactout;fout=var(v)};


Interv<-c();Interv[1]=fout;for(j in 2:steps){Interv[j]=Interv[j-1]+phi^(2*(j-2))*K^(2)*fout};
lower0<-upper0<-fo;lower50<-fo-qnorm((1+prob1)/2)*sqrt(Interv);lower0[lower0<0]=0
lower67<-fo-qnorm((1+prob2)/2)*sqrt(Interv);lower95<-fo-qnorm((1+prob3)/2)*sqrt(Interv);lower50[lower50<0]=0
lower99<-fo-qnorm((1+prob4)/2)*sqrt(Interv);upper50<-fo+qnorm((1+prob1)/2)*sqrt(Interv);lower67[lower67<0]=0
upper67<-fo+qnorm((1+prob2)/2)*sqrt(Interv);upper95<-fo+qnorm((1+prob3)/2)*sqrt(Interv);lower95[lower95<0]=0
upper99<-fo+qnorm((1+prob4)/2)*sqrt(Interv);lower99[lower99<0]=0
list(mean=fo,lower=cbind(lower0,lower50,lower67,lower95,lower99),     
     upper=cbind(upper0,upper50,upper67,upper95,upper99))
}

 
steps=28

spl=function(y,act,metodo,steps){#y=ts(y,frequency = 7)
  #n=forecast(auto.arima(ts(y,frequency = 7)),h=steps,level = c(0,50,67,95,99))
  #n=naive(y,steps,level = c(0,50,67,95,99))
  n=metodo
  spl<-matrix(NA,9,steps)
  u=c(0.75,0.835,0.975,0.995)
  for(g in 1:4){
    for(s in 1:steps){
      if(n$upper[s,g+1]<= act[s]){spl[g,s]=u[g]*(act[s]-n$upper[s,g+1])}else{spl[g,s]=(1-u[g])*(n$upper[s,g+1]-act[s])}
    }
  }
  u=c(0.25,0.165,0.025,0.005)
  for(g in 1:4){
    for(s in 1:steps){n$lower[,g+1][n$lower[,g+1]<0]=0
    if(n$lower[s,g+1]<= act[s]){spl[g+4,s]=u[g]*(act[s]-n$lower[s,g+1])}else{spl[g+4,s]=(1-u[g])*(n$lower[s,g+1]-act[s])}
    }
  }
  for(s in 1:steps){
    if(n$upper[s,1]<= act[s]){spl[9,s]=.5*(act[s]-n$upper[s,1])}else{spl[9,s]=(1-.5)*(n$upper[s,1]-act[s])}
  }
  mean(rowMeans(spl)/mean(abs(diff(y))))
}


stBU=matrix(0,nrow(sales),steps)

####################### Level 12 ###################

WRMSSE12<-0
SPL12<-0
for(good in 1:nrow(sales)){print(good);
  tims=sales[good,]
  
  for(t in 1:length(tims)){if(tims[t]!=0){y=tims[t:length(tims)];break}}
  act=outsample[good,]
  st=ArmaNoNegative(y,steps)
  stBU[good,]=st$mean
  metodo=as.numeric(stBU[good,])
  err<-act-metodo
  SPL12<-SPL12+spl(y,act,st,steps)*W$weight[good]
  WRMSSE12<-WRMSSE12+sqrt(mean(err^2)/mean(diff(y)^2))*W$weight[good]
  
}

c(SPL12,WRMSSE12)

########################################## LEVEL 11 #######################



st11out<-list()
Level11out<-list()
Level11<-list()
ITEMS=names(table(who$item_id))
STATE=names(table(who$state_id ))

h=1
for(i in STATE){
  for(j in ITEMS){
    Level11[[h]]=as.numeric(colSums(sales[who$state_id==i&who$item_id==j,]))
    st11out[[h]]=as.numeric(colSums(stBU[who$state_id==i&who$item_id==j,]))
    Level11out[[h]]=as.numeric(colSums(outsample[who$state_id==i&who$item_id==j,]))
    
    h=h+1}
  
}


WRMSSE11td<-WRMSSE11bu<-SPL11<-0


for(good in 1:length(Level11)){print(good);
  tims=Level11[[good]]
  
  for(t in 1:length(tims)){if(tims[t]!=0){y=tims[t:length(tims)];break}}
  st=ArmaNoNegative(y,steps)
  err<-Level11out[[good]]-st11out[[good]]
  errtd<-Level11out[[good]]-st$mean
  
  SPL11<-SPL11+spl(y,Level11out[[good]],st,steps)*W$weight[30490+good]
  WRMSSE11bu<-WRMSSE11bu+sqrt(mean(err^2)/mean(diff(y)^2))*W$weight[30490+good]
  WRMSSE11td<-WRMSSE11td+sqrt(mean(errtd^2)/mean(diff(y)^2))*W$weight[30490+good]
  
}

c(SPL11,WRMSSE11bu,WRMSSE11td)


######################################### LEVEL 10 #######################

st10out<-list()
Level10out<-list()
Level10<-list()
ITEMS=names(table(who$item_id))

h=1
for(j in ITEMS){
  Level10[[h]]=as.numeric(colSums(sales[who$item_id==j,]))
  st10out[[h]]=as.numeric(colSums(stBU[who$item_id==j,]))
  Level10out[[h]]=as.numeric(colSums(outsample[who$item_id==j,]))
  h=h+1}


WRMSSE10td<-WRMSSE10bu<-SPL10<-0

for(good in 1:length(Level10)){print(good);
  tims=Level10[[good]]
  
  for(t in 1:length(tims)){if(tims[t]!=0){y=tims[t:length(tims)];break}}
  st=ArmaNoNegative(y,steps)
  err<-Level10out[[good]]-st10out[[good]]#st11out[[good]]
  errtd<-Level10out[[good]]-st$mean
  SPL10<-SPL10+spl(y,Level10out[[good]],st,steps)*W$weight[30490+9147+good]
  WRMSSE10bu<-WRMSSE10bu+sqrt(mean(err^2)/mean(diff(y)^2))*W$weight[30490+9147+good]
  WRMSSE10td<-WRMSSE10td+sqrt(mean(errtd^2)/mean(diff(y)^2))*W$weight[30490+9147+good]
}

c(WRMSSE12,WRMSSE11bu,WRMSSE10bu)

c(SPL12,SPL11,SPL10)
