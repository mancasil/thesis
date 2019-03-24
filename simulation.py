import random
import numpy as np
import math
from math import pow

n=50
m=3

rfix=128000
s0=0.0000000000001
w=5000000

wperrfix=w/rfix
  
inbits = np.zeros(n) 
cycles = np.zeros(n) 
intensity = np.zeros(n) 
time = np.zeros(n) 
energy = np.zeros(n)

lcomp = np.zeros(n)
lenergy = np.zeros(n)

B=np.zeros(m)
F=np.zeros(m)

p = np.zeros((n,m)) 
g = np.zeros((n,m))

totalbytes=0


for i in range(0,n):
  #input bytes
  inbits[i]=(random.randint(1000, 5000))*1000*8
  totalbytes+=inbits[i]
  #CPU cycles needed
  cycles[i]=(random.randint(1000, 5000))*1000000
  #computational intensity
  intensity[i]=cycles[i]/inbits[i]
  #computation capability cycles/sec
  lcomp[i]=random.randint(1, 10)*100000000
  #energy consumption joules/cycle
  lenergy[i]=0.000000001
  
  extime=intensity[i]*inbits[i]/lcomp[i]
  enconsumption=intensity[i]*inbits[i]*lenergy[i]
  
  #time constraint
  time[i]=random.randint(3, 8)/10*extime
  #energy availability
  energy[i]=random.randint(3, 8)/10*enconsumption
  
  
for j in range(0,m):
  #data limitation of server
  B[j]=random.randint(3,8)/10*totalbytes
  #computation capability of server
  F[j]=random.randint(1,4)*1000000000000
  for i in range(0,n):
    #transmission power, channel gain
    dij=random.randint(10,100)
    
    p[i][j]=pow(dij/100,2)
    g[i][j]=1/pow(dij,2)

  
#available strategies
An=[x / 10 for x in range(1, 11)]


#random allocation of users to available MEC servers
belongs=[]
for i in range(0,n):
  belongs.append(random.randint(0, m-1))
  
#auxiliary arrays
timedif=[0]*n
energydif=[0]*n 
payoff=[0]*n
experiment=[False]*n

#benchmark action and benchmark payoff
a_bench=np.zeros(n) 
u_bench=np.zeros(n) 
#state: discontent->0 watchful->1 hopeful->2 content->3
state=np.zeros(n) 

#exp exploration rate
exp_rate=0.8

#acceptance functions F and G
def calc_q(uibench,ui):
  dif=ui-uibench
  g=-0.1*dif+0.27
  q=pow(exp_rate,g)
  return q

def calc_p(ui):
  f=-0.001*ui+0.008
  p=pow(exp_rate,f)
  return p

belongs=[]
for i in range(0,n):
  belongs.append(random.randint(0, m-1))

for period in range(0,10000):
 # print(belongs)
  
  #array with transmission data rates
  r = np.zeros((n,m)) 
  
  #we keep the needed sums for each mec server, so that we don't compute them everytime
  pgsum=[0]*m
  intensitysum=[0]*m
  bsum=[0]*m
  
  #current strategy profile
  a=[0]*n
  
  bi=[0]*n
  
  for j in range(0,m):
    #print("MEC SERVER "+str(j))
    players=[]
    for i in range(0,n):
      if belongs[i]==j:
        players.append(i)
    #print(players)
      
    
    ite=0
    convergence=0
    
    while convergence==0:
  
      convergence=1
      ite=ite+1
      #random.shuffle(players)
     
      
      for i in players:
          #print("new player "+str(i))
      
          #if previous SR still works
          aprev=a[i]
          bprev=bi[i]
        
          temp=pgsum[j]
          if a[i]!=0:
            temp=temp-p[i][j]*g[i][j]
          
          r[i][j]=w*math.log2(1+wperrfix*((p[i][j]*g[i][j])/(s0+temp)))
          
          Otime=bi[i]*(1/r[i][j]+((B[j]*intensitysum[j])/((B[j]-bsum[j])*F[j]))-intensity[i]/lcomp[i])+intensity[i]*inbits[i]/lcomp[i]
          Oenergy=bi[i]*(p[i][j]/r[i][j]-intensity[i]*lenergy[i])+intensity[i]*inbits[i]*lenergy[i]
          timedif[i]=time[i]-Otime
          energydif[i]=energy[i]-Oenergy
          
          if timedif[i]>=0 and energydif[i]>=0:
            #print("i remain satisfied")
            continue;
          
          else:
            #print("explore higher strategies")
            found=0
            #explore higher strategies
            bsumprev=bsum[j]-bi[i]
          
            if aprev==0:
              pgtemp=pgsum[j]+p[i][j]*g[i][j]
              intsumtemp=intensitysum[j]+intensity[i] 
              #r[i][j] remains the same 
            else:
              pgtemp=pgsum[j]
              intsumtemp=intensitysum[j]
          
            for ai in range(max(2,int(aprev*10)),9):
              atemp=ai/10
            #  print(atemp)
              btemp=atemp*inbits[i]
            
              bsumtemp=bsumprev+btemp
            
            
              Otime=btemp*(1/r[i][j]+((B[j]*intsumtemp)/((B[j]-bsumtemp)*F[j]))-intensity[i]/lcomp[i])+intensity[i]*inbits[i]/lcomp[i]
              Oenergy=btemp*(p[i][j]/r[i][j]-intensity[i]*lenergy[i])+intensity[i]*inbits[i]*lenergy[i]
            
              timedif[i]=time[i]-Otime
              energydif[i]=energy[i]-Oenergy
              
              if timedif[i]>=0 and energydif[i]>=0:
                found=1
                #print("found ai")
                convergence=0
                a[i]=atemp
                bi[i]=btemp
                bsum[j]=bsumtemp
                pgsum[j]=pgtemp
                intensitysum[j]=intsumtemp
                break
            if found==0:
              btemp=0
              bsumtemp=bsumprev+btemp
              Otime=btemp*(1/r[i][j]+((B[j]*intsumtemp)/((B[j]-bsumtemp)*F[j]))-intensity[i]/lcomp[i])+intensity[ i]*inbits[i]/lcomp[i]
              Oenergy=btemp*(p[i][j]/r[i][j]-intensity[i]*lenergy[i])+intensity[i]*inbits[i]*lenergy[i]
              timedif[i]=time[i]-Otime
              energydif[i]=energy[i]-Oenergy
              
              if aprev==0:
                #print("didn't found, same as before")
                continue
              else:
                #print("didn't find this time")
                convergence=0
                a[i]=0
                bi[i]=0
                bsum[j]=bsumprev
                intensitysum[j]-=intensity[i]
                pgsum[j]-=p[i][j]*g[i][j] 
                continue  
  
  #calculate payoff
  for i in range(0,n):
    ot=time[i]-timedif[i]
    oe=energy[i]-energydif[i]
    payoff1=(timedif[i]*energydif[i])/(max(time[i],ot)*max(energy[i],oe))
    if timedif[i]<0 and energydif[i]<0:   
      payoff1=-payoff1
    payoff[i]=payoff1
  print(payoff)

  #if period=0 initialize benchmarks and state
  if period==0:
    for i in range(0,n):
      a_bench[i]=belongs[i]
      u_bench[i]=payoff[i]
      if payoff[i]>=0:
        state[i]=3
      else:
        state[i]=0
    
  #else calculate new states based on payoff ui'
  else:
    for i in range(0,n):
      curstate=state[i]
      #if content
      if curstate==3:
        #if player i experiments
        if experiment[i]==True:
          if payoff[i]>u_bench[i]:
            #accept with probability q:
            choose=random.uniform(0, 1)
            if choose<calc_q(u_bench[i],payoff[i]):
              a_bench[i]=belongs[i]
              u_bench[i]=payoff[i]
        #if player i does not experiment
        else:
          if payoff[i]>u_bench[i]:  
            state[i]=2    
          elif payoff[i]<u_bench[i]: 
            state[i]=1
      #if hopeful
      elif curstate==2:
        if payoff[i]>u_bench[i]:  
          state[i]=3
          u_bench[i]=payoff[i]    
        elif payoff[i]<u_bench[i]: 
          state[i]=1
        elif payoff[i]==u_bench[i]:
          state[i]=3
      #if watchful
      elif curstate==1:
        if payoff[i]>u_bench[i]:  
          state[i]=2   
        elif payoff[i]<u_bench[i]: 
          state[i]=0
        elif payoff[i]==u_bench[i]:
          state[i]=3
      #if discontent
      elif curstate==0:
        #accept new state with probability p:
        choose=random.uniform(0, 1)
        if choose<calc_p(payoff[i]):
          a_bench[i]=belongs[i]
          u_bench[i]=payoff[i]
          state[i]=3

  # START OF NEXT PERIOD  
  
  
  for i in range(0,n):
    #if content explore with probability ε (exp_rate)
    if state[i]==3:
      #print("GEIAUS "+str(i))
      choose=random.uniform(0, 1)
      if choose<exp_rate:
        #print("YES imun "+str(belongs[i]))
        experiment[i]=True
        while(belongs[i]==a_bench[i]):
          belongs[i]=random.randint(0, m-1)
        #print("YES eimai "+str(belongs[i]))
      else:
        experiment[i]=False
    #if discontent play random ai'
    elif state[i]==0:
      belongs[i]=random.randint(0, m-1)



          
