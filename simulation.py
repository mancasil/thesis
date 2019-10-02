import datetime
import random
import numpy as np
import math
from math import pow

n = 100
m = 5


rfix = 128000
s0 = 0.0000000000001
w = 5000000
wperrfix = w / rfix

# auxiliary arrays
ot = [0] * n
oe = [0] * n
payoff = [0] * n

# SLA parameters
prob = np.zeros((n, m))
l = np.arange(0, m)
learning_step = 0.4

inbits = np.zeros(n)
cycles = np.zeros(n)
intensity = np.zeros(n)
time = np.zeros(n)
energy = np.zeros(n)

lcomp = np.zeros(n)
lenergy = np.zeros(n)

B = np.zeros(m)
F = np.zeros(m)

p = np.zeros((n, m))
g = np.zeros((n, m))
d = np.zeros((n, m))

totalbytes = 0

for i in range(0, n):
    # input bytes
    inbits[i] = (random.randint(1000, 5000)) * 1000 * 8
    totalbytes += inbits[i]
    # CPU cycles needed
    cycles[i] = (random.randint(1000, 5000)) * 1000000
    # computational intensity
    intensity[i] = cycles[i] / inbits[i]
    # computation capability cycles/sec
    lcomp[i] = random.randint(1, 10) * 100000000
    # energy consumption joules/cycle
    lenergy[i] = 0.000000001

    extime = intensity[i] * inbits[i] / lcomp[i]
    enconsumption = intensity[i] * inbits[i] * lenergy[i]

    # time constraint
    time[i] = random.randint(6, 9) / 10 * extime
    # energy availability
    energy[i] = random.randint(6, 9) / 10 * enconsumption

belongs = [0] * n
belongs1 = [0] * n

for i in range(0, n):
    belongs1[i] = random.randint(0, m - 1)

totalbytes = sum(inbits)
for j in range(0, m):
    #maximum data
    B[j] = random.randint(6, 10) / 10 * (1 / m) *totalbytes
    #computational capability
    F[j]=random.randint(1, 4)*1000000000000
    for i in range(0, n):
        # distance, transmission power, channel gain
        d[i][j] = random.randint(10, 100)
        p[i][j] = pow(d[i][j] / 100, 2)
        g[i][j] = 1 / pow(d[i][j], 2)

#maximum periods of sla algorithm
nperiods = 10000
learning_step = 0.4

#number for Monte Carlo periods
nepochs = 10000

test3 = [0] * nepochs
for epoch in range(0, nepochs):
    for i in range(0, n):
        belongs[i] = belongs1[i]
    for i in range(0, n):
        for j in range(0, m):
            prob[i][j] = 1 / m
    avg = n
    welfaresum = 0
    #########################################
    for period in range(0, nperiods):
        ## LAYER2
        r = np.zeros((n, m))
        # we keep the needed sums for each mec server, so that we don't compute them everytime
        pgsum = [0] * m
        intensitysum = [0] * m
        bsum = [0] * m
        # current strategy profile
        a = [0] * n
        bi = [0] * n
        for j in range(0, m):
            players = []
            for i in range(0, n):
                if belongs[i] == j:
                    players.append(i)
            ite = 0
            convergence = 0
            while convergence == 0:
                convergence = 1
                ite = ite + 1
                random.shuffle(players)
                for i in players:
                    aprev = a[i]
                    bprev = bi[i]
                    temp = pgsum[j]
                    if a[i] != 0:
                        temp = temp - p[i][j] * g[i][j]
                    r[i][j] = w * math.log2(1 + wperrfix * ((p[i][j] * g[i][j]) / (s0 + temp)))
                    if (B[j] <= bsum[j]):
                        Otime = np.inf
                    else:
                        t1 = bi[i] * (1 / r[i][j] + ((B[j] * intensitysum[j]) / ((B[j] - bsum[j]) * F[j])))
                        t2 = intensity[i] * (inbits[i] - bi[i]) / lcomp[i]
                        Otime = max(t1 + t2)
                    # Otime=bi[i]*(1/r[i][j]+((B[j]*intensitysum[j])/((B[j]-bsum[j])*F[j]))-intensity[i]/lcomp[i])+intensity[i]*inbits[i]/lcomp[i]
                    Oenergy = bi[i] * (p[i][j] / r[i][j] - intensity[i] * lenergy[i]) + intensity[i] * inbits[i] * \
                              lenergy[i]
                    ot[i] = Otime
                    oe[i] = Oenergy
                    if time[i] >= ot[i] and energy[i] >= oe[i]:
                        # remain satisfied
                        continue;
                    else:
                        found = 0
                        # explore higher strategies
                        bsumprev = bsum[j] - bi[i]
                        if aprev == 0:
                            pgtemp = pgsum[j] + p[i][j] * g[i][j]
                            intsumtemp = intensitysum[j] + intensity[i]
                            # r[i][j] remains the same
                        else:
                            pgtemp = pgsum[j]
                            intsumtemp = intensitysum[j]
                        for ai in range(max(2, int(aprev * 10)), 10):
                            atemp = ai / 10
                            btemp = atemp * inbits[i]
                            bsumtemp = bsumprev + btemp
                            if (B[j] <= bsumtemp):
                                Otime = np.inf
                            else:
                                t1 = btemp * (1 / r[i][j] + ((B[j] * intsumtemp) / ((B[j] - bsumtemp) * F[j])))
                                t2 = intensity[i] * (inbits[i] - btemp) / lcomp[i]
                                Otime = max(t1 + t2)
                            Oenergy = btemp * (p[i][j] / r[i][j] - intensity[i] * lenergy[i]) + intensity[i] * \
                                      inbits[i] * lenergy[i]
                            ot[i] = Otime
                            oe[i] = Oenergy
                            if time[i] >= ot[i] and energy[i] >= oe[i]:
                                found = 1
                                convergence = 0
                                a[i] = atemp
                                bi[i] = btemp
                                bsum[j] = bsumtemp
                                pgsum[j] = pgtemp
                                intensitysum[j] = intsumtemp
                                break
                        if found == 0:
                            btemp = 0
                            bsumtemp = bsumprev + btemp
                            Otime = intensity[i] * inbits[i] / lcomp[i]
                            Oenergy = intensity[i] * inbits[i] * lenergy[i]
                            ot[i] = Otime
                            oe[i] = Oenergy
                            if aprev == 0:
                                # print("didn't find, same as before")
                                continue
                            else:
                                # print("didn't find this time")
                                convergence = 0
                                a[i] = 0
                                bi[i] = 0
                                bsum[j] = bsumprev
                                intensitysum[j] -= intensity[i]
                                pgsum[j] -= p[i][j] * g[i][j]
                                continue
        # calculate payoff
        totalpayoff = 0
        for i in range(0, n):
            timedif = time[i] - ot[i]
            energydif = energy[i] - oe[i]
            payoff1 = (timedif * energydif) / (max(time[i], ot[i]) * max(energy[i], oe[i]))
            if time[i] < ot[i] and energy[i] < oe[i]:
                payoff1 = -payoff1
            elif time[i] >= ot[i] and energy[i] >= oe[i]:
                payoff1 = pow(payoff1, 0.5)
            payoff[i] = payoff1
            totalpayoff += payoff[i]
        welfaresum += totalpayoff
        #   welfare[period]+=totalpayoff
        #   wolf[epoch][period]=totalpayoff
        # SLA Algorithm
        for i in range(0, n):
            if payoff[i] >= 0:
                for j in range(0, m):
                    if j == belongs[i]:
                        prob[i][j] = prob[i][j] + learning_step * payoff[i] * (1 - prob[i][j])
                    else:
                        prob[i][j] = prob[i][j] - learning_step * payoff[i] * prob[i][j]
        for i in range(0, n):
            # choose=random.choices(l,prob[i])
            pith = prob[i].tolist()
            choose = np.random.choice(m, 1, p=pith)
            belongs[i] = choose[0]
        parathiro = 100
        if period % parathiro == 0 and period != 0:
            avgprin = avg
            avg = welfaresum / parathiro
            # avgnsat = sumsat/parathiro
            if abs(avg - avgprin) < 0.01 and test3[epoch] == 0:
                test3[epoch] = period
            welfaresum = 0
            # sumsat=0
        if test3[epoch] != 0:
            break

        # an teleiosan oi epanalipeis
        if period == nperiods - 1:
            test3[epoch] = period

