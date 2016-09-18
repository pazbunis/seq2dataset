__author__ = 'pazbu'
import matplotlib.pyplot as plt

pentamers = [line.rstrip('\n') for line in open('/cs/grad/pazbu/Desktop/combs/sorted_5mers')]
ProT = [line.rstrip('\n') for line in open('/cs/grad/pazbu/Desktop/combs/ProT')]
MGW = [line.rstrip('\n') for line in open('/cs/grad/pazbu/Desktop/combs/MGW')]
HelT = [line.rstrip('\n').split(',') for line in open('/cs/grad/pazbu/Desktop/combs/HelT')]
Roll = [line.rstrip('\n').split(',') for line in open('/cs/grad/pazbu/Desktop/combs/Roll')]

pentamer_lu = {}
for n in range(0, len(ProT)):
    HelT_a, HelT_b = HelT[n]
    Roll_a, Roll_b = Roll[n]
    pentamer_lu[pentamers[n]] = (ProT[n], MGW[n], HelT_a, HelT_b, Roll_a, Roll_b)

print(pentamer_lu["ACGTA"])
print(pentamer_lu["TACGT"])

# plt.plot(HelT)
# plt.show()