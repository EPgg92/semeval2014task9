
#!/usr/bin/python3.4
# -*-coding: utf-8 

import sys, re, numpy
f = open(sys.argv[1], 'r').read()
resultat=re.findall(r"average.{14} +(\d\.\d{4})", f)
somme=0
for i in range(len(resultat)):
	resultat[i]=float(resultat[i])

print(resultat)
print(numpy.average(resultat))