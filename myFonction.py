
#!/usr/bin/python3.4
# -*-coding: utf-8

import sys, re, numpy

fonctions=[]
dejaecrite=[]

for i in range(int(sys.argv[1])):
    f= open("perceptron3."+str(i)+".py" , 'r').read()
    for fonc in re.findall(r"def (.*):", f):
        if fonc not in fonctions:
            fonctions.append(fonc)

    print("fonction de perceptron3."+str(i)+".py :")


    for fonc in fonctions:
        if fonc not in dejaecrite:
            dejaecrite.append(fonc)
            print(fonc)
