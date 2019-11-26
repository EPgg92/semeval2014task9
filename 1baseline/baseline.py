#!/usr/bin/python3.4
# -*-coding: utf-8 -*

import sys, os, random, re

####################################
# Modules d'écriture et de lecture #
####################################
def readFile(file):
	d={}
	l=[]
	openFile = open(file, 'r')
	i=0
	for line in openFile:
		elts=line.split() # on divise la ligne en éléments que 
		firstNum=elts[0]
		secondNum=elts[1] # l'on récupère ensuite dans nos varaibles 
		value=elts[2]
		elts.remove(elts[2])
		elts.remove(elts[1])
		elts.remove(elts[0])
		tweet=elts # on récupère seulement le tweet
		posScore=0
		negScore=0 # on initialise ses scores 
		d[i]=(firstNum , secondNum, value, tweet, posScore, negScore) # on crée l'enregistrement 
		i+=1 # i nous permet de connaître la longueur du dictionnaire
	return d, i

def writeData(data):
	dictTw=data[0] # on récupére le dictionnaire
	lenDictTw=data[1] # on récupére sa longueur 
	#print(lenDictTw)
	for i in range(lenDictTw+1): 
		tweet="" 
		for elt in dictTw[i][3]:
			tweet+=elt # on recrée le tweet
			tweet+=" "
		print(dictTw[i][0]+ "\t" +dictTw[i][1]+ "\t" +dictTw[i][2]+ "\t" +tweet) # et on l'écrit avec ses paramétres

########################################
# Modules de création de dictionnaires #
########################################


# Parfois dans la bibliothèe que de sentiWordNet il y adeux mots qui sont pareil mais qui n'ont pas la même valeur sentimentale
# alors j'ai d'additionner les valeurs et de compter le nombre de fois ou on le trouve pour l'utiliser comme diviseur des valeurs 
# sentimentales 
def sentiWordNet(swntxt):
	openFile = open(swntxt,'r')
	swn={} 
	listKey=[]
	for line in openFile:
		elts=line.split()
		#key= elts[0]+elts[1]
		posScore= float(elts[2])
		negScore= float(elts[3])
		i=4
		#listKey.append(key)
		while (i!=len(elts)) : # tant que l'on arrive pas au dernier élément 
			wordHashtag= re.match(r'^(.+)\#[0-9]',elts[i]) # avec des hashtags et un numéro à la fin
			if(wordHashtag):
				word=wordHashtag.group(1)
				doubleWord=re.match(r'([a-z-\.]+)\_([a-z-\.]+)',word) # composé de un ou deux mots
				if (doubleWord):
					wordkey=doubleWord.group(1)+" "+doubleWord.group(2)
				else:	
					wordkey=word

				if wordkey in swn:
					posScore+=swn[wordkey][0]
					negScore+=swn[wordkey][1]
					occWord=0
					occWord+=swn[wordkey][2]+1

					swn[wordkey]=[posScore,negScore, occWord]	 # on ajoute au dictionnaire une nouvelle entée ou on modifie une ancienne			
				else:
					swn[wordkey]=[posScore,negScore,1]
			else:
				break
			i+=1
	#imprimé le dictionnaire trié:
	#for word in sorted(swn):
	#	print(word, swn[word] )
	return swn

# Dans pmiLexicon on a un score additionner des valeurs positives et négatives et les nombres de fois ou il a était trouvé de manière
# positive et de manière négative


def pmilexicon(pmitxt):
	openFile = open(pmitxt,'r')
	pmi={}
	for line in openFile:
		elts=line.split()
		word=elts[0]
		score=float(elts[1])
		occPos=float(elts[2])
		occNeg=float(elts[3])
		pmi[word]=(score, occPos, occNeg) # je récupère toutes ces valeurs dans un dictionnaires
	#imprimé le dictionnaire trié:
	#for word in sorted(pmi):
	#	print(word, pmi[word] )

	return pmi

#####################################################
# Modules d'analyse des tweets basée sur les termes #
#####################################################


def termeAnalyzerSWN(swn, data):
	dictTw=data[0]
	lenDictTw=data[1]
	for i in range(lenDictTw):
		firstNum=dictTw[i][0]
		secondNum=dictTw[i][1]
		value=dictTw[i][2]
		tweet=dictTw[i][3]
		posScore = dictTw[i][4]
		negScore = dictTw[i][5]

		for elt in tweet: 
			elt.lower()
			if elt in swn: # ja compare chaque élément du tweet au mots du dictionnaire
				pos=swn[elt][0]
				neg=swn[elt][1]
				weakener=swn[elt][2]
				posScore+=pos/weakener # si le mot est dedans 
				negScore+=neg/weakener # je modifie les scores ( affaiblit par le nombre d'occurence du mot dans swn )

		dictTw[i]=(firstNum , secondNum, value, tweet, posScore, negScore)
		
	d=dictTw

	return d, i

def termeAnalyzerPMI(pmi, data):
	dictTw=data[0]
	lenDictTw=data[1]
	for i in range(lenDictTw):
		firstNum=dictTw[i][0]
		secondNum=dictTw[i][1]
		value=dictTw[i][2]
		tweet=dictTw[i][3]
		posScore = dictTw[i][4]
		negScore = dictTw[i][5]

		for elt in tweet:
			elt.lower()
			if elt in pmi: # pour chaque élément du tweet je regarde si il se trouve dans le dictionnaire pmi
				score=pmi[elt][0]
				occPos=pmi[elt][1]
				occNeg=pmi[elt][2]
				occTot=occNeg+occPos				

				negScore-=(score*occNeg/occTot) # si oui j'ajoute sa valeur relative au total d'occurence 
				posScore+=(score*occPos/occTot) # aux scores


		dictTw[i]=(firstNum , secondNum, value, tweet, posScore, negScore)
		
	d=dictTw
	i+=1

	return d, i
	
###########################################################
# Module d'Affectation de la valeur sentimentale du tweet #
# & de calcul des coeficiants d'affectation 			  #
###########################################################
def  affectationPNN(data): # spécifique au données de développement
	dictTw=data[0]
	lenDictTw=data[1]
	for i in range(lenDictTw):
		firstNum=dictTw[i][0]
		secondNum=dictTw[i][1]
		tweet=dictTw[i][3]
		posScore = dictTw[i][4]
		negScore = dictTw[i][5]

		value="neutral"

		# les valeurs suivantes sont étudiées pour données la meilleur baseline 
		if (negScore>0.29 and posScore>0.59 ): # ici je pondère les tweets ambigues
			if negScore>posScore:
				value="negative"
			elif negScore<posScore:
				value="positive"
		elif negScore>0.32: # ici je pondère les tweets non ambigues
			value="negative"
		elif posScore>0.63:
			value="positive"
			
		dictTw[i]=(firstNum , secondNum, value, tweet, posScore, negScore)
	
	d=dictTw

	return d, i

def  affectationPNNCoef(data,coef):
	posAmb=coef[0] # ici je récupère les coefficiants de sélections
	posNonAmb=coef[1]
	negAmb=coef[2]
	negNonAmb=coef[3]

	dictTw=data[0]
	lenDictTw=data[1]
	for i in range(lenDictTw):
		firstNum=dictTw[i][0]
		secondNum=dictTw[i][1]
		tweet=dictTw[i][3]
		posScore = dictTw[i][4]
		negScore = dictTw[i][5]

		value="neutral"

		if (negScore>negAmb and posScore>posAmb ): # le code est a affectationPNN(data) 
			if negScore>posScore: # seul les valeurs sont recalculés pour l'ensemble de données traité
				value="negative"
			elif negScore<posScore:
				value="positive"
		elif negScore>negNonAmb:
			value="negative"
		elif posScore>posNonAmb:
			value="positive"
			
		dictTw[i]=(firstNum , secondNum, value, tweet, posScore, negScore)
	
	d=dictTw

	return d, i

def coefficateur(data, version):
	dictTw=data[0]
	lenDictTw=data[1]
	listpos=[] # on crée des liste 
	listneg=[]
	for i in range(lenDictTw):

		posScore = dictTw[i][4] 
		negScore = dictTw[i][5]

		listpos.append(posScore) # qui vont contenir les valeurs des scores
		listneg.append(negScore)

	
	coefposAmb =0.868694232
	coefnegAmb =0.743464884
	coefmoyenneAmb = 0.806079558

	coefnegNonAmb = 0.820375044
	coefposNonAmb = 0.927588756
	coefmoyenneNonAmb = 0.8739819

	moyennelistpos =sum(listpos)/(len(listpos))
	moyennelistneg =sum(listneg)/(len(listneg)) # on calcules les moyennes 

	coef =[]
	# ici on choisis si on veut se baser sur les coef moyen ou non
	#version 4 valeurs pas de moyenne
	if version=="noMoyenne": 
		posAmb=moyennelistpos*coefposAmb # on calcul les coefs
		posNonAmb=moyennelistpos*coefposNonAmb 
		negAmb=moyennelistneg*coefnegAmb
		negNonAmb=moyennelistneg*coefnegNonAmb
		coef=[posAmb,posNonAmb,negAmb,negNonAmb]
	#version 2 valeurs avec moyenne
	if version=="Moyenne":
		posAmb=moyennelistpos*coefmoyenneAmb
		posNonAmb=moyennelistpos*coefmoyenneNonAmb
		negAmb=moyennelistneg*coefmoyenneAmb
		negNonAmb=moyennelistneg*coefmoyenneNonAmb
		coef=[posAmb,posNonAmb,negAmb,negNonAmb]

	return coef

###########################
# Module de développement #
###########################

def randomizer(data):
	valuePos=["positive", "negative", "neutral"] # valeur possibles du tweet
	dictTw=data[0]
	lenDictTw=data[1]
	d={}
	for i in range(lenDictTw):
		rando=random.randint(0,2) # on choisit un int entre 0, 1 et 2
		
		firstNum=dictTw[i][0]
		secondNum=dictTw[i][1]
		value=valuePos[rando] # on attribut une valeur aléatoire
		tweet=dictTw[i][3]
		posScore = dictTw[i][4]
		negScore = dictTw[i][5]
		d[i]=(firstNum , secondNum, value, tweet, posScore, negScore)

	return d, i


# ce module sert à comprendre les données pendant le dévelopment  en affichant la valeur 
# du minimum et du maximum et la moyenne des scores obtenue pour chaque tweet
def understandData(data): 
	dictTw=data[0]
	lenDictTw=data[1]
	listpos=[]
	listneg=[]

	for i in range(lenDictTw):
		posScore = dictTw[i][4]
		negScore = dictTw[i][5]

		listpos.append(posScore)
		listneg.append(negScore)


	print("Moyenne Positive = "+str(sum(listpos)/(len(listpos)))+ "\t Moyenne Négative = "+str(sum(listneg)/(len(listneg))))
	print("Minimum Positif = "+str(min(listpos))+ "\t Minimum Négatif = "+str(min(listneg)))
	print("Maximum Positif = "+str(max(listpos))+ "\t Maximum Négatif = "+str(max(listneg)))



def main():
	#swntxt="sentiWordNet.txt"
	#swn=sentiWordNet(swntxt)

	pmiUnigram="pmilexicon/unigrams-pmilexicon.txt"
	pmi=pmilexicon(pmiUnigram)
	
	file=sys.argv[1]

	data=readFile(file)

	data=termeAnalyzerPMI(pmi, data)

	#data=termeAnalyzerSWN(swn, data)	

	#data=affectationPNN(data)

	coef=coefficateur(data, "Moyenne")

	data=affectationPNNCoef(data,coef)

	#understandData(data)

	writeData(data)


if __name__ == '__main__':
    main()