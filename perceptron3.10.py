
#!/usr/bin/python3.4
# -*-coding: utf-8 -*
import sys, os, random, re, time, pickle
import numpy as np

def main():
	lim=int(sys.argv[1])
	debut=time.time() # on lance le chrono
	for i in range(0,lim):
		file= "test.input.txt"#input("Nom du fichier à traiter : ") # On récupére le nom du fichier à traité
		nomfichier= "test3.10."+str(i)+".txt" #input("Nommer votre fichier de sortie : ")+".txt" # on nomme le fichier que l'on va créer à partir
		files=["pmilexicon/unigrams-pmilexicon.txt","pmilexicon/bigrams-pmilexicon.txt","pmilexicon/sentimenthashtags.txt", "smiley.txt"]
		recurSBD=sentimentalBindingDictionnary(files)
		sbd=unFileVar("newBindingDictionnary.p") #sentimentalBindingDictionnary(files)
		train="training.txt"# notre fichier d'entraînement
		vectors, dictonary, tweets, ponderation = tools(train,sbd) # vectors représente les vecteurs d'entraînement généré des tweets
		del train # dictonary est l'ensemble clé->mot minuscule : valeur-> indice pour tableau vectoriel
		data=readFile(file, dictonary, sbd) # data est l'ensemble des tweets à traité avec leurs vecteurs crée selon les mots du dictonnaire contnu
		weights, lingWeights =weightsValue(vectors)# weights est l'ensemble des vecteurs de poids pour les trois valeurs
		data=affectationPNN(data,weights,0) # data est l'ensemble des tweets testés évalués
		data=affectationPNN(data,lingWeights,1)
		classify={}
		noClassify={}
		data=actifLearnig(data, vectors, dictonary, tweets, classify, noClassify, recurSBD)
		#data=sarcastique(data)
		writeData(data, nomfichier)# va écrire un nouveau fichier à partir du nomfichier donné et des nouvelles données générées
		del data, nomfichier
	fin=time.time()-debut # fin du chrono
	print("Chronométre: "+ str(int(fin))+" sec") # on affiche le tempes pris environ 1min30

def fileVar(var,strName):
	strName=strName+".p"
	pickle.dump(var,open(strName,"wb"))

def unFileVar(file):
	return pickle.load(open(file,"rb"))

def sentimentalBindingDictionnary(files):
	dictUnigrams=readPmiFile(files[0])
	dictBigrams=readPmiFile(files[1])
	hashtag=readPmiFile(files[2])
	listSmiley=open(files[3],'r').read().split()
	dictHashtag={}
	dictsmiley={}
	for key in hashtag.keys():
		dictHashtag["#"+key]=9

	for elt in listSmiley:
		dictsmiley[elt]=9

	indice=0
	dictSentimental={}
	dictSentIndice={}
	dictSentimental,dictSentIndice, indice=fullThisDict(dictUnigrams,indice,dictSentimental,dictSentIndice)
	dictSentimental,dictSentIndice, indice=fullThisDict(dictBigrams,indice,dictSentimental,dictSentIndice)
	dictSentimental,dictSentIndice, indice=fullThisDict(dictsmiley,indice,dictSentimental,dictSentIndice)
	dictSentimental,dictSentIndice, indice=fullThisDict(dictHashtag,indice,dictSentimental,dictSentIndice)
	return dictSentimental,dictSentIndice, indice+1

def fullThisDict(dico,indice,dictSentimental,dictSentIndice):
	for key in dico.keys():
		dictSentimental[key]=dico[key]
		dictSentIndice[key]=indice
		indice+=1
	return dictSentimental,dictSentIndice, indice

def readPmiFile(file):
	dictPmi={}
	lines=open(file,'r').readlines()
	for line in lines:
		line=line.split("\t")
		key, value = line[:2]
		try:
			dictPmi[key]=valeurAbsolue(float(value))
		except Exception as e:
			dictPmi["#"+key]=9

	return dictPmi

def valeurAbsolue(x):
	if x<0:
		return -x
	else:
		return x

def tools(train,bindingDictionnary): # crée les outils nécessire à l'analyse à partir des données d'entrainment
	dictSentimental,dictSentIndice, indice =bindingDictionnary
	openFile = open(train,'r')
	tweets=[]
	stopwords=set(["a","against","about","above","after","again","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","some","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"])
	dictonary={}
	ponderation={"positive":0,"negative":0,"neutral":0}
	i=0
	for line in openFile:# pour chaque ligne du fichier
		elts=line.split() # on divise la ligne en éléments que
		value=elts[2] # l'on assigne au valeur correspondante
		ponderation[value]+=1
		tweet=elts[3:]# puis en même tempes
		tweets.append((tweet, value))
		for word in tweet:# on crée le dictonary
			word=checkWord(word) # avec les mots en minuscule
			if word not in stopwords and word not in dictonary: # si il ne sont pas dans la liste des stopwords arranger
				dictonary[word]=i # la clé est le mot et la valeur l'indice du tableau du futur vecteur
				i+=1
	for key in ponderation.keys():
		ponderation[key]=1/(ponderation[key]/len(tweets))
	#print(ponderation)
	vectors=[]
	for elts in tweets: # pour chauqe tweet:
		tweet, value=elts
		values={"positive":0,"negative":0,"neutral":0}
		values[value]=1
		vector=np.zeros(len(dictonary))
		for word in tweet: # on regarde si chaque mot est
			if word in dictonary: # dans le dictionnaire
				vector[dictonary[word]]=ponderation[value] # si oui on modifie le vecteur à l'indice du mot

		setTweet= set(tweet)
		j=0 # on crée les bigrams dans l'ensemble des mot du tweet.
		for bi in tweet:
			if j+1 < len(tweet):
				setTweet.add(bi+" "+tweet[j+1])
				j+=1
		vector2 = np.zeros(indice)
		for word in tweet:
			if word in dictSentIndice:
				vector2[dictSentIndice[word]]=dictSentimental[word]*ponderation[value]
		vectors.append((vector,vector2,values))
	tools=[vectors,dictonary, tweets, ponderation]
	print("Vectorification du fichier d'Entrainement terminé:")
	print("1er Vectorification fait sur :"+str(len(vectors))+" tweets" )
	print("Longueur de vector :"+str(len(vectors[0][0]))+" traits" )
	print("Longueur de vector2 :"+str(len(vectors[0][1]))+" traits" )
	return tools

def deleteNotUseValue(tweets,bindingDictionnary):
	dictSentimental,dictSentIndice, indice= bindingDictionnary
	setSBD=set(dictSentimental.keys())
	setTweets=set()
	for tweet in tweets:
		i=0
		words=tweet[0]
		for word in words:
			setTweets.add(word)
			if i+1 < len(words):
				setTweets.add(word+" "+words[i+1])
			i+=1
	setInter=setSBD.intersection(setTweets)
	newDictSentimental={}
	newDictSentIndice={}
	indice=0
	for elt in setInter:
		newDictSentimental[elt]=dictSentimental[elt]
		newDictSentIndice[elt]=indice
		indice+=1
	newBindingDictionnary=[newDictSentimental,newDictSentIndice, indice]
	return newBindingDictionnary

def checkWord(word): # la fonction checkWord(word) sert à trier les mot à évaluer
	wordOK=re.search( r'[,!?.@#]*([A-Za-z]*)[,!?.]*', word) # selon cette expression réguliére
	word=wordOK.group(1) # si le mot est valide il est retourné
	if not wordOK:# sinon
		word="a" # il est transformer en stopword pour pas que l'on en tienne compte
	return word

def readFile(file, dictonary, bindingDictionnary): # readFile(file, dictonary) lit le fichier à testé et crée les vecteurs des tweets grace au dictionnaire
	notuseOne,dictSentIndice,indice =bindingDictionnary
	del notuseOne
	dictTw={}
	openFile = open(file, 'r')
	i=0
	for line in openFile: # pour chaque tweets à testé
		elts=line.split()
		firstNum, secondNum, value = elts[:3]# on stocke ces variables pour la réécriture
		tweet=elts[3:] # on récupère seulement le tweet ici
		vector=np.zeros(len(dictonary)) # on initialise le vecteur avec des zéros
		for word in tweet: # et pour chaque mot du tweet
			wordEva=re.search( r'[,!?.@#]*([A-Za-z]*)[,!?.]*', word) # selon cette expression réguliére
			word=wordEva.group(1)
			if word in dictonary: # on regarde si ces mots en minuscule se trouve dans le dictionnaire
				vector[dictonary[word]]=1# si ou on change la valeur à l'indice du mot du dico dans le vecteur de ce tweet
		setTweet= set(tweet)
		j=0 # on crée les bigrams dans l'ensemble des mot du tweet.
		for bi in tweet:
			if j+1 < len(tweet):
				setTweet.add(bi+" "+tweet[j+1])
				j+=1
		vector2 = np.zeros(indice)
		for word in tweet:
			if word in dictSentIndice:
				vector2[dictSentIndice[word]]=1

		twoValues=["novalue1","novalue2"]
		dictTw[i]=(firstNum , secondNum, value, tweet, vector, vector2,twoValues) # on crée l'enregistrement
		i+=1
	#print("Vectorification du fichier à traité terminé:")
	#print("1er Vectorification fait sur :"+str(len(dictTw))+" tweets" )
	#print("Longueur de vector: "+str(len(vector))+" traits" )
	#print("Longueur de vector2: "+str(len(vector2))+" traits" )
	return dictTw

def weightsValue(vectors):# weightsValue calcule les différents vecteurs de poids avec une tolérance d'erreur
	allWeight=np.random.random_sample((len(vectors[0][0]),)) # on choisit d'initialiser un seul vecteur de
	#allWeight=np.zeros(len(vectors[0][0]))
	print("Calcul des poids pour les neutres:")	# poids aléatoire utiliser pour les trois valeurs
	wNeutral=neurone(vectors,allWeight, "neutral","stati", 0.024)
	print("Calcul des poids pour les positifs:")
	wPositive=neurone(vectors,allWeight, "positive","stati", 0.024)
	print("Calcul des poids pour les negatifs:") # best 0.024
	wNegative=neurone(vectors,allWeight, "negative","stati", 0.024)
	weights=(wPositive,wNegative,wNeutral)
	newAllWeight=np.zeros((len(vectors[0][1]),))
	print("Calcul des poids ling pour les positifs:")
	wLingPositive=neurone(vectors,newAllWeight, "positive","ling", 0.045)
	print("Calcul des poids pour les négatifs:")
	wLingNegative=neurone(vectors,newAllWeight, "negative","ling", 0.045)
	print("Calcul des poids pour les neutres:")
	wLingNeutral=neurone(vectors,newAllWeight, "neutral","ling", 0.045)
	lingWeights=(wLingPositive,wLingNegative,wLingNeutral)

	#print("Entrainement terminé & calcul des poids effectués")
	#print("Longueur des vecteursde poids pour vector: "+str(len(wNegative)))
	#print("Longueur des vecteursde poids pour vector: "+str(len(wLingNeutral)))

	return weights, lingWeights# le neuronne calcule les poids  à partir des poids aléatoires, des vecteurs d'apprentissage de la valeur selectionnée et du pourcentage d'erreur

def neurone(vectors,weight, SelecValue,traitement,pourcentErreur):
	typeTraitement={"stati":0, "ling":1}
	SumWeights=np.zeros(len(weight))
	nbTraining=0
	while True :# on commence une boucle perpétuelle
		erreur=0
		nbTraining+=1
		for entry in vectors:# ou pour chaque entrée
			value=entry[2][SelecValue]
			#print(len(entry[typeTraitement[traitement]]))
			weight, e=weightCalculation(entry[typeTraitement[traitement]],weight, value) # on calcule de nouveau poids tant
			erreur+=e
		SumWeights=np.sum([weight, SumWeights], axis=0)
		#print(str(erreur),str(len(weight)),str((erreur/len(weight)*100)) +"->"+ str(pourcentErreur*100)+"%" )
		if  erreur<pourcentErreur*len(weight): # que l'on a pas un taux d'erreur satisfaisant
			break
	# on fait la moyenne de tous les poids obtenus
	return SumWeights/nbTraining

def weightCalculation(entry,weight,t): # weightCalculation est la suite d'opération pour mettre à jour un poids
	theta=0.17
	o=binarisation(np.dot(entry,weight))# produit interne des poids avec l'entrée passé par la fonction de binarisation
	prediction=np.subtract(t,o) # qui détemine le signe de la prédiction
	deltaWeight= entry*prediction*theta # produit des entrée des predictions et de theta
	newWeight=np.sum([weight, deltaWeight], axis=0) # ajout de deltaW au poids existant
	erreur=0
	if not prediction==0: # incréementeur d'erreur selon la prediction et le résultat de la binarisation du produit interne
		erreur=1
	return newWeight, erreur

def binarisation(x): # binarisation transforme en 1  le parametre
	if x>0:# s'il est plus grand de 0
		x=1
	else: # sinon 0
		x=0
	return x

def affectationPNN(dictTw,weights,x):#  affectationPNN affecte une valeur à chaque tweet selon les poids calculés
	wPositive,wNegative,wNeutral=weights
	values={0:"positive",1:"negative",2:"neutral"}
	for key in dictTw.keys(): # pour chaque tweet
		firstNum, secondNum, value, tweet, vector, vector2,twoValues=dictTw[key]# on calcule
		v=vector
		if x>0:
			v=vector2
		positivePotential=np.dot(wPositive,v)# les produits internes de chaque
		negativePotential=np.dot(wNegative,v)# vecteur du tweet
		neutralPotential=np.dot(wNeutral,v)# et des poids # ensuite on selection la valeur selon le produit le plus grand des trois !
		potentials=[positivePotential,negativePotential,neutralPotential]
		twoValues[x]=values[potentials.index(max(potentials))]
		dictTw[key]=(firstNum, secondNum, value, tweet, vector, vector2,twoValues)
	return dictTw

def actifLearnig(dictTw, vectors, dictonary, tweets, classify, noClassify, recurSBD):


	for key in dictTw.keys(): # pour chaque tweet
		firstNum, secondNum, value, tweet, vector, vector2,twoValues=dictTw[key]
		values={"positive":0,"negative":0,"neutral":0}


		if twoValues[0]==twoValues[1]:
			value= twoValues[0]
			values[value]=1
			tweets.append((tweet, value))
			classify[key]=firstNum, secondNum, value, tweet, vector, vector2,twoValues
		else:
			noClassify[key]=firstNum, secondNum, value, tweet, vector, vector2,twoValues

	print("Éléments classés: "+str(len(classify))+" soit "+str(len(classify)/(len(classify)+len(noClassify))) +"%")
	print("Éléments non classés: "+str(len(noClassify)))
	pourcentClassi=len(classify)*100/(len(classify)+len(noClassify))
	print("len de classify: "+str(len(classify)))

	if pourcentClassi<=60:
		sbd = deleteNotUseValue(tweets,recurSBD)
		print()
		vectorsRec,dictonaryRec, tweetsRec, ponderationRec = recalculVectors(tweets,sbd)
		datayo= reRead(noClassify,dictonaryRec,sbd)
		weightsRec, lingWeightsRec = weightsValue(vectorsRec)
		noClassify={}
		data=affectationPNN(datayo,weightsRec,0) # data est l'ensemble des tweets testés évalués
		data=affectationPNN(data,lingWeightsRec,1)
		data=actifLearnig(data, vectorsRec, dictonaryRec, tweetsRec, classify, noClassify, recurSBD)
	print("Done")

	for key in noClassify.keys():
		firstNum, secondNum, value, tweet, vector, vector2,twoValues=noClassify[key]
		opiff= random.randint(0, 1)
		value=twoValues[0]
		dictTw[key]=firstNum, secondNum, value, tweet, vector, vector2,twoValues

	for key in classify.keys():
		dictTw[key]=classify[key]

	return dictTw

def recalculVectors(tweets,bindingDictionnary):
	dictSentimental,dictSentIndice, indice= bindingDictionnary
	vectors=[]
	recdictonary={}
	stopwords=set(["a","against","about","above","after","again","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","some","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"])
	ponderation={"neutral":0,"positive":0,"negative":0}
	i=0
	j=0
	for elts in tweets:
		tweet, value= elts
		ponderation[value]+=1
		for word in tweet:
			word=checkWord(word) # avec les mots en minuscule
			if word not in stopwords and word not in recdictonary: # si il ne sont pas dans la liste des stopwords arranger
				recdictonary[word]=i # la clé est le mot et la valeur l'indice du tableau du futur vecteur
				i+=1
	for key in ponderation.keys():
		ponderation[key]=1/(ponderation[key]/len(tweets))
		vectors=[]
	for elts in tweets: # pour chauqe tweet:
		tweet, value=elts
		values={"positive":0,"negative":0,"neutral":0}
		values[value]=1
		vector=np.zeros(len(recdictonary))
		for word in tweet: # on regarde si chaque mot est
			if word in recdictonary: # dans le dictionnaire
				vector[recdictonary[word]]=ponderation[value] # si oui on modifie le vecteur à l'indice du mot

		setTweet= set(tweet)
		j=0 # on crée les bigrams dans l'ensemble des mot du tweet.
		for bi in tweet:
			if j+1 < len(tweet):
				setTweet.add(bi+" "+tweet[j+1])
			j+=1
		vector2 = np.zeros(indice)
		for word in tweet:
			if word in dictSentIndice:
				vector2[dictSentIndice[word]]=dictSentimental[word]*ponderation[value]
		vectors.append((vector,vector2,values))
	print()
	print("Vectorification du fichier d'Entrainement terminé:")
	print("Vectorification fait sur: "+str(len(vectors))+" tweets" )
	print("Longueur de vector: "+str(len(vectors[0][0]))+" traits" )
	print("Longueur de vector2: "+str(len(vectors[0][1]))+" traits" )
	return vectors,recdictonary, tweets, ponderation

def reRead(dictTw, recdictonary, bindingDictionnary):
	notuseOne,dictSentIndice,indice =bindingDictionnary
	del notuseOne
	i=0
	for key in dictTw.keys()  : # pour chaque tweets à testé
		firstNum , secondNum, value, tweet, vector, vector2,twoValues=dictTw[key]

		vector=np.zeros(len(recdictonary)) # on initialise le vecteur avec des zéros

		for word in tweet: # et pour chaque mot du tweet
			wordEva=re.search( r'[,!?.@#]*([A-Za-z]*)[,!?.]*', word) # selon cette expression réguliére
			word=wordEva.group(1)
			if word in recdictonary: # on regarde si ces mots en minuscule se trouve dans le dictionnaire
				vector[recdictonary[word]]=1# si ou on change la valeur à l'indice du mot du dico dans le vecteur de ce tweet
		setTweet= set(tweet)
		j=0 # on crée les bigrams dans l'ensemble des mot du tweet.
		for bi in tweet:
			if j+1 < len(tweet):
				setTweet.add(bi+" "+tweet[j+1])
				j+=1

		vector2 = np.zeros(indice)

		for word in tweet:
			if word in dictSentIndice:
				vector2[dictSentIndice[word]]=1

		twoValues=["novalue1","novalue2"]
		dictTw[key]=(firstNum , secondNum, value, tweet, vector, vector2,twoValues) # on crée l'enregistrement
	print()
	print("Données à traité terminé:")
	print("[REC] Vectorification fait sur :"+str(len(dictTw))+" tweets" )
	print("Longueur de vector :"+str(len(vector))+" traits" )
	print("Longueur de vector2 :"+str(len(vector2))+" traits" )
	return dictTw

def sarcastique(dictTw):
	pattern=re
	for key in dictTw.keys():
		firstNum , secondNum, value, tweet, vector, vector2,twoValues= dictTw[key]
		if re.search("TS\d*", secondNum):
			if value=="positive":
				value="negative"
			elif value=="negative":
				value="positive"

		dictTw[key]= firstNum , secondNum, value, tweet, vector, vector2,twoValues

	return dictTw

def writeData(dictTw,nomfichier): # writeData écrit les données dans un nouveau fichier selon le nom donné
	fichier=""
	for i in range(len(dictTw)):
		tweet=""
		for elt in dictTw[i][3]:
			tweet+=elt+" "
		fichier+=dictTw[i][0]+"\t"+dictTw[i][1]+"\t"+dictTw[i][2]+"\t"+tweet+"\n" # recréation du tweet
	output = open(nomfichier,"w")
	output.write(fichier)
	output.close()

if __name__ == '__main__':
    main()
