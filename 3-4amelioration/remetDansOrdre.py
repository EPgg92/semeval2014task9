
file=open("test.input.txt",'r')
ordre=[]
for line in file:
    tweet=line.split()
    cle = (tweet[0],tweet[1])
    ordre.append(cle)

def lireFichierTweets(file):
    # on ouvre le fichier et on le lit ligne par ligne
    file=open(file,'r')
    dicoTweets={}
    for line in file:
        # pour chaque tweet on récupére les élément qui vont nous être utile
        tweet=line.split()
        cle = (tweet[0],tweet[1])
        value= tweet[2]
        tweet=tweet[3:]
        dicoTweets[cle]=(value,tweet)
    return dicoTweets

def ecrireTweets(tweetClassifie,nomfichier):
	fichier=""
	for key in ordre:
		tweet=""
		for elt in tweetClassifie[key][1]:
			tweet+=elt+" "
		fichier+=key[0]+"\t"+key[1]+"\t"+tweetClassifie[key][0]+"\t"+tweet+"\n" # recréation du tweet
	output = open(nomfichier,"w")
	output.write(fichier)
	output.close()

ecrireTweets(lireFichierTweets("testMod1.output.txt"),"testMod1bis.txt")
ecrireTweets(lireFichierTweets("testMod1.output.txt"),"testMod2bis.txt")
ecrireTweets(lireFichierTweets("testAL.output.txt"),"testModALbis.txt")
