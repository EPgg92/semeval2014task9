#!/usr/bin/python3.4
# -*-coding: utf-8 -*
import sys
import os
import random
import re
import time
import pickle
import numpy as np

"""
On crée tout d'abord une variable global qui est le dictionnaire complet
créer à parti des fichier PMI comme dans 3.6 !
Pour créer ce dictionnaire on a utiliser createSBD.py
Pour le recréer il vous suffit d'envoyer la commande :
 python3 createSBD.py dans le bon dossier parent.
"""
global dictionnaireModele2
dictionnaireModele2 = pickle.load(open("sbd2.p", "rb"))


def main():
    tweetsEntrainement = lireFichierTweets("training.txt", True)
    tweetsAClassifier = lireFichierTweets("test.input.txt", False)
    #tweetsClassifie,tweetsAClassifier= gestionTweetCommun(tweetsEntrainement,tweetsAClassifier)
    vecteursE, dictionnaireModele1, dictionnaireModele2Local = vecteursEntrainement(
        tweetsEntrainement)
    tweetsAClassifier = vecteursClassication(
        tweetsAClassifier, dictionnaireModele1, dictionnaireModele2Local)
    poidsMod1, poidsMod2 = calculsValeursPoids(vecteursE)
    data = affectetion(tweetsAClassifier, poidsMod1, 0, True)
    ecrireTweets(data, "testMod1.output.txt")
    data = affectetion(tweetsAClassifier, poidsMod2, 1, True)
    ecrireTweets(data, "testMod2.output.txt")
    data = actifLearnig(data, tweetsEntrainement, {}, {})
    tweetsClassifie = {}
    for key in data.keys():
        tweetsClassifie[key] = data[key]
    ecrireTweets(tweetsClassifie, "testAL.output.txt")
    tweetsClassifie = sarcastique(tweetsClassifie)
    ecrireTweets(tweetsClassifie, "testSarAL.output.txt")


###################################################################################################
"""
lireFichierTweets(file, doublon):
Ici on lit les données soit d'entrainement soit à classifier
si on croise un doublon alors on élimine toutes valeurs qui
peuvent potentiellement porter à confusion car étant plusieurs
fois mais n'ayant pas la même valeur sont éliminées.

Dans l'ensemble d'entrainement il y a 7610 tweet, 7550 sont
uniques, 24 sont doublés et ont la même valeur et 36 sont doublés
et n'ont pas la même valeur.

Dans l'ensemble des tweets development.input.txt il y a 1298 tweets
dont deux qui sont doublés! (Ou il y quelque part dans le fichier
des saut de ligne qui traine)

Il y a 6 tweets en commun entre l'ensemble de dévélopment et d'entrainement.
Confusion table:

et le plus drôle quand on execute:
python3 scoredev.py b training.txt development.gold.txt
gs \ pred| positive| negative|  neutral
---------------------------------------
 positive|        1|        0|        0
 negative|        0|        1|        0
  neutral|        0|        1|        3
C'est que entre le résultat et l'entrainement les données pour un tweet
négatif ne corespond même pas!

malheureusement il n'y pas de tweets communs entre les données d'entrainement
et de test.

Dans l'ensemble des tweets test.input.txt il y a 4633 tweets
dont aucuns sont doublés!
"""


def lireFichierTweets(file, doublon):
    # on ouvre le fichier et on le lit ligne par ligne
    file = open(file, 'r')
    dicoTweets = {}
    elimine = set()
    for line in file:
        # pour chaque tweet on récupére les élément qui vont nous être utile
        tweet = line.split()
        # on crée la clé du dictionnaire avec les deux élément unique du tweets
        cle = (tweet[0], tweet[1])
        value = tweet[2]
        tweet = tweet[3:]
        vector, vector2, valeurPossibles = [], [], ["unknow", "unknow2"]
        setTweet = set(tweet)
        j = 0
        #
        for bi in tweet:  # on met déjà les bigrams dans les ressources du tweets en préviosion
            if j + 1 < len(tweet):  # de l'utilisation du second modèle
                setTweet.add(bi + " " + tweet[j + 1])
            j += 1
        if ((cle in dicoTweets and value != dicoTweets[cle][0])or cle in elimine) and doublon:
            # Si le tweet est déjà de dans avec une autre valeur alors on enléve toutes ses occurences
            del dicoTweets[cle]
            elimine.add(cle)
        else:
            dicoTweets[cle] = (value, tweet, setTweet,
                               vector, vector2, valeurPossibles)
    return dicoTweets


"""
Ici juste pour amélioré mes résultat en dévelopement je reprends les
valeur des tweets en communs pour ne avoir à les classifié (même si comme
on l'a vue il y a une erreur)
"""


def gestionTweetCommun(tweetsEntrainement, tweetsAClassifier):
    set1 = set(tweetsEntrainement.keys())
    set2 = set(tweetsAClassifier.keys())
    tweetsClassifie = {}
    for key in set1.intersection(set2):
        tweetsClassifie[key] = tweetsAClassifier[key]
        del tweetsAClassifier[key]
    return tweetsClassifie, tweetsAClassifier


"""
Dans vecteursEntrainement(tweetsEntrainement) on crée les vecteur d'entrainement
pour les deux modèles en même temps. Puis on les retournes pour  le dictionnarires d'apprentissage pour
plus tard!
"""


def vecteursEntrainement(tweetsEntrainement):
    vecteurs = []
    dictionnaire = {}
    ponderation = {"neutral": 0, "positive": 0, "negative": 0}
    i = 0
    setMod2 = set()
    for elts in tweetsEntrainement.values():
        valeur, tweet = elts[:2]
        setMod2 = setMod2.union(elts[2])
        ponderation[valeur] += 1
        for mot in tweet:
            mot, motOK = verifierMot(mot)
            if motOK and mot not in dictionnaire:
                dictionnaire[mot] = i
                i += 1
    for key in ponderation.keys():
        ponderation[key] = 1 / (ponderation[key] / len(tweetsEntrainement))
    indice = 0
    setSbd2 = set(dictionnaireModele2.keys())
    dictionnaireModele2Local = {}
    for key in setMod2.intersection(setSbd2):
        dictionnaireModele2Local[key] = (indice, dictionnaireModele2[key])
        indice += 1
    for key in tweetsEntrainement.keys():  # pour chauqe tweet:
        value, tweet, setTweet, vector, vector2, valeurPossibles = tweetsEntrainement[key]
        values = {"positive": 0, "negative": 0, "neutral": 0}
        values[value] = 1
        vector = np.zeros(len(dictionnaire))
        for word in tweet:  # on regarde si chaque mot est
            if word in dictionnaire:  # dans le dictionnaire
                # si oui on modifie le vecteur à l'indice du mot
                vector[dictionnaire[word]] = ponderation[value]
        vector2 = np.zeros(len(dictionnaireModele2Local))
        for key in setTweet.intersection(set(dictionnaireModele2Local.keys())):
            vector2[dictionnaireModele2Local[key][0]
                    ] = dictionnaireModele2Local[key][0] * ponderation[value]
        vecteurs.append((vector, vector2, values))
    return vecteurs, dictionnaire, dictionnaireModele2Local


"""
Pour chaque tweet à traité on crée ses deux vecteurs grace aux dictionnaires génèrés précedement!

Par rapport à 3.10 on remarquera que quelque soit la récursion on gére tout le temps les bigrams!
Mais ceci apparement cause des problèmes d'overfeed

"""


def vecteursClassication(tweetsAClassifier, dictionnaireEntrainement, dictionnaireModele2Local):
    for key in tweetsAClassifier.keys():  # pour chaque tweets à testé
        value, tweet, setTweet, vector, vector2, valeurPossibles = tweetsAClassifier[key]
        # on initialise le vecteur avec des zéros
        vector = np.zeros(len(dictionnaireEntrainement))
        vector2 = np.zeros(len(dictionnaireModele2Local))
        for mot in tweet:  # et pour chaque mot du tweet
            mot, motOK = verifierMot(mot)  # selon cette expression réguliére
            # on regarde si ces mots en minuscule se trouve dans le dictionnaire
            if motOK and mot in dictionnaireEntrainement:
                # si ou on change la valeur à l'indice du mot du dico dans le vecteur de ce tweet
                vector[dictionnaireEntrainement[mot]] = 1
        for mot in setTweet:  # pareil pour de deuxième dictionnaire mais cette fois-ci en gérant les bigrams
            if mot in dictionnaireModele2Local:
                vector2[dictionnaireModele2Local[mot][0]] = 1
        tweetsAClassifier[key] = (
            value, tweet, setTweet, vector, vector2, valeurPossibles)
    return tweetsAClassifier


"""
Vérifier mot nous sert pour le premier modèle pour éviter les lien et les mot inutiles (stopwords)
"""


def verifierMot(mot):
    stopwords = set(["a", "against", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "of", "off",
                     "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "some", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])
    # selon cette expression réguliére
    wordOK = re.search(r'[,!?.@#]*([A-Za-z]*)[,!?.]*', mot)
    word = wordOK.group(1)  # si le mot est valide il est retourné
    wordOK = True
    if word in stopwords:
        wordOK = False
    return word, wordOK


"""
Ici est gerer l'appel de tout les calcul de poids de vecteurs d'entrainement créer !
On récuprére ensuite les listes de tout les poids facilement!
"""


def calculsValeursPoids(vecteursE):
    # on choisit d'initialiser un seul vecteur de
    allWeight = np.random.random_sample((len(vecteursE[0][0]),))
    # allWeight=np.zeros(len(vectors[0][0]))
    # poids aléatoire utiliser pour les trois valeurs
    print("Calcul des poids pour les neutres:")
    wNeutral = neurone(vecteursE, allWeight, "neutral", "stati", 0.024)
    print("Calcul des poids pour les positifs:")
    wPositive = neurone(vecteursE, allWeight, "positive", "stati", 0.024)
    print("Calcul des poids pour les negatifs:")  # best 0.024
    wNegative = neurone(vecteursE, allWeight, "negative", "stati", 0.024)
    weights = (wPositive, wNegative, wNeutral)
    newAllWeight = np.zeros((len(vecteursE[0][1]),))
    print("Calcul des poids ling pour les positifs:")
    wLingPositive = neurone(vecteursE, newAllWeight, "positive", "ling", 0.045)
    print("Calcul des poids pour les négatifs:")
    wLingNegative = neurone(vecteursE, newAllWeight, "negative", "ling", 0.045)
    print("Calcul des poids pour les neutres:")
    wLingNeutral = neurone(vecteursE, newAllWeight, "neutral", "ling", 0.045)
    lingWeights = (wLingPositive, wLingNegative, wLingNeutral)
    return weights, lingWeights


"""
Le coeur de notre système enfin plutot le cerveau!
Cet average perceptron ne s'arrete que quand la tolérance d'erreur permise est atteinte!
Puis fais le moyenne de tout les poids qu'il a produit
"""


def neurone(vectors, weight, SelecValue, traitement, pourcentErreur):
    typeTraitement = {"stati": 0, "ling": 1}
    SumWeights = np.zeros(len(weight))
    nbTraining = 0
    while True:  # on commence une boucle perpétuelle
        erreur = 0
        nbTraining += 1
        for entry in vectors:  # ou pour chaque entrée
            value = entry[2][SelecValue]
            # print(len(entry[typeTraitement[traitement]]))
            # on calcule de nouveau poids tant
            weight, e = weightCalculation(
                entry[typeTraitement[traitement]], weight, value)
            erreur += e
        SumWeights = np.sum([weight, SumWeights], axis=0)
        #print(str(erreur),str(len(weight)),str((erreur/len(weight)*100)) +"->"+ str(pourcentErreur*100)+"%" )
        # que l'on a pas un taux d'erreur satisfaisant
        if erreur < pourcentErreur * len(weight):
            break
    # on fait la moyenne de tous les poids obtenus
    return SumWeights / nbTraining


"""
Ce calcul de pods séparer du neurone nous permet de mieux voir séquentielement ce que fait les neurone à chaque itération
Ainsi on évite les erreurs de modèle d'apprentissage!
"""


def weightCalculation(entry, weight, t):
    theta = 0.17
    # produit interne des poids avec l'entrée passé par la fonction de binarisation
    o = binarisation(np.dot(entry, weight))
    prediction = np.subtract(t, o)  # qui détemine le signe de la prédiction
    # produit des entrée des predictions et de theta
    deltaWeight = entry * prediction * theta
    # ajout de deltaW au poids existant
    newWeight = np.sum([weight, deltaWeight], axis=0)
    erreur = 0
    if not prediction == 0:  # incréementeur d'erreur selon la prediction et le résultat de la binarisation du produit interne
        erreur = 1
    return newWeight, erreur


"""
Fonction retournat 1 si la si x >0 et 0 si x<0
"""


def binarisation(x):  # binarisation transforme en 1  le parametre
    if x > 0:  # s'il est plus grand de 0
        x = 1
    else:  # sinon 0
        x = 0
    return x


"""
Notre module d'affectation prend plusieursargument pour savoir quelles données il doit traité
il à même un parametre t qui lui permet d'écritetout de suite une valeur final au tweet si on veut
esayer des condinaison de vecteur précedentes!
"""


def affectetion(tweetsAClassifier, weights, x, t):
    wPositive, wNegative, wNeutral = weights
    values = {0: "positive", 1: "negative", 2: "neutral"}
    for key in tweetsAClassifier.keys():  # pour chaque tweet
        # on calcule
        value, tweet, setTweet, vector, vector2, valeurPossibles = tweetsAClassifier[key]
        v = vector
        if x > 0:
            v = vector2
        # les produits internes de chaque
        positivePotential = np.dot(wPositive, v)
        negativePotential = np.dot(wNegative, v)  # vecteur du tweet
        # et des poids # ensuite on selection la valeur selon le produit le plus grand des trois !
        neutralPotential = np.dot(wNeutral, v)
        potentials = [positivePotential, negativePotential, neutralPotential]
        valeurPossibles[x] = values[potentials.index(max(potentials))]
        if t:
            value = values[potentials.index(max(potentials))]
        tweetsAClassifier[key] = value, tweet, setTweet, vector, vector2, valeurPossibles
    return tweetsAClassifier


"""
Voici le fameux modules qui nous prend temps de temps à l'execution
en effet il est récursif et donc  temps quil n'a pas atteint sa valeur de sortie il recommence à :
Recréer le vecteur avec les données classé selon lui correctement
puisrecalcule les poids puis reaffecte les poids au reste ds tweets à classifier
puis reteste si les valeurs sont similaires entre les deux modèles pour enfin sortir
si le seuil de compatibilité est dépassé ou recommencer!
"""


def actifLearnig(dictTw, tweetsEntrainement, classify, noClassify):
    for key in dictTw.keys():  # pour chaque tweet
        value, tweet, setTweet, vector, vector2, valeurPossibles = dictTw[key]
        values = {"positive": 0, "negative": 0, "neutral": 0}
        nbClassify = 0
        if valeurPossibles[0] == valeurPossibles[1]:
            value = valeurPossibles[0]
            values[value] = 1
            tweetsEntrainement[key] = value, tweet, setTweet, vector, vector2, valeurPossibles
            classify[key] = value, tweet, setTweet, vector, vector2, valeurPossibles
        else:
            noClassify[key] = value, tweet, setTweet, vector, vector2, valeurPossibles
    pourcentClassi = len(classify) * 100 / (len(classify) + len(noClassify))
    print("Éléments classés: " + str(len(classify)) +
          " soit " + str(pourcentClassi) + "%")
    print("Éléments non classés: " + str(len(noClassify)))
    if pourcentClassi <= 60:
        vecteursE, dictionnaireModele1, dictionnaireModele2Local = vecteursEntrainement(
            tweetsEntrainement)
        tweetsAClassifier = vecteursClassication(
            noClassify, dictionnaireModele1, dictionnaireModele2Local)
        noClassify = {}
        poidsMod1, poidsMod2 = calculsValeursPoids(vecteursE)
        data = affectetion(tweetsAClassifier, poidsMod1, 0, False)
        data = affectetion(data, poidsMod2, 1, False)
        data = actifLearnig(data, tweetsEntrainement, classify, noClassify)
        print("Done")
    for key in noClassify.keys():
        value, tweet, setTweet, vector, vector2, valeurPossibles = noClassify[key]
        #opiff= random.randint(0, 1)
        value = valeurPossibles[0]
        dictTw[key] = value, tweet, setTweet, vector, vector2, valeurPossibles
    for key in classify.keys():
        dictTw[key] = classify[key]
    return dictTw


"""
Sarcastique Permet d'inverser simplement la valeur des tweet dit sarcastique!
"""


def sarcastique(dictTw):
    for key in dictTw.keys():
        value, tweet, setTweet, vector, vector2, valeurPossibles = dictTw[key]
        if re.search("TS\d*", secondNum):
            if value == "positive":
                value = "negative"
            elif value == "negative":
                value = "positive"
        dictTw[key] = value, tweet, setTweet, vector, vector2, valeurPossibles
    return dictTw


"""
Voici l'étape finale de notre modifiacation de donneés ici sot re écrite les données à l'intérieur
d'un fichier txt!
"""


def ecrireTweets(tweetClassifie, nomfichier):
    fichier = ""
    for key in tweetClassifie.keys():
        tweet = ""
        for elt in tweetClassifie[key][1]:
            tweet += elt + " "
        fichier += key[0] + "\t" + key[1] + "\t" + \
            tweetClassifie[key][0] + "\t" + tweet + "\n"  # recréation du tweet
    output = open(nomfichier, "w")
    output.write(fichier)
    output.close()


if __name__ == '__main__':
    main()
