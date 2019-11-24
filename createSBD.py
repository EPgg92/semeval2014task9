#!/usr/bin/python3.4
# -*-coding: utf-8 -*
import sys, os, random, re, time, pickle
import numpy as np
"""
Dans ce fichier on recrée le dictionnaire sbd comme dans 3.6 dans le fichier sbd2.p
"""

files=["pmilexicon/unigrams-pmilexicon.txt","pmilexicon/bigrams-pmilexicon.txt","pmilexicon/sentimenthashtags.txt", "smiley.txt"]

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

	dictSentimental,indice=fullThisDict(dictUnigrams,indice,dictSentimental)
	dictSentimental,indice=fullThisDict(dictBigrams,indice,dictSentimental)
	dictSentimental,indice=fullThisDict(dictsmiley,indice,dictSentimental)
	dictSentimental,indice=fullThisDict(dictHashtag,indice,dictSentimental)
	return dictSentimental

def readPmiFile(file):
	dictPmi={}
	lines=open(file,'r').readlines()
	for line in lines:
		line=line.split("\t")
		key, value = line[:2]
		try:
			dictPmi[key]=abs(float(value))
		except Exception as e:
			dictPmi["#"+key]=9

	return dictPmi


def fullThisDict(dico,indice,dictSentimental):
		for key in dico.keys():
			dictSentimental[key]=dico[key]
			indice+=1
		return dictSentimental, indice

def fileVar(var,strName):
	strName=strName+".p"
	pickle.dump(var,open(strName,"wb"))

sbd2=sentimentalBindingDictionnary(files)
fileVar(sbd2,"sbd2")
