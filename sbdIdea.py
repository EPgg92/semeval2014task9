def sentimentalBindingDictionnary(files):
	dictUnigrams=readPmiFile(files[0])
	dictBigrams=readPmiFile(files[1])
	hashtag=readPmiFile(files[2])
	listSmiley=open(files[3],'r').read().split()
	setHashtagPos=set()
	setHashtagNeg=set()
	for key in hashtag.keys():
		if hashtag[key]=="positive":
			setHashtagPos.add("#"+key)
		else:
			setHashtagNeg.add("#"+key)
	setUniPos, setUniNeg =dictPmi(dictUnigrams)
	setBiPos, setBiNeg = dictPmi(dictBigrams)

	setpos=setUniPos.union(setBiPos).union(setHashtagPos)
	setNeg=setUniNeg.union(setBiNeg).union(setHashtagNeg)

	smiBindingDictionnary={}
	posBindingDictionnary={}
	NegBindingDictionnary={}
	indice=0
	for elt in listSmiley:
		smiBindingDictionnary[elt]=indice;
		indice+=1

	for elt in setpos:
		posBindingDictionnary[elt]=indice;
		indice+=1

	for elt in setpos:
		negbindingDictionnary[elt]=indice;
		indice+=1

	return smiBindingDictionnary,posBindingDictionnary,negBindingDictionnary, indice+1
