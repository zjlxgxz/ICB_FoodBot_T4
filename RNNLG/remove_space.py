import glob
import gensim
import nltk

def tokenize():
	file_list = glob.glob("./sentence_pattern/agent/*.txt")
	sents = []
	for txt in file_list:
		f = open(txt, 'r')
		sents+=f.read().split('\n')
	#f1 = open("little_test.txt", 'r+')
	f2 = open("little_test1.txt", 'w')
	sent_tokens = []
	
	#sents = f1.read().split('\n')
	
	punc = ".?,:'"
	#print string.punctuation
	
	for line in sents:
		#line1 = line.translate(None, punc)
		#print line1
		if line != '':
			f2.write(line + "\n")
			sent_tokens.append(nltk.word_tokenize(line))
	#f1.close()
	f2.close()
	return sent_tokens

if __name__ == '__main__':
	sent_tokens = tokenize()
	#print sent_tokens
	model = gensim.models.Word2Vec(sent_tokens, min_count = 1, size=80, workers=4)
	f3 = open("./vec/vec_80.txt", 'w')
	f4 = open("./resource/vocab1", 'w')
	for i, word in enumerate(model.wv.vocab):
		f4.write(word)
		if i != len(model.wv.vocab)-1:
			f4.write('\n')
		temp = model.wv[word].tolist()
		temp1 = [str(round(item, 6)) for item in temp]
		f3.write(word + " ")
		f3.write(' '.join(temp1))
		f3.write('\n')
	f4.close()
	f3.close()

	#print model.wv.vocab
