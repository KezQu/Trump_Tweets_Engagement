import pandas as pd
import numpy as np
import re
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def normalize(vector) :
	return vector / np.linalg.norm(vector)

################################################################################################################################################################################

def accuracy(score : pd.DataFrame) :
	acc = 0
	for score_df in score :
		acc += np.log(score_df["score"]).sum()
	return acc

################################################################################################################################################################################

def score(words_df : pd.DataFrame) :
	scores = []
	for i in range(words_df.shape[0]) :
		vector_sum = np.array([words_df["ctx vector"][j] for j in words_df["context"][i]["idx"]])
		vector_sum_idx = np.array([j for j in words_df["context"][i]["idx"]])
		scores.append(pd.DataFrame(index=vector_sum_idx, data=sp.special.expit(np.dot(vector_sum, words_df["vector"][i])), columns=["score"]))

	return scores
	
################################################################################################################################################################################

def plot(words_df : pd.DataFrame, words = None, context = False) :
	if words is None :
		words = words_df["word"].tolist()

	pca = PCA(n_components=2)
	vec = pca.fit_transform(np.vstack(words_df["vector"].loc[words_df["word"].isin(words)]))
	ctx_vec = pca.fit_transform(np.vstack(words_df["ctx vector"].loc[words_df["word"].isin(words)]))

	if context is True:
		plt.figure(figsize=(12, 5))
		plt.subplot(1,2,1)
	else :
		plt.figure(figsize=(7, 7))
	x = [v[0] for v in vec]
	y = [v[1] for v in vec]
	plt.scatter(x, y)
	plt.title("Word vectors")
	for i, word in enumerate(words_df["word"].loc[words_df["word"].isin(words)]) :
		plt.annotate(word, (x[i], y[i]))

	if context is True:
		plt.subplot(1,2,2)
		x_ctx = [v[0] for v in ctx_vec]
		y_ctx = [v[1] for v in ctx_vec]
		plt.scatter(x_ctx, y_ctx, color=['orange'])
		plt.title("Word context vectors")
		for i, word in enumerate(words_df["word"].loc[words_df["word"].isin(words)]) :
			plt.annotate(word, (x_ctx[i], y_ctx[i]))
	
################################################################################################################################################################################

def injectVectors(words_df : pd.DataFrame, vectors) :
	words_df["vectors"] = [v for v in vectors[0][0]]
	return words_df

################################################################################################################################################################################

class Word2Vec :
	words_df = pd.DataFrame()
	EMBEDDING = 0
	CONTEXT_SIZE = 0
	NEG_SAMPLING = 0
	def __init__(self, text: str, embedding: int = 5, context_size : int = 2, negative_sampling : int = 3) :
		self.EMBEDDING = embedding
		self.CONTEXT_SIZE = context_size
		self.NEG_SAMPLING = negative_sampling
		
		self.words_df["word"] = [word for word in re.split("\,\s|\.\s|\:\s|\s", text)]
		self.words_df["vector"] = [normalize(v) for v in [np.random.rand(self.EMBEDDING) * 2 - 1 for _ in range(self.words_df.shape[0])]]
		self.words_df["ctx vector"] = [normalize(v) for v in [np.random.rand(self.EMBEDDING) * 2 - 1 for _ in range(self.words_df.shape[0])]]

		self.FindContext()

################################################################################################################################################################################

	def GetWindowRange(self, wordIdx) :
		Lshift, Rshift = [0,0]
		if wordIdx - self.CONTEXT_SIZE < 0 :
			Lshift = self.CONTEXT_SIZE - wordIdx
		if wordIdx + self.CONTEXT_SIZE + 1 > self.words_df.shape[0] :
			Rshift = self.CONTEXT_SIZE + wordIdx + 1 - self.words_df.shape[0]
		return (wordIdx - self.CONTEXT_SIZE + Lshift, wordIdx + self.CONTEXT_SIZE + 1 - Rshift)

################################################################################################################################################################################

	def FindContext(self) :
		self.words_df["context"] = [pd.DataFrame() for _ in range(self.words_df.shape[0])]
		for i in range(self.words_df.shape[0]) :
			tmp = {self.words_df["word"][offset] : [offset, 1] for offset in range(self.GetWindowRange(i)[0], self.GetWindowRange(i)[1]) if self.words_df["word"][offset] != self.words_df["word"][i]}
			self.words_df["context"][i]["idx"] = [pair[0] for pair in tmp.values()]
			self.words_df["context"][i]["label"] = [pair[1] for pair in tmp.values()]
			self.words_df["context"][i].index = tmp.keys()

			for _ in range(self.NEG_SAMPLING) :
				timeout = 100
				while True :
					sampleWordIdx = np.random.randint(0, self.words_df.shape[0])
					sampleWord = self.words_df["word"][sampleWordIdx]
					if sampleWord not in self.words_df["context"][i].index :
						break
					timeout -= 1
					if timeout == 0 :
						print("Unable to find word outside context window for word " + self.words_df["word"][i])
						sampleWord = np.NAN
						break
				self.words_df["context"][i].loc[sampleWord] = [sampleWordIdx, 0]

################################################################################################################################################################################

	def update(self, learning_rate : float, tol : float) :
		scores = score(self.words_df)
		copy = self.words_df.copy()
		for i in range(self.words_df.shape[0]) :
			for j in self.words_df["context"][i]["idx"] :
				bias = (self.words_df["ctx vector"][j] - self.words_df["vector"][i]) * (self.words_df["context"][i]["label"][self.words_df["word"][j]] - scores[i]["score"][j]) * learning_rate
				copy["vector"][i] = bias + copy["vector"][i]
				copy["ctx vector"][j] = bias - copy["ctx vector"][j]
		copy["vector"] = [normalize(copy["vector"][i]) for i in range(copy.shape[0])]
		copy["ctx vector"] = [normalize(copy["ctx vector"][i]) for i in range(copy.shape[0])]

		return copy
	
################################################################################################################################################################################

	def train(self, learning_rate : float, max_steps: int, tol : float = 1e-1, draw_middle_results = False) :
		word2vec_list = []
		for i in range(max_steps) :
			self.words_df = self.update(learning_rate, tol)
			word2vec_list.append((np.vstack(self.words_df["vector"].values), accuracy(score(self.words_df))))
			if (i + 1) % int(max_steps / 10) == 0 :
				print("Processing: %3d" % int((i + 1) / max_steps * 100) + "%" + " --- accuracy in current step: %4.5f" % accuracy(score(self.words_df)))
				if draw_middle_results:
					self.plot()

		word2vec_list = sorted(word2vec_list, key=lambda rep: -rep[1])
		self.words_df = injectVectors(self.words_df, word2vec_list)

		return self.words_df, word2vec_list
