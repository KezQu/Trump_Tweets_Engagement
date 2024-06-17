import pandas as pd
# import cupy as np
import numpy as np
import re
import os
import string
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import multiprocessing as mp
import random

def normalize(vector) :
	return vector / np.linalg.norm(vector)

################################################################################################################################################################################

def accuracy(score : list) :
	acc = 0
	for score_df in score :
		acc += np.log(score_df["score"]).mean()
	return acc / len(score)

################################################################################################################################################################################

def score(words_df : pd.DataFrame) :
	scores = []
	for i in range(words_df.shape[0]) :
		vector_sum = np.vstack(words_df["context"][i]["vector"])
		vector_sum_idx = np.array(words_df["context"][i]["idx"])
		scores.append(pd.DataFrame(index=vector_sum_idx, data=sp.special.expit(np.dot(vector_sum, words_df["vector"][i])), columns=["score"]))
		# print(vector_sum.shape)
	return scores
	
################################################################################################################################################################################

def plot(words_df : pd.DataFrame, words = None, labels = True) :
	set_ax_limits = True
	if words is None :
		words = words_df["word"].tolist()
		set_ax_limits = False

	if np.vstack(words_df["vector"]).shape[1] > 2 :
		print("dim too large using PCA")
		pca = PCA(n_components=2)
		vec = pca.fit_transform(np.vstack(words_df["vector"].loc[words_df["word"].isin(words)]))
	else:
		vec = np.vstack(words_df["vector"].loc[words_df["word"].isin(words)])

	plt.figure(figsize=(7,7))
	x = [v[0] for v in vec]
	y = [v[1] for v in vec]
	if set_ax_limits :
		plt.xlim(-1.5, 1.5)
		plt.ylim(-1.5, 1.5)
	plt.scatter(x, y)
	plt.title("Word vectors")
	if labels :
		for i, word in enumerate(words_df["word"].loc[words_df["word"].isin(words)]) :
			plt.annotate(word, (x[i], y[i]))

	plt.show()

################################################################################################################################################################################

def stripPunctuation(list_of_sentences : list) :
	list_of_words = []
	for i, text in enumerate(list_of_sentences) :
		tmp = np.array([word for word in re.sub('\W', ' ', re.sub("[’']", "", re.sub("http[\S]+", "URL", text, flags=re.U), flags=re.U), flags=re.U).lower().split()])
		if len(tmp) > 1 :
			list_of_words.append(tmp)
	return list_of_words

################################################################################################################################################################################

class Word2Vec :
	words_df = pd.DataFrame(columns=["word", "vector", "context"])
	EMBEDDING = 0
	CONTEXT_SIZE = 0
	NEG_SAMPLING = 0
	cpu_count = 1
	list_of_tweets = []
	def __init__(self, list_of_sentences: list = [], embedding: int = 5, context_size : int = 2, negative_sampling : int = 3, cpu_count : int = 1) :
		self.words_df = pd.DataFrame(columns=["word", "vector", "context"])
		if len(list_of_sentences) == 0 :
			return
		
		self.EMBEDDING = embedding
		self.CONTEXT_SIZE = context_size
		self.NEG_SAMPLING = negative_sampling

		self.cpu_count = cpu_count
		self.list_of_tweets = stripPunctuation(list_of_sentences)

		for word in np.unique(np.concatenate(self.list_of_tweets)) :
			self.words_df.loc[self.words_df.shape[0]] = [
				word,
				normalize(np.random.rand(self.EMBEDDING) * 2 - 1),
				pd.DataFrame(columns=["idx", "label", "vector"])
				]
		print("core words found:", self.words_df.shape[0], ", continuing to context on", os.getpid())
		self.FindContext()

################################################################################################################################################################################

	def randmizeVectors(self, EMBEDDING) :
		self.EMBEDDING = EMBEDDING
		self.words_df["vector"] = [normalize(np.random.rand(self.EMBEDDING) * 2 - 1) for _ in self.words_df.index.values]
		for i in range(self.words_df.shape[0]) :
			self.words_df.loc[i, "context"]["vector"] = self.words_df["vector"][self.words_df["context"][i]["idx"]].values

################################################################################################################################################################################


	def partialCtx(self, arr) :
		for i in arr :
			try:
				ctx_range = []
				print("Finding context -- Process", os.getpid(), "%3.1f" % ((i - arr[0]) / len(arr) * 100), "%", "completed")
				for id, corpus in enumerate(self.list_of_tweets) :
					for wordIdx in np.where(corpus == self.words_df["word"][i])[0] :
						window = [self.CONTEXT_SIZE,self.CONTEXT_SIZE]
						if wordIdx < window[0] : window[0] = wordIdx
						if (len(corpus) - 1) - wordIdx < window[1] : window[1] = (len(corpus) - 1) - wordIdx
						if window[0] > 0 : 
							ctx_range.append([(id, i) for i in range(wordIdx - window[0], wordIdx)])
						if window[1] > 0 : 
							ctx_range.append([(id, i) for i in range(wordIdx + 1, wordIdx + window[1] + 1)])
				try :
					ctx_range = np.concatenate(ctx_range)
				except Exception as e :
					with open("./DEBUG/error_concat" + str(i) + ".txt", "w") as f:
						f.write(str(i) + " " + str(self.words_df["word"][i]) + " " + str(e))

				context = self.words_df.loc[self.words_df["word"].isin([self.list_of_tweets[pair[0]][pair[1]] for pair in ctx_range])]
				self.words_df["context"][i]["idx"] = context["word"].index.values
				self.words_df["context"][i]["label"] = [1 for _ in range(context.shape[0])]
				self.words_df["context"][i]["vector"] = context["vector"].values
				self.words_df["context"][i].index = context["word"].values
			except Exception as e:
				with open("./DEBUG/error_ctx" + str(i) + ".txt", "w") as f:
					f.write(str(e) + " " + str(i))

		print("Falling for searching context on", os.getpid())
		for i in arr :
			try:
				print("Finding neg_sampling -- Process", os.getpid(), "%3.1f" % ((i - arr[0]) / len(arr) * 100), "%", "completed")
				to_choose_from = [idx for idx in self.words_df.index.values if idx not in self.words_df["context"][i]["idx"].values]
				for _ in range(self.NEG_SAMPLING * self.words_df["context"][i].shape[0]) :
					try:
						sample_idx = random.choice(to_choose_from)
						to_choose_from.remove(sample_idx)
						self.words_df["context"][i].loc[self.words_df["word"][sample_idx]] = [sample_idx, 0, self.words_df["vector"][sample_idx]]
					except Exception as e :
						print("unable to find more negative samples for: \"" + self.words_df["word"][i] + "\" skipping...")
						break
			except Exception as e:
				with open("./DEBUG/error_neg" + str(i) + ".txt", "w") as f:
					f.write(str(e) + " " + str(i))
		print("Finished batch", arr)
		return self.words_df["context"]

################################################################################################################################################################################

	def FindContext(self) :
		thread_pool = mp.Pool(processes=self.cpu_count)
		contexts = thread_pool.map(self.partialCtx, np.array_split(np.arange(self.words_df.shape[0]), self.cpu_count))
		self.words_df["context"] = [row for row in np.concatenate(contexts) if row.size > 0]
		thread_pool.close()

################################################################################################################################################################################

	def saveToFile(self, filename):
		file = open(filename, "w")
		for i in self.words_df.index.values:
			file.write(self.words_df.loc[i]["word"] + "\n")
			file.write(str(self.words_df.loc[i]["vector"]) + "\n")
			file.write("-----------------------------------------------\n")
			for j in self.words_df.loc[i]["context"].index.values:
				file.write(str(j) + ", " + str(self.words_df.loc[i]["context"].loc[j]["idx"]) + ", " + str(self.words_df.loc[i]["context"].loc[j]["label"]) + "\n")
			file.write("-----------------------------------------------\n")
			
		file.close()

################################################################################################################################################################################

	def loadFromFile(self, filename):
		self.words_df = pd.DataFrame(columns=["word", "vector", "context"])
		buffer = ""
		with open(filename, "r") as file :
			buffer = file.read()
		
		self.words_df["word"] = [re.sub("\s*\[", "", word) for word in re.findall(r"\w+\s\[", buffer)]
		self.words_df["vector"] = [np.array([float(x) for x in re.findall(r"[\-\.\de]+", vec)]) for vec in re.findall(r"\[[\-?\de\.\s]+\]", buffer)]
		
		contexts = [np.array([re.findall(r"\w+", row) for row in re.findall(r"\w+\,\s\w+\,\s\w+", ctx)]) for ctx in re.findall(r"\-{2,}[\w\,\s]+\-{2,}", buffer)]
		contexts = [[[row[0], int(row[1]), int(row[2])] for row in ctx] for ctx in contexts]
		self.words_df["context"] = [pd.DataFrame(index=[row[0] for row in ctx], data=[[row[1], row[2]] for row in ctx], columns=["idx", "label"]) for ctx in contexts]

		for i in range(self.words_df.shape[0]) :
			self.words_df.loc[i, "context"]["vector"] = self.words_df["vector"][self.words_df["context"][i]["idx"]].values

################################################################################################################################################################################

	def calcBias(self, arr) :
		scores = score(self.processed_df)
		copy = self.processed_df.copy()
		
		for i in arr :
			bias = (np.vstack(self.processed_df["context"][i]["vector"]) - self.processed_df["vector"][i]) * (self.processed_df["context"][i]["label"].values - scores[i]["score"].values).reshape(-1,1) * self.learning_rate
			copy.loc[i, "vector"] = normalize(bias.sum(axis=0) + copy["vector"][i])
		return copy["vector"]

################################################################################################################################################################################
	
	def update(self, processed_df : pd.DataFrame) :
		copy = processed_df.copy()

		# copy["vector"] = thread_pool.map(self.calcBias, np.array_split(np.arange(self.words_df.shape[0]), 2))[0]
		copy["vector"] = self.calcBias(np.arange(self.words_df.shape[0]))
		
		for i in range(processed_df.shape[0]) :
			copy.loc[i, "context"]["vector"] = copy["vector"][copy["context"][i]["idx"]].values

		return copy
	
################################################################################################################################################################################

	def train(self, learning_rate : float, max_steps: int, early_stop : float = None, draw_middle_results = False, words = None) :
		self.learning_rate = learning_rate
		self.processed_df = self.words_df.copy()
		# thread_pool = mp.Pool(processes=self.cpu_count)
		word2vec_list = [(np.vstack(self.processed_df["vector"].values), accuracy(score(self.processed_df)))]

		print("Processing: %3d" % int(0 / max_steps * 100) + "%" + " --- accuracy in current step: %4.5f" % accuracy(score(self.processed_df)))
		if draw_middle_results:
			plot(self.processed_df, words=words)
		
		for i in range(max_steps) :
			self.processed_df = self.update(self.processed_df)
			word2vec_list.append((np.vstack(self.processed_df["vector"].values), accuracy(score(self.processed_df))))
			
			print("Processing: %3.1f" % (float(i + 1) / max_steps * 100) + "%" + " --- accuracy in current step: %4.5f" % accuracy(score(self.processed_df)))
			if (i + 1) % int(max_steps / 10) == 0 :
				if draw_middle_results:
					plot(self.processed_df, words=words)
			if early_stop is not None :
				if word2vec_list[-1][1] - word2vec_list[-2][1] < early_stop * word2vec_list[-2][1]:
					print("Algorithm is trained to well... Exiting.")
					break
			
		self.processed_df["vector"] = [v for v in sorted(word2vec_list, key=lambda rep: -rep[1])[0][0]]
		for i in range(self.processed_df.shape[0]) :
			self.processed_df["context"][i]["vector"] = self.processed_df["vector"][self.processed_df["context"][i]["idx"]].values

		return self.processed_df, word2vec_list
