import matplotlib.pyplot as plt
import os
import warnings
import numpy as np 
import word2vec as w2v
import pandas as pd
import warnings
import datetime
import time
import re
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
# tf.debugging.set_log_device_placement(False)
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def FilterDataset(dataset, size) :
	filtered = dataset.copy().head(size)
	for text, idx in zip(filtered.content.values, filtered.index.values) :
		tmp = np.array([word for word in re.sub('\W', ' ', re.sub("[’']", "", re.sub("http[\S]+", "URL", text, flags=re.U), flags=re.U), flags=re.U).lower().split()])
		if len(tmp) <= 1 :
			filtered.loc[idx]
			filtered = filtered.drop(idx)
	return filtered

def LoadDataset(size, vec_size, version, bounds = [0,1e+2,1e+3,1e+4,1e+5]):
	filename = ""
	if version == "v1":
		filename = "realdonaldtrump.csv"
	elif version == "v2":
		filename = "realerdonaldertrumper.csv"
	elif version == "v3":
		filename = "realestdonaldestrumpest.csv"
	trumpTweets_DF = pd.read_csv(filename, sep=",")
	trumpTweets_DF = trumpTweets_DF.dropna(subset=["content"], axis=0)

	print("CSV version:",version,"--",filename)

	result_data = w2v.Word2Vec()
	result_data.loadFromFile("./models_" + str(version) + "/calculated_model_" + str(size) + "_" + str(vec_size) + ".txt")
	result = result_data.words_df
	filtered_df = FilterDataset(trumpTweets_DF, size)

	dictionary = pd.DataFrame(index=result.word.values, data=result.vector.values)

	list_of_tweets = w2v.stripPunctuation(filtered_df.content[:size])

	filtered_df["content_vectors"] = [np.vstack(result.loc[result.word.isin(tweet)].vector.values) for tweet in list_of_tweets]
	longest_tweet = 0
	for tweet in filtered_df["content_vectors"] :
		if tweet.shape[0] > longest_tweet :
			longest_tweet = tweet.shape[0]
	for i, tweet in zip(filtered_df.index, filtered_df["content_vectors"]) :
		filtered_df["content_vectors"][i] = np.pad(tweet, [(0, longest_tweet - tweet.shape[0]), (0,0)])
	input_mtx = np.stack(filtered_df["content_vectors"])
	
	groups = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
	input_y = filtered_df.retweets.values
	
	y_bounds = [input_y.min(),input_y.max()]
	groups_y = []
	for y in input_y:
		if y >= y_bounds[0] and y < bounds[1]:
			groups_y.append(groups[0])
		elif y >= bounds[1] and y < bounds[2]:
			groups_y.append(groups[1])
		elif y >= bounds[2] and y < bounds[3]:
			groups_y.append(groups[2])
		elif y >= bounds[3] and y < bounds[4]:
			groups_y.append(groups[3])
		else :
			groups_y.append(groups[-1])
	
	groups_y = np.array(groups_y)
	
	flat_X = input_mtx.mean(axis=1)
	flat_X = np.c_[flat_X, np.array([tweet.shape[0] / longest_tweet for tweet in list_of_tweets])]
	timestamp = []
	datetime_frame = [np.inf, -np.inf]
	for date in filtered_df.date.values :
		timestamp.append(time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timetuple()))
		if timestamp[-1] < datetime_frame[0]:
			datetime_frame[0] = timestamp[-1]
		elif timestamp[-1] > datetime_frame[1]:
			datetime_frame[1] = timestamp[-1]

	flat_X = np.c_[flat_X, np.array([(date - datetime_frame[0]) / datetime_frame[1] for date in timestamp])]
	flat_X = np.c_[flat_X, filtered_df.haslink.values]
	
	scaler = StandardScaler()
	scaler.fit(flat_X)
	flat_X = scaler.transform(flat_X)

	return flat_X, groups_y, dictionary

# def train_test_split(input_mtx, input_y, test_size, random_state):
# 	random.seed(random_state)
# 	test_split = random.choices([i for i in range(input_mtx.shape[0])], k= int(np.round(test_size * input_mtx.shape[0])))
# 	train_split = [i for i in range(input_mtx.shape[0]) if i not in test_split]
# 	return input_mtx[train_split], input_mtx[test_split], input_y[train_split], input_y[test_split]

def calculateModel(input_x, input_y, batch_proc, epocs=200, verbose=1) :
	tf.keras.utils.set_random_seed(0)

	retweets_model = keras.Sequential()
	retweets_model.add(keras.Input(shape=input_x.shape[1:]))
	retweets_model.add(keras.layers.Dense(128, activation="relu"))
	retweets_model.add(keras.layers.Dense(64, activation="relu"))
	retweets_model.add(keras.layers.Dense(32, activation="relu"))
	retweets_model.add(keras.layers.Dense(input_y.shape[1], activation="sigmoid"))
	retweets_model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=['categorical_accuracy'])
	
	computation_unit = ""
	if len(tf.config.list_physical_devices('GPU')) > 0 :
		computation_unit = "/GPU:0"
	else :
		computation_unit = "/CPU:0"
	
	with tf.device(computation_unit) :
		history = retweets_model.fit(input_x, input_y, validation_split=0.2, verbose=verbose, epochs=epocs, batch_size=int(batch_proc * input_x.shape[0]), shuffle=True)
	return retweets_model, history

def plot(X_range, score_list, labels) :
	plt.figure(figsize=(15,10))
	for idx, sub_list in enumerate(score_list) :
		plt.plot([i for i in X_range],[scores[0] for scores in sub_list], label="train " + str(labels[idx]))
		plt.plot([i for i in X_range],[scores[1] for scores in sub_list], label="test " + str(labels[idx]))
		for i, pair in enumerate(zip(sub_list, X_range)) :
			plt.annotate(np.round(pair[0][0], 2), [pair[1],pair[0][0]])
			plt.annotate(np.round(pair[0][1], 2), [pair[1],pair[0][1]])
	plt.legend()
	plt.show()

if __name__ == "__main__" :

	model_size = int(sys.argv[1])
	vec_size = int(sys.argv[2])
	version = sys.argv[3]
	show_metrics = bool(sys.argv[4])
	print("Requested model_size:",sys.argv[1])
	print("Requested vec_size:",sys.argv[2])

	X, Y, dictionary = LoadDataset(model_size, vec_size, version)
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42, stratify = Y)
	result = calculateModel(X_train, Y_train, 0.1, 200, verbose=show_metrics)
	history = result[1]

	if show_metrics :
		plt.figure(figsize=[10,7])
		plt.subplot(2,1,1)
		plt.xlabel("number of epocs")
		plt.ylabel("categorical cross-entropy")
		plt.plot(history.history['loss'], label = 'TRAIN')
		plt.plot(history.history['val_loss'], label = 'TEST')
		plt.legend()
		plt.subplot(2,1,2)
		plt.xlabel("number of epocs")
		plt.ylabel("cross-entropy accuracy")
		plt.plot(history.history['categorical_accuracy'], label = 'TRAIN')
		plt.plot(history.history['val_categorical_accuracy'], label = 'TEST')
		plt.legend()
		plt.show()