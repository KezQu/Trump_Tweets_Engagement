import word2vec as w2v
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import sys

size = int(sys.argv[1]) #trumpTweets_DF.shape[0]
vec_size = int(sys.argv[2])
version = sys.argv[3]
try:
	window = int(sys.argv[4])
except IndexError:
	window = 3
try:
	negative = int(sys.argv[5])
except IndexError:
	negative = 2

print(window, negative)

filename = ""
if version == "v1":
	filename = "realdonaldtrump.csv"
elif version == "v2":
	filename = "realerdonaldertrumper.csv"
elif version == "v3":
	filename = "realestdonaldestrumpest.csv"

trumpTweets_DF = pd.read_csv(filename, sep=",")
trumpTweets_DF = trumpTweets_DF.dropna(subset=["content"], axis=0)
tweetsContent = trumpTweets_DF["content"].to_numpy()

print("Requested model_size:",sys.argv[1])
print("Requested vec_size:",sys.argv[2])
print("CSV version:",version,"--",filename)

model = w2v.Word2Vec(tweetsContent[:size], vec_size, window, negative, cpu_count=14)
model.saveToFile("./test_" + str(version) + "/model_" + str(size) + ".txt")