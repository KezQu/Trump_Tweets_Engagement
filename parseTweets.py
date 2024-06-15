import word2vec as w2v
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import sys

filename = ""
if sys.argv[3] == "v1":
	filename = "realdonaldtrump.csv"
elif sys.argv[3] == "v2":
	filename = "realerdonaldertrumper.csv"
elif sys.argv[3] == "v3":
	filename = "realestdonaldestrumpest.csv"

trumpTweets_DF = pd.read_csv(filename, sep=",")
tweetsContent = trumpTweets_DF["content"].to_numpy()

print("Requested model_size:",sys.argv[1])
print("Requested vec_size:",sys.argv[2])
print("CSV version:",sys.argv[3],"--",filename)
size = int(sys.argv[1]) #trumpTweets_DF.shape[0]
vec_size = int(sys.argv[2])

model = w2v.Word2Vec(tweetsContent[:size], vec_size, 3, 2, cpu_count=14)
model.saveToFile("./test_" + str(sys.argv[3]) + "/model_" + str(size) + ".txt")