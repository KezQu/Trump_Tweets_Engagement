import word2vec as w2v
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import sys

size = int(sys.argv[1])
vec_size = int(sys.argv[2])
version = sys.argv[3]

trumpTweets_DF = None

filename = ""
if version == "v1":
	filename = "realdonaldtrump.csv"
elif version == "v2":
	filename = "realerdonaldertrumper.csv"
elif version == "v3":
	filename = "realestdonaldestrumpest.csv"

trumpTweets_DF = pd.read_csv(filename, sep=",")

print("Requested model_size:",size)
print("Requested vec_size:",vec_size)
print("CSV version:",version,"--",filename)

tweetsContent = trumpTweets_DF["content"].to_numpy()

model = w2v.Word2Vec()

print("Loading model",size,"from file")
model.loadFromFile("./test_" + str(version) + "/model_" + str(size) + ".txt")

np.random.seed(0)
print("Randomizing vectors")
model.randmizeVectors(vec_size)
print(model.words_df.vector[:2])

save_res = w2v.Word2Vec()
save_res.words_df = model.words_df.copy()

print("Training...")
save_res.words_df, found_representations = model.train(0.01, 80)

print(1- cosine(model.words_df.loc[save_res.words_df["word"].isin(["donald"])]["vector"].values[0], model.words_df.loc[save_res.words_df["word"].isin(["trump"])]["vector"].values[0]))
print(1- cosine(save_res.words_df.loc[save_res.words_df["word"].isin(["donald"])]["vector"].values[0], save_res.words_df.loc[save_res.words_df["word"].isin(["trump"])]["vector"].values[0]))

save_res.saveToFile("./models_" + str(version) + "/calculated_model_" + str(size) + "_" + str(vec_size) + ".txt")