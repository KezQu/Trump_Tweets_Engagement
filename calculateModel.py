import word2vec as w2v
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import sys

trumpTweets_DF = pd.read_csv("realestdonaldestrumpest.csv", sep=",")
trumpTweets_DF
tweetsContent = trumpTweets_DF["content"].to_numpy()

model = w2v.Word2Vec()
print("Requested model_size:",sys.argv[1])
print("Requested vec_size:",sys.argv[2])
print("CSV version:",sys.argv[3])

size = int(sys.argv[1])
vec_size = int(sys.argv[2])
print("Loading model",size,"from file")
model.loadFromFile("./test_" + str(sys.argv[3]) + "/model_" + str(size) + ".txt")

np.random.seed(0)
print("Randomizing vectors")
model.randmizeVectors(vec_size)
print(model.words_df.vector[:2])

save_res = w2v.Word2Vec()
save_res.words_df = model.words_df.copy()

print("Training...")
save_res.words_df, found_representations = model.train(0.01, 100)

print(1- cosine(model.words_df.loc[save_res.words_df["word"].isin(["donald"])]["vector"].values[0], model.words_df.loc[save_res.words_df["word"].isin(["trump"])]["vector"].values[0]))
print(1- cosine(save_res.words_df.loc[save_res.words_df["word"].isin(["donald"])]["vector"].values[0], save_res.words_df.loc[save_res.words_df["word"].isin(["trump"])]["vector"].values[0]))

save_res.saveToFile("./models_" + str(sys.argv[3]) + "/calculated_model_" + str(size) + "_" + str(vec_size) + ".txt")