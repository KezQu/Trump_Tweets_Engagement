import word2vec as w2v
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

trumpTweets_DF = pd.read_csv("realdonaldtrump.csv", sep=",")
trumpTweets_DF
tweetsContent = trumpTweets_DF["content"].to_numpy()

model = w2v.Word2Vec()
size = trumpTweets_DF.shape[0]
vec_size = 300
print("Loading model",size,"from file")
model.loadFromFile("./models/model_" + str(size) + ".txt")

np.random.seed(0)
print("Randomizing vectors")
model.randmizeVectors(vec_size)
print(model.words_df.vector[:2])

save_res = w2v.Word2Vec()
save_res.words_df = model.words_df.copy()

print("Training...")
save_res.words_df, found_representations = model.train(0.005, 50)

print(1- cosine(model.words_df.loc[save_res.words_df["word"].isin(["donald"])]["vector"].values[0], model.words_df.loc[save_res.words_df["word"].isin(["trump"])]["vector"].values[0]))
print(1- cosine(save_res.words_df.loc[save_res.words_df["word"].isin(["donald"])]["vector"].values[0], save_res.words_df.loc[save_res.words_df["word"].isin(["trump"])]["vector"].values[0]))

save_res.saveToFile("./models/calculated_model_" + str(size) + "_" + str(vec_size) + ".txt")