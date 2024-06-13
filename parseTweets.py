import word2vec as w2v
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

trumpTweets_DF = pd.read_csv("realdonaldtrump.csv", sep=",")
trumpTweets_DF
tweetsContent = trumpTweets_DF["content"].to_numpy()

size = 1000
model = w2v.Word2Vec(tweetsContent[:size], 100, 3, 2, cpu_count=14)
model.saveToFile("./models/model_" + str(size) + ".txt")