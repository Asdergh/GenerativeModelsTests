import pandas as pd
import os

# data = pd.read_csv("C:\\Users\\1\\Downloads\\archive (12)\\tweet_emotions.csv")
# data = data["content"]
# data.to_csv("C:\\Users\\1\\Downloads\\archive (12)\\tweet_emotions.csv")

with open("C:\\Users\\1\\Desktop\\GenerativeNeuralNetworkStud\\dataset_txt\\tweet_text_data\\tweet_emotions.txt", "r") as file:
    data = file.readlines()

text_buffer = ""
for string_row in data:
    text_buffer += ("\n" + string_row.split(",")[1])

print(text_buffer)
with open("C:\\Users\\1\\Desktop\\GenerativeNeuralNetworkStud\\dataset_txt\\tweet_text_data\\tweet_emotions.txt", "w") as file:
    file.write(text_buffer)
