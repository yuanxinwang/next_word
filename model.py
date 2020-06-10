import torch
from transformers import GPT2LMHeadModel

#model = GPT2LMHeadModel.from_pretrained('gpt2')
#torch.save(model, "gpt2")

import pickle
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
embed = hub.KerasLayer("static/nnlm-en-dim128_2/")
embeddings = embed(["A long sentence.", "single-word",
                    "http://example.com"])
print(embeddings.shape)  # (3,128)
with open('static/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model("static/my_model.h5")
def predict(model, sentence):
  predicted = model.predict_classes(embed([sentence]), verbose=0)
  for word,index in tokenizer.word_index.items():
      if index == predicted[0]:
          output_word = word
          print(output_word)
          break

predict(model, "whats the craziest")