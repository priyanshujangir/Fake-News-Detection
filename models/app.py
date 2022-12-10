import gradio as gr
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import keras
from keras.models import load_model
import tensorflow_hub as hub
import tensorflow_text as text

def make_prediction(tweet):
  ser = pd.Series([tweet])
  x_val = ser.to_frame(name="tweet")
  x_val = x_val['tweet']
  model = load_model("bert_small.h5" ,custom_objects={'KerasLayer': hub.KerasLayer} )
  y = model.predict(x_val)
  return y[0][0]

tweet_input = gr.Textbox(label = "Enter the tweet")
output = gr.Number(label = "Probability of a True claim")


app = gr.Interface(fn = make_prediction, inputs=tweet_input , outputs=output, title="COVID News Ground Reality Predictor")
app.launch()

