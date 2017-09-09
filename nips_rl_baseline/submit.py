import numpy as np
from keras.models import load_model
import argparse
from osim.http.client import Client

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token')
args = parser.parse_args()

actor = load_model(args.model)
print(actor.summary())

# Load walking environment
remote_base = 'http://grader.crowdai.org:1729'
client = Client(remote_base)
s = client.env_create(args.token)
while True:
    s = np.asarray([[s]])
    a = actor.predict(s)
    s, r, done, _ = client.env_step(a[0].tolist())
    if done:
        s = client.env_reset()
        if not s:
            break

client.submit()
