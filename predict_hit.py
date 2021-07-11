import numpy as np
from project_functions import *

audio_track_info = read_json("audio_track_info.json")
features = get_features()

def predict_hit(id):
    print("Track: "+ audio_track_info[id]['track'] +" Artist: "+audio_track_info[id]["artist"])
    y=[]
    for feature in features:
        y.append(audio_track_info[id][feature])
    loaded_model = joblib.load(open("models/" + audio_track_info[id]['decade'] + "_model.sav", 'rb'))
    return(loaded_model.predict(np.asarray(y).reshape(1, -1)))

hit = predict_hit("5")
if hit == 1:
    print("This will be a hit!!")
else:
    print("This might not be a hit")