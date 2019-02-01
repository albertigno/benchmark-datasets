from keras.models import load_model

model = load_model('model_batch_norm_smaller.h5')

print(model.summary())

get_weights = True

if get_weights:
    weights = []
    layers = []
    for k in model.layers:
        layers.append(k)
        weights.append(k.get_weights())


d=0

