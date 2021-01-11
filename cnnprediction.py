
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import os


jsonfile =open('malaria_model.json', 'r')
loadedmodeljson = jsonfile.read()
jsonfile.close()

loadedmodel = model_from_json(loadedmodeljson)
loadedmodel.load_weights('malaria_model.h5')
loadedmodel.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

files = []
tp = 0
fp = 0
tn = 0
fn = 0

for r, d, f in os.walk('./dataset/valid_set/Parasitized'):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

for f in files:
    test_image = image.load_img(f, target_size=(100, 100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loadedmodel.predict(test_image)
    if result[0][0] != 0:
        fp = fp + 1
        #prediction = 'uninfected'
    else:
        tp = tp + 1
        #prediction = 'infected'
    #print(prediction)

print('tp : ', tp, ' | fp : ', fp)

files = []
for r, d, f in os.walk('./dataset/valid_set/Uninfected'):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

for f in files:
    test_image = image.load_img(f, target_size=(100, 100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loadedmodel.predict(test_image)
    if result[0][0] != 0:
        tn = tn + 1
        #prediction = 'uninfected'
    else:
        fn = fn + 1
        #prediction = 'infected'
    #print(prediction)

print('tn : ', tn, ' | fn : ', fn)


