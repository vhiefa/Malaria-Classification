from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import keras.backend as K
import numpy as np

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def confusion_matrix(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    directory='./dataset/valid_set',
    target_size=(100,100),
    color_mode='rgb',
    batch_size=4132,
    #shuffle=False,
    #seed = 42,
    class_mode='binary'
)

inputdata, labeldata = test_set.next()


jsonfile =open('malaria_model.json', 'r')
loadedmodeljson = jsonfile.read()
jsonfile.close()

loadedmodel = model_from_json(loadedmodeljson)
loadedmodel.load_weights('malaria_model.h5')
loadedmodel.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy', confusion_matrix, recall_m, precision_m, f1_m])

print("input data size : %d" % (len(labeldata)))

score = loadedmodel.evaluate(inputdata, labeldata, verbose=1)

for i in range(len(score)) :
    print("%s: %.2f%%" % (loadedmodel.metrics_names[i], score[i] * 100))