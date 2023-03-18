import tensorflow
import pickle
import numpy as np
import os
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tqdm import tqdm
model = ResNet50(weights = 'imagenet',include_top = False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
def extract_features(path,model):
    img = image.load_img(path,target_size =(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result/ norm(result)

    return normalized_result

images_list = []

for file in os.listdir('train-samples'):
    images_list.append(os.path.join('train-samples',file))

print(len(images_list))
features = []
for file in tqdm(images_list):
    features.append(extract_features(file,model))

pickle.dump(images_list,open('images_list.pkl','wb')) 
pickle.dump(features,open('features.pkl','wb'))
