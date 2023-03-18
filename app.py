import random
import json
from flask import Flask, request, app, render_template, url_for, jsonify
import pickle
import requests
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image


features = np.array(pickle.load(open('pickle-files/features.pkl', 'rb')))
images_list = pickle.load(open('pickle-files/images_list.pkl', 'rb'))
dataset = pickle.load(open('pickle-files/dataset.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def feature_extraction(path, model):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result)

    neighbors = NearestNeighbors(
        n_neighbors=8, algorithm='brute', metric='euclidean')
    neighbors.fit(features)

    distances, indices = neighbors.kneighbors([normalized_result])

    filenames_to_return = []
    for file in indices[0]:
        filenames_to_return.append(images_list[file].replace('\\', '/'))

    data = dataset

    output_links = []

    for output_filename in filenames_to_return:
        for i in range(len(data['filename'])):
            if output_filename == data['filename'][i]:
                output_links.append(data['link'][i])

    return output_links


app = Flask(__name__)


@app.route('/home', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        path = request.form["path"]
        result = feature_extraction(path, model)
        return render_template("buy.html", result=result, path=path)
    top_selling = [{'path': 'static/image/watchus.jpg','name':'Watch','price':2999}, {'path': 'static/image/shoeus1.jpg','name':'Shoe','price':3500},
                   {'path': 'static/image/handbag4.jpeg','name':'Handbag','price':1900}, {'path': "static/image/shirtimg13.jpg",'name':'Shirt','price':950}, {'path':"static/image/shoesimg4.jpg",'name':'Sneaker','price':3200},{'path': 'static/image/bag3.jpg','name':'Bag','price':1600},{'path': 'static/image/sgimg.jpg','name':'Shades','price':799},{'path': "static/image/jeans5.jpeg",'name':'Jogger','price':950},{'path': 'static/image/tshirt3.jpg','name':'Tshirt','price':400},{'path': "static/image/jeans4.jpeg",'name':'Jeans','price':1200}]
    return render_template('index.html',tops = top_selling , result=[])


@app.route("/explore",methods=['POST','GET'])
def explore():
    if request.method == 'POST':
        path = request.form["path"]
        result = feature_extraction(path, model)
        return render_template("buy.html",result=result,path=path)
    shirts = [{'path': 'static/image/shirt-explore.jpeg'}, {'path': 'static/image/shirt-explore-1.jpeg'},
              {'path': 'static/image/shirt-explore-3.jpeg'}, {'path': "static/image/shirt-explore-5.jpeg"}, {'path': "static/image/shirt-explore-6.jpeg"}]
    shoes = [{'path': 'static/image/shoe-explore.jpg'}, {'path': 'static/image/shoe-explore-2.jpg'}, {'path': 'static/image/shoe-explore-1.jpg'},
             {'path': 'static/image/shoe-explore-3.jpg'}, {'path': "static/image/shoe-explore-6.jpg"}]
    jeans = [{'path': 'static/image/jeans-explore.jpg'}, {'path': 'static/image/jeans-explore-1.jpg'}, {'path': 'static/image/jeans-explore-2.jpg'},
             {'path': 'static/image/jeans-explore-3.jpg'}, {'path': "static/image/jeans-explore-5.jpg"}]
    sunglasses = [{'path': 'static/image/sunglass-explore-1.jpg'}, {'path': 'static/image/sunglass-explore-2.jpg'},
                  {'path': 'static/image/sunglass-explore-3.jpg'}, {'path': "static/image/sunglass-explore-7.jpg"}, {'path': "static/image/sunglass-explore-5.jpg"}]
    watches = [{'path': 'static/image/watch-explore.jpg'}, {'path': 'static/image/watch-explore-1.jpg'},
               {'path': 'static/image/watch-explore-5.jpg'}, {'path': "static/image/watch-explore-3.jpg"}, {'path': "static/image/watch-explore-6.jpg"}]
    tshirts = [{'path': 'static/image/tshirt-explore-2.jpg'}, {'path': 'static/image/tshirt-explore-3.jpg'},
               {'path': 'static/image/tshirt-explore-4.jpg'}, {'path': "static/image/tshirt-explore-5.jpg"}, {'path': "static/image/tshirt-explore-6.jpg"}]
    return render_template("explore.html",shirts=shirts,shoes=shoes,jeans=jeans,sunglasses=sunglasses,watches=watches,tshirts=tshirts)




@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route("/store", methods=["GET", "POST"])
def store():
    if request.method == "POST":
        path = request.form["path"]
        result = feature_extraction(path, model)
        return render_template("buy.html", result=result, path=path)
    shirts = [{'path': 'static/image/shirtsimg.jpg'}, {'path': 'static/image/shirt-explore-8.jpeg'},
              {'path': 'static/image/shirt-explore-2.jpeg'}, {'path': "static/image/shirtimg13.jpg"}, {'path': "static/image/shirtsimg11.jpg"}]
    handbags = [{'path': 'static/image/hand.jpg'}, {'path': 'static/image/handbag4.jpeg'},
                {'path': 'static/image/bag3.jpg'}, {'path': "static/image/bag4.jpg"}, {'path': "static/image/handbag1.jpg"}]
    watches = [{'path': 'static/image/watchus.jpg'}, {'path': 'static/image/watchimg2.jpg'},
               {'path': 'static/image/watchimg8.jpg'}, {'path': "static/image/watchimg7.jpg"}, {'path': "static/image/watchus1.jpg"}]
    sunglasses = [{'path': 'static/image/sgimg.jpg'}, {'path': 'static/image/sgimg1.jpg'},
                  {'path': 'static/image/sg11.jpg'}, {'path': "static/image/sgimg3.jpg"}, {'path': "static/image/sgimg4.jpg"}]
    shoes = [{'path': 'static/image/shoeus.jpg'}, {'path': 'static/image/shoeus1.jpg'},
             {'path': 'static/image/shoesimg7.jpg'}, {'path': "static/image/shoesimg4.jpg"}, {'path': "static/image/shoeus2.jpg"}]
    return render_template("store.html", result=[], shirts=shirts, shoes=shoes, sunglasses=sunglasses, watches=watches, handbags=handbags)

@app.route('/payment')
def payment():
    return render_template("payment.html")

@app.route('/info')
def address():
    return render_template("address.html")


@app.route('/offerwall', methods=["GET", "POST"])
def offerwall():
    if request.method == "POST":
        path = request.form["path"]
        result = feature_extraction(path, model)
        return render_template("buy.html", result=result, path=path)
    watches = [{'path': 'static/image/watch1.jpg'}, {'path': 'static/image/watch2.jpeg'}, {'path': 'static/image/watch3.jpg'},
               {'path': 'static/image/watch4.jpg'}, {'path': "static/image/watch5.jpg"}, {'path': "static/image/watch6.jpeg"}]
    shoes = [{'path': 'static/image/shoe1.jpg'}, {'path': 'static/image/shoe2.jpg'}, {'path': 'static/image/shoe3.jpg'},
             {'path': 'static/image/shoe4.jpg'}, {'path': "static/image/shoe5.jpg"}, {'path': "static/image/shoe6.jpg"}]
    tshirts = [{'path': 'static/image/tshirt7.jpg'}, {'path': 'static/image/tshirt8.jpg'}, {'path': 'static/image/tshirt2.jpg'},
               {'path': 'static/image/tshirt3.jpg'}, {'path': "static/image/tshirt4.jpeg"}, {'path': "static/image/tshirt5.jpg"}]
    jeans = [{'path': 'static/image/jeans1.jpeg'}, {'path': 'static/image/jeans6.jpeg'}, {'path': 'static/image/jeans2.jpeg'},
             {'path': 'static/image/jeans3.jpg'}, {'path': "static/image/jeans4.jpeg"}, {'path': "static/image/jeans5.jpeg"}]

    return render_template('offer.html', jeans=jeans, tshirts=tshirts, watches=watches, shoes=shoes)


@app.route('/signup')
def signup():
    return render_template('signup.html')


if __name__ == '__main__':
    app.run(debug=True)
