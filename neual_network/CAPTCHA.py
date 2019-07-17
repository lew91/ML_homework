import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import skimage.transform as tf
from skimage.measure import label, regionprops
from sklearn.utils import check_random_state

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


random_state = check_random_state(14)
letters = list("ABCDEFGHIGKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)
scale_values = np.arange(0.5, 1.5, 0.1)


def create_captcha(text, shear=0, size=(100, 30), scale=1):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r"/library/fonts/Hack-Regular.ttf", 22)
    draw.text((0, 0), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    image = image / image.max()
    # Apply scale
    shape = image.shape
    shapex, shapey = (int(shape[0] * scale), int(shape[1] * scale))
    image = tf.resize(image, (shapex, shapey))
    return image


def segment_image(image):
    # label will find subimages of connected non-black pixels
    labeled_image = label(image > 0.2, connectivity=1, background=0)
    subimages = []
    # regionprops splits up the subimages
    for region in regionprops(labeled_image):
        # extract the subimage
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
        if len(subimages) == 0:
            # no subimages found, so return the entire image
            return [image, ]
    return subimages


def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    scale = random_state.choice(scale_values)
    # we use 30, 30 as the image size to ensure we get all the text in the image
    return create_captcha(letter, shear=shear, size=(30, 30), scale=scale), letters.index(letter)


#############################################
# image, target = generate_sample(random_state)
# plt.imshow(image, cmap='Greys')
# print("The target for this image is: {0}".format(letters[target]))
# plt.savefig('./image/random_temp.png')

#####################

# TODO: set(target), missing index 9, ???
dataset, target = zip(*(generate_sample(random_state) for i in range(1000)))
dataset = np.array([tf.resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
dataset = np.array(dataset, dtype='float')
target = np.array(target)

# transform
onehot = OneHotEncoder()
y = onehot.fit_transform(target.reshape(target.shape[0], 1))
y = y.todense()    # to Numpy array
X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)


clf = MLPClassifier(hidden_layer_sizes=(100, ), random_state=14)
#print(clf.get_params())

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
##
print("the f1 score is: {0:.3f}".format(f1_score(y_pred=y_pred, y_true=y_test, average='macro')))
print("Classification report \n")
print(classification_report(y_pred=y_pred, y_true=y_test))


def predict_captcha(captcha_image, neural_network):
    subimages = segment_image(captcha_image)
    # Perform the same tarnsformations we did for our training data
    dataset = np.array([np.resize(subimage, (20, 20)) for subimage in subimages])
    X_test = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
    # Use predict proba and argmax to get the most likely prediction
    y_pred = neural_network.predict_proba(X_test)
    predictions = np.argmax(y_pred, axis=1)

    # Convert predictions to letters
    predicted_word = str.join("", [letters[prediction] for prediction in predictions])
    return predicted_word


def test_prediction(word, net=clf, shear=0.2, scale=1):
    captcha = create_captcha(word, shear=shear, scale=scale, size=(len(word) * 25, 30))
    prediction = predict_captcha(captcha, net)
    return word == prediction, word, prediction

#######################################

