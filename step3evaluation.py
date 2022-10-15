"""#Step3Evaluation"""

"""## Imports"""
import logging
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from collections import Counter
from datetime import datetime, timedelta
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from warnings import warn, simplefilter

"""## Inputs"""

classes = range(1000)
k = [1, 5, 25]
pickle_file = ".../2020-01-01-0000-LF-VGG16-vars"  # file from step 2
images_dir = ".../ILSVRC2012_img_train"  # directory with the imagefolder(s)
save_dir = ""
res_classes = None  # number classes to inspect, restricts the data

# same input as in step1
split = 0.2
seed_rn = 123
network = pickle_file.split("-")[-2]

# Test and debug
loglevel = "INFO"

"""## Execution"""

# Preparations

# create ID for files
now = datetime.utcnow() + timedelta(hours=0)
runid = now.strftime("%Y-%m-%d-%H%M")+"-E"
if not save_dir:
    save_dir = os.getcwd()
save_vars = save_dir + "/" + runid + "-" + network + "-vars"
save_vars_names = save_vars + "-names"
save_vars_list = []
save_vars_info = save_vars + "-info.txt"
save_vars_results = save_vars + "-results.txt"
with open(save_vars_info, "a") as q:
    q.write("Vars Info E\n----------\n\n")
    q.write("This file gives an overview of the stored variables in pickle file "
            + save_vars + "\n")
    q.write("To load them use e.g.:\n")
    q.write("with open(\"" + save_vars_names + "\",\"rb\") as r:\n")
    q.write("  var_names=pickle.load(r)\n")
    q.write("with open(\"" + save_vars + "\",\"rb\") as p:\n")
    q.write("  for var in var_names:\n")
    q.write("    globals()[var]=pickle.load(p)\n\n")
    q.write("stored variables are:\n")

# logging
logging.captureWarnings(True)
logger = logging.getLogger('logger1')
logger.setLevel(loglevel)
if (logger.hasHandlers()):
    logger.handlers.clear()
info_handler = logging.FileHandler(save_dir + "/" + runid + '-info.log')
info_handler.setLevel(logging.INFO)
infoformat = logging.Formatter('%(asctime)s - %(message)s',
                               datefmt='%d.%m.%Y %H:%M:%S')
info_handler.setFormatter(infoformat)
logger.addHandler(info_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(infoformat)
logger.addHandler(stream_handler)
simplefilter("always")
os.environ["PYTHONWARNINGS"] = "always"
warnings_logger = logging.getLogger("py.warnings")
warnings_handler = logging.FileHandler(save_dir + "/" + runid + '-warnings.log')
warnings_logger.addHandler(warnings_handler)
if logger.getEffectiveLevel() == 10:
    debug_handler = logging.FileHandler(save_dir + "/" + runid + '-debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debugformat = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
                                    datefmt='%d.%m.%Y %H:%M:%S')
    debug_handler.setFormatter(debugformat)
    logger.addHandler(debug_handler)
logger.propagate = False

# load files from step1
with open(pickle_file + "-names", "rb") as r:
    var_names = pickle.load(r)
with open(pickle_file, "rb") as p:
    for var in var_names:
        globals()[var] = pickle.load(p)

logger.info("following variables loaded:" + str(var_names))
logger.debug("preparations done")

# images
if "ception" in network:  # Inception networks have other image resolution as input
    img_rows = 299
    img_cols = 299
else:
    img_rows = 224
    img_cols = 224

# network
network_mod = {"VGG16": "vgg16",
               "VGG19": "vgg19",
               "InceptionV3": "inception_v3",
               "ResNet50": "resnet",
               "ResNet101": "resnet",
               "ResNet152": "resnet",
               "ResNet50V2": "resnet_v2",
               "ResNet101V2": "resnet_v2",
               "ResNet152V2": "resnet_v2",
               "InceptionResNetV2": "inception_resnet_v2"}
cnn = getattr(keras.applications, network)()  # import network
network_module = getattr(keras.applications, network_mod[network])
preprocess_input = getattr(network_module, "preprocess_input")
layers = []  # interesting layers, omitting pooling layer and fc
for i, layer in enumerate(cnn.layers):
    if 'conv' not in layer.name:
        continue
    layers.append(i)
outputs = [cnn.layers[i].output for i in layers]
outputs_model = keras.Model([cnn.input], outputs)


def preprocess_images_from_dir(i_dir, preprocess_fun, i_rows=224, i_cols=224,
                               seed_rn="123", val_split=0.2):
    """
    prepares the images in images_dir as input for the networks.

    input:
    - i_dir: path to the directory with image data, the folders in it should be
             named after the image class of the images they contain.
    - preprocess_fun: preprocessing function
    - i_rows: height of images
    - i_cols: width of images
    - seed_rn: random seed
    - val_split: fraction used for test set, 1-split is used for train set

    returns:
    - images_train: train set
    - images_test: test set

    remark:
    uses tensorflow.keras as keras and ImageDataGenerator from
    tensorflow.keras.preprocessing.image
    """
    # prepare datagenerator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fun,
                                 validation_split=val_split)
    # create dataframe with infos for imagespaths and shuffle it
    folders = os.listdir(i_dir)
    numfolders = len(folders)
    files = []
    target = []
    for folder in folders[0:numfolders]:
        imagepaths = os.listdir(i_dir + "/" + folder)
        for imagepath in imagepaths:
            files.append(i_dir + "/" + folder + "/" + imagepath)
            target.append(folder)
    df_images = pd.DataFrame({"imagepath": files, "wnid": target})
    df_images = df_images.sample(frac=1,
                                 random_state=seed_rn).reset_index(drop=True)
    # collect train images
    logger.info("Preparing Train Set")
    images_train = datagen.flow_from_dataframe(df_images, x_col='imagepath',
                                               y_col='wnid', batch_size=1,
                                               target_size=(i_rows, i_cols),
                                               shuffle=False, seed=seed_rn,
                                               subset="training")
    # collect test images
    logger.info("Preparing Test Set")
    images_test = datagen.flow_from_dataframe(df_images, x_col='imagepath',
                                              y_col='wnid', batch_size=1,
                                              target_size=(i_rows, i_cols),
                                              shuffle=False, seed=seed_rn,
                                              subset="validation")
    logger.info("images ready")
    return images_train, images_test


def get_activations(prep_image, outputs_model):
    """
    computes the activations for an image (prep_image) for network (cnn)
    in layers (hl)

    inputs:
    - prep_image: image
    - outputs_model: initialized keras model

    returns:
    - prep_acts: activations of the image
    """
    logger.debug("collecting activations")
    prep_acts = outputs_model([prep_image])
    logger.debug("activations collected")
    return prep_acts


def featuremap_values(acts):
    '''
    averages the activations of each feature map to get one single number for
    each feature map

    input:
    - acts: activations

    return:
    - featuremap_val: feature map values (averaged activations for each feature
      map)
    '''
    featuremap_val = [[] for x in range(len(acts))]
    for i, layer in enumerate(acts):
        featuremap_val[i] = layer.mean(axis=(0, 1, 2))

    return featuremap_val


def set_threshold(fil_act,k,percent=False):
  '''
  determines the most active filters by choosing the k or k% highest average activations

  input:
    fil_act: (averaged) activation of filters
    k: number indicating used method for threshold. 0 for layer average as threshold, else k-most activations per layer
    percent: boolean wether k-percent most active filters or k-most active filters
  output:
    fil_act_t: thresholded 'most active' filters
  '''
  logger.debug("determine most active filters")
  fil_act_t=fil_act.copy()
  for i,layer in enumerate(fil_act_t):
    l=k
    if percent:
      con=100
    else:
      con=len(layer)
    #select threshold method:
    if l==0:
      logger.debug("Selected Method for layer "+str(i)+" is average")
      threshold=np.average(layer)
      ind=np.nonzero(layer>threshold)
    elif k>con:
      logger.warning("k too big for layer "+str(i)+". k is set to 0 (average)")
      l=0
      threshold=np.average(layer)
      ind=np.nonzero(layer>threshold)
    elif l<0:
      logger.warning("k too small. k is set to 0 (average) for all layers")
      k=0
      l=0
      threshold=np.average(layer)
      ind=np.nonzero(layer>threshold)
    else:
      if percent:
        logger.debug("Calculating "+str(l)+"%-most active filters for layer "+str(i))
        p=int(round((l/100)*len(layer)))
        ind=np.argpartition(layer, -p)[-p:]
      else:
        logger.debug("Calculating "+str(l)+"-most active filters for layer "+str(i))
        ind=np.argpartition(layer, -l)[-l:]
    fil_act_t[i]=np.zeros(len(fil_act_t[i]))
    fil_act_t[i][ind]=1
  return fil_act_t



def get_labels(filterlabels, fil_act_t):
    '''
    assigns the corresponding labels from the filters of the network (filterlabels) 
    to the most active filters of the image (fil_act_t)

    inputs:
    - filterlabels: filterlabels (from step2)
    - fil_act_t: thresholded activations

    returns:
    - prep_fil_labels: for each filter either "inactive" or contains the corresponding 
      filterlabels from step2 if the filter belongs to the most active ones
    '''
    logger.debug("get labels of active filters")
    prep_fil_labels = {}
    for step, layer in enumerate(filterlabels):
        prep_fil_labels[step] = {}
        for count, fil in enumerate(layer):
            prep_fil_labels[step][count] = []
            if fil_act_t[step][count] == 1:
                for l in fil:
                    prep_fil_labels[step][count].append(l)
            else:
                prep_fil_labels[step][count] = "inactive"
    return prep_fil_labels


def load_assign_labels(k, networklabels, fil_act, scope):
    '''
    loads files with labels for the filters of the network and assigns them

    inputs:
    - k: list with the different values for k
    - networklabels: filterlabels from step2)
    - fil_act: filter activations
    - scope: either "k" or "k%"

    returns:
    - lab: labelled filters
    - fil_act_t: thresholded activations, for each filter 1 if above and 0 otherwise
    '''
    logger.debug("loading and assigning labels")
    lab = {}
    fil_act_t = {}
    for i, el in enumerate(k):
        try:
            networklabels[el]
        except KeyError:
            warn("No labels available for " + scope + "=" + str(el))
            continue
        if not any(networklabels[el]):
            warn("No labels available for " + scope + "=" + str(el))
            continue
        fil_act_t[el] = set_threshold(fil_act, el)
        lab[el] = get_labels(networklabels[el], fil_act_t[el])
        logger.debug(scope + "=" + str(el) + "labels:\n" + str(lab[el]))
        logger.debug(scope + "=" + str(el) + "active filters:\n" + str(fil_act_t[el]))

    return lab, fil_act_t


def counting_labels(lab):
    '''
    counts the labels of the filters

    inputs:
    - lab: labels of each filter

    returns:
    - counts: labelcounts for each layer and k
    - uniquelabels: filters with only one label
    '''
    logger.debug("counting labels")
    counts = {}
    uniquelabels = {}
    for i, el in enumerate(lab):
        uniquelabels[el] = []
        counts[el] = [None] * len(lab[el])
        for j, layer in enumerate(lab[el]):
            counts[el][j] = Counter()
            for m, fil in enumerate(lab[el][j]):
                if lab[el][j][m] is not "inactive":
                    for label in lab[el][j][m]:
                        counts[el][j][label] += 1
                        if len(lab[el][j][m]) == 1:
                            uniquelabels[el].append([j, m, label])
                            logger.debug(
                                "unique label in layer " + str(j) + ", filter " + str(m) + " for k or k% =" + str(
                                    el) + ": " + str(label))
    return counts, uniquelabels


def totalcounts_predict(counts):
    '''
    computes the total counts for the labels and the predictions based on labels of last layer (appred_ll) and labels of all layers (appred)

    inputs:
    - counts: labelcounts for each layer and k

    returns:
    - appred: totallabelcounts for all layers together
    - appred_ll: totallabelcounts for last layer
    '''
    logger.debug("computing totalcounts and predictions")
    totalcounts = {}
    appred = {}
    appred_ll = {}
    for i, el in enumerate(counts):
        totalcounts[el] = Counter()
        for j, layer in enumerate(counts[el]):
            totalcounts[el] = totalcounts[el] + layer
        appred[el] = totalcounts[el].most_common()  # change counter to list object
        appred_ll[el] = counts[el][len(counts[el]) - 1].most_common()
    return appred, appred_ll


# load images
train, test = preprocess_images_from_dir(images_dir, img_rows, img_cols, cnn, seed_rn, split, res_classes)

# extract class labels
test_labels_id_inv = test.class_indices  # get dict with mapping of labels
test_labels_id = {y: x for x, y in test_labels_id_inv.items()}  # invert dict

num_classes = len(test_labels_id)

for image_class in tqdm(range(num_classes)):
    logger.info("class: " + str(test_labels_id[image_class]) + " (labelnum:" + str(image_class) + ")")

    # reset the dicts every step to save RAM
    y = {}  # true label for each image
    y_hat = {}  # prediction(s) of k for each image
    y_hat_ll = {}  # prediction(s) of k for each image according to last layer
    y_hat_per = {}  # prediction(s) of k% for each image
    y_hat_per_ll = {}  # prediction(s) of k% for each image according to last layer
    class_images_dict = {}  # images of each class

    y[image_class] = {}  # add class info
    y_hat[image_class] = {}  # add class info
    y_hat_ll[image_class] = {}  # add class info
    y_hat_per[image_class] = {}  # add class info
    y_hat_per_ll[image_class] = {}  # add class info
    class_images = [test[j] for j in range(test.n) if test.labels[j] == image_class]
    class_images_dict[image_class] = [test.filenames[j] for j in range(test.n) if test.labels[j] == image_class]
    numimages = len(class_images)
    logger.debug("number of images in class " + str(image_class) + ": " + str(len(class_images)))
    if not class_images:
        warn("no images for class " + str(test_labels_id[image_class]) + " (labelnum:" + str(
            image_class) + ") in train set")
        continue
    for imgnum, image in enumerate(class_images):
        logger.info(
            "image " + str(imgnum + 1) + " of " + str(numimages) + " (class " + str(image_class + 1) + " of " + str(
                num_classes) + ")")

        # True Label (Prediction of the network)
        preds = cnn(image[0]).numpy()  # [0]imagearray,[1]label
        y[image_class][imgnum] = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=3)

        # comppute activations of filters
        logger.debug("collect activations and labels")
        out = get_activations(image, outputs_model)
        fil_act = featuremap_values(out)

        # get labels of active filters
        lab_k, fil_act_t_k = load_assign_labels(k, labels_filters, fil_act, "k")
        lab_k_per, fil_act_t_k_per = load_assign_labels(k, labels_filters_per, fil_act, "k%")

        # statistics
        ##counting the labels of the filters for each layer
        logger.debug("get counts and predictions")
        counts_k, uniquelabel_k = counting_labels(lab_k)
        counts_k_per, uniquelabel_k_per = counting_labels(lab_k_per)

        ##totalcounts(predictions)
        y_hat[image_class][imgnum], y_hat_ll[image_class][imgnum] = totalcounts_predict(counts_k)
        y_hat_per[image_class][imgnum], y_hat_per_ll[image_class][imgnum] = totalcounts_predict(counts_k_per)
    with open(save_vars + str(image_class), "wb") as p:
        for el in [y, y_hat, y_hat_ll, y_hat_per, y_hat_per_ll, class_images_dict]:
            pickle.dump(el, p)
    logger.info(
        "saved intermediate predictions (y,y_hat,y_hat_per,y_hat_ll,y_hat_per_ll,class_images_dict) in:" + save_vars + str(
            image_class))

# info about saved variables
for el in ["y", "y_hat", "y_hat_per", "y_hat_ll", "y_hat_per_ll", "class_images_dict"]:
    save_vars_list.append(el)
with open(save_vars_names, "wb") as r:
    pickle.dump(save_vars_list, r)
with open(save_vars_info, "a") as txt:
    txt.write("y: predictions of cnn\n")
    txt.write("y_hat: labelcounts (predictions) of approach for different k\n")
    txt.write("y_hat_per: labelcounts (predictions) of approach for different q%\n")
    txt.write("y_hat_ll: labelcounts (predictions) of last layer for different k\n")
    txt.write("y_hat_per_ll: labelcounts (predictions) of last layer for different q%\n")
    txt.write("class_images_dict: contains the corresponding file names for each class\n")