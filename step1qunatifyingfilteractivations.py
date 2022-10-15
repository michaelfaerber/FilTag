"""#Step1 Quantifying Filter Activations"""

# Imports
import logging
import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

"""## Inputs"""

# Input
network = "VGG16"
images_dir = ".../FilterLabel/ILSVRC2012_img_train"
save_dir = ".../FilterLabel/codes/resultstest4"
seed_rn = 123
split = 0.2  # data taken for validation
time_delta=0

# debugging and testing input
loglevel = "INFO"

"""## Preparations"""

# create ID and paths for files
now = datetime.utcnow() + timedelta(hours=time_delta)
runid = now.strftime("%Y-%m-%d-%H%M")+'-CA'
if not save_dir:
    save_dir = os.getcwd()
save_vars = save_dir + "/" + runid + "-" + network + "-vars"
save_vars_names = save_vars + "-names"
save_vars_list = []
save_vars_info = save_vars + "-info.txt"

# create info file
with open(save_vars_info, "a") as q:
    q.write("Vars Info CA\n----------\n\n")
    q.write("This file gives an overview of the stored variables in pickle file"
            " "+ save_vars + "\n")
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
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()

info_handler = logging.FileHandler(save_dir + "/" + runid + '-info.log')
info_handler.setLevel(loglevel)
infoformat = logging.Formatter('%(asctime)s - %(message)s', 
                               datefmt='%d.%m.%Y %H:%M:%S')
info_handler.setFormatter(infoformat)
logger.addHandler(info_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(infoformat)
logger.addHandler(stream_handler)

os.environ["PYTHONWARNINGS"] = "always"
warnings_logger = logging.getLogger("py.warnings")
warnings_handler = logging.FileHandler(save_dir + "/" + runid + 
                                       '-warnings.log')
warnings_logger.addHandler(warnings_handler)
if logger.getEffectiveLevel() == 10:
    debug_handler = logging.FileHandler(save_dir + "/" + runid + 
                                        '-debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debugformat = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s'
    ' - %(message)s',
                                    datefmt='%d.%m.%Y %H:%M:%S')
    debug_handler.setFormatter(debugformat)
    logger.addHandler(debug_handler)
logger.propagate = False

# images
if "ception" in network:  # Inception networks have other image resolution
    img_rows = 299
    img_cols = 299
else:
    img_rows = 224
    img_cols = 224

# network and preprocessing function
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
logger.info("importing network:")
cnn = getattr(keras.applications, network)()
network_module = getattr(keras.applications, network_mod[network])
preprocess_input = getattr(network_module, "preprocess_input")
layers = [] # interesting layers (conv), omitting pooling layer and fc
for i, layer in enumerate(cnn.layers):
    if 'conv' not in layer.name:
        continue
    layers.append(i)
outputs = [cnn.layers[i].output for i in layers]
outputs_model = keras.Model([cnn.input], outputs)

"""## functions"""

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

def get_activations(imgs, nn_model):
    """
    computes the average or max activations of prep_images according to nn_model

    input:
    - imgs (list): image data as np.arrays
    - nn_model (keras.model): initialized keras model to obtain the activations

    returns:
    - filter_act: featuremap values (activations of filters)
    """
    for step,img in enumerate(tqdm(imgs)):#colour="blue"
        raw_acts=nn_model.predict(img)
        out_pic_new = scale_out(raw_acts)
        fm_val_new = featuremap_values(out_pic_new)
        if step==0:
            prep_act = fm_val_new.copy()
        else:
            for count, layer in enumerate(prep_act):
                prep_act[count] = (fm_val_new[count] * (1 / (step + 1)) +
                                   fm_val[count] * (step / (step + 1)))
        fm_val = prep_act.copy()
    return fm_val

def scale_out(act_raw):
    '''
    scales the activations of each layer to a range of [0,1]

    input:
    - a: raw activations

    return:
    - b: scaled activations
    '''
    act_scaled = act_raw.copy()
    scaler = MinMaxScaler(copy=False)  # inplace operation
    for layer in act_scaled:
        layer = layer.reshape(-1, 1)
        scaler.fit_transform(layer)
    return act_scaled

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
        featuremap_val[i] = layer.mean(axis=(0,1,2))

    return featuremap_val

""" #Execution """

logger.info("Selected network is " + network)

# preprocess images
train, test = preprocess_images_from_dir(images_dir, preprocess_input, img_rows,
                                         img_cols,  seed_rn, split)

# extract class labels
labels_id_inv = train.class_indices  # get dict with mapping of labels
labels_id = {y: x for x, y in labels_id_inv.items()}  # invert dict
with open(save_vars, "wb") as p:
    pickle.dump(train.filenames, p)
    pickle.dump(test.filenames, p)
    pickle.dump(labels_id, p)
save_vars_list.append("train_filenames")
save_vars_list.append("test_filenames")
save_vars_list.append("labels_id")
with open(save_vars_info, "a") as q:
    q.write("- train_filenames: list with filenames of train set\n")
    q.write("- test_filenames: list with filenames of test set\n")
    q.write("- labels_id: dictionary of numerical train labels and "
            "corresponding folder name (class)\n")
logger.info("class labels added to " + save_vars)
num_classes = len(labels_id)
outputs = [cnn.layers[i].output for i in layers]
outputs_model = keras.Model([cnn.input], outputs)

fil_act = []
for i in tqdm(range(num_classes)):#colour=green
    logger.info("class: " + str(labels_id[i]) + " (labelnum:" + str(i) + ")")
    class_images = [train[j][0] for j in range(train.n) if train.labels[j] == i]
    out = get_activations(class_images, outputs_model)
    logger.info("saving class activations of class "+str(labels_id[i]))
    with open(save_vars,"ab") as p:
        pickle.dump(out, p)
    save_vars_list.append("filter_act_"+str(i)+": filter activations of class "+str(labels_id[i]))
    with open(save_vars_names, "wb") as p: #overwrites vars list
        pickle.dump(save_vars_list, p)
    with open(save_vars_info, "a") as txt:
        txt.write("- filter_act_"+str(i)+": filter activations of class "+str(labels_id[i])+"\n")

logger.info("---------------Run succesful-----------------")