"""# Step2 Tag Filters"""

"""## Imports"""

import logging
import numpy as np
import os
import pickle
from datetime import datetime,timedelta
from tqdm import tqdm
from warnings import warn, simplefilter

"""## Inputs"""

#input
k=[i for i in range(101)]
pickle_file=".../2021-01-01-0000-CA-VGG16-vars"
save_dir="..."

#debugging and testing input
loglevel="INFO"

"""## Execution"""

#Preparations

#create ID for files
network=pickle_file.split("-")[-2]
now = datetime.utcnow()+timedelta(hours=0)
runid = now.strftime("%Y-%m-%d-%H%M")+"-LF"
if not save_dir:
  save_dir=os.getcwd()
save_vars=save_dir+"/"+runid+"-"+network+"-vars"
save_vars_names=save_vars+"-names"
save_vars_list=[]
save_vars_info=save_vars+"-info.txt"
with open(save_vars_info,"a") as q:
  q.write("Vars Info LF\n----------\n\n")
  q.write("This file gives an overview of the stored variables in pickle file "+save_vars+"\n")
  q.write("To load them use e.g.:\n")
  q.write("with open(\""+save_vars_names+"\",\"rb\") as r:\n")
  q.write("  var_names=pickle.load(r)\n")
  q.write("with open(\""+save_vars+"\",\"rb\") as p:\n")
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

simplefilter("always") # Change the filter in this process
os.environ["PYTHONWARNINGS"] = "always" # affect subprocesses
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

#load files from step1
with open(pickle_file+"-names", "rb") as p:
  varnames_step1 = pickle.load(p)
for i, j in enumerate(varnames_step1):
  varnames_step1[i] = j.split(":")[0]
with open(pickle_file,"rb") as p:
  for var in varnames_step1:
    globals()[var]=pickle.load(p)

logger.info("following variables loaded:"+str(varnames_step1))

def set_threshold(fil_act,k,percent=False):
  '''
  determines the most active filters by choosing the k or k% highest average activations

  input:
    fil_act: (averaged) activation of filters
    k: number indicating used method for threshold. 0 for layer average as threshold, else k-most activations per layer
    percent: boolean wether q-percent most active filters or k-most active filters
  
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
      warn("k too big for layer "+str(i)+". k is set to 0 (average)")
      l=0
      threshold=np.average(layer)
      ind=np.nonzero(layer>threshold)
    elif l<0:
      warn("k too small. k is set to 0 (average) for all layers")
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

def label_it(fil_act_t,label,labels_old=[]):
  '''
  labels the most active filters for each class with corresponding class label, adds label to labels_old for all filters above threshold determined by fil_act

  inputs:
  - fil_act_t: thresholded active filters; list with np-arrays for each layer as elements; these np-arrays contain the thresholded activations (0:filter inactive, 1: filter active)
  - label: name of the class
  - labels_old: list with labels (3.dim) of each filter (2.dim) of each layer (1.dim)

  returns:
  - labels_new: updated list of labels
  '''
  logger.debug("label_it")
  if not fil_act_t:
    logger.warning("Warning: label_it input is empty for this class")
    warn("label_it input is empty for this class")
    return []
    
  if not labels_old: #initialization
    labels_old=[None]*len(fil_act_t)
    for count,layerf in enumerate(fil_act_t):
      labels_old[count]=[None]*len(layerf)
      for step,filterf in enumerate(layerf):
        labels_old[count][step]=[]
  for count,layerf in enumerate(fil_act_t):
    for step,filterf  in enumerate(layerf):
      if filterf==1.0: #append if active
        labels_old[count][step].append(label)
  labels_new=labels_old
  return labels_new

#k labels
labels_filters=[[] for i in range(len(k))]
filterlabels={}
num_classes=len(labels_id)
for i,el in enumerate(tqdm(k)):
  for j in range(num_classes):
    logger.debug("determining most active filters, k="+str(el)+",class "+str(j+1)+" of "+str(num_classes))
    fil_act_t=set_threshold(globals()['filter_act_'+str(j)],el,False)
    labels_filters[i]=label_it(fil_act_t,labels_id[j],labels_filters[i])
  filterlabels[i]=labels_filters[i]
with open(save_vars,"wb") as p:  
  pickle.dump(filterlabels,p)
save_vars_list.append("labels_filters")
with open(save_vars_info,"a") as q:
  q.write("- filterlabels: labels of filters for k in "+str(k)+"\n")
logger.info("k labels added to: "+save_vars)

#k%labels
labels_filters_per=[[] for i in range(len(k))]
filterlabels_per={}
for i,el in enumerate(tqdm(k)):
  for j in range(num_classes): 
    logger.debug("determining most active filters, q="+str(el)+"%,class "+str(j+1)+" of "+str(num_classes))
    fil_act_t=set_threshold(globals()['filter_act_'+str(j)],el,True)
    labels_filters_per[i]=label_it(fil_act_t,labels_id[j],labels_filters_per[i])
  filterlabels_per[el]=labels_filters_per[i]
with open(save_vars,"ab") as p:
  pickle.dump(filterlabels_per,p)
save_vars_list.append("labels_filters_per")
with open(save_vars_names,"wb") as r:
  pickle.dump(save_vars_list,r)
with open(save_vars_info,"a") as q:
  q.write("- filterlabels_per: labels of filters for k% in "+str(k))
logger.info("k% labels added to: "+save_vars)
logger.info("---------------Run succesful-----------------")