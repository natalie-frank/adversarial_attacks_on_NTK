import os
import torch
from matplotlib import pyplot
import platform


#sets current working directory
def set_cwd():
    if platform.platform() == "Windows-10-10.0.19045-SP0":#current working directory on your local computer
        cwd=os.path.join("C:\\Users","Natalie","Documents","PNNL code","bitbucket")
    else: #current working directory on cluster
        cwd=os.path.join(os.sep,"rcfs","projects","task0_pmm","natnew")
    os.chdir(cwd)

#checkes if the directory dir_name exists, and creates it if it doesn't
#returns true if the directory was created and false otherwise
def exist_and_create_dir(dir_name):
    ext=os.path.isdir(dir_name)
    if not ext:
        os.mkdir(dir_name)
    return ~ext



#net: a neural net
#xs: a tensor of data points, the first dimension is assumed to be the batch dimension
#class_names: the names of each class corresponding to an output neuron, can be none

#if class_names is None, returns a tensor of the index of the maximum output neuron at each data point in xs.
#otherwise resturns a list of the predicted classes of each data point in xs as specified by class_names 
def predict_class(net, xs, class_names=None):
    ys=net(xs)
    inds=torch.max(ys,1)[1]
    if class_names is None:
        return inds
    else:
        lst=[class_names[i.item()] for i in inds]
        return lst

def plot_grayscale(x):
    pyplot.imshow(x, cmap=pyplot.get_cmap('gray'))
    pyplot.show()
