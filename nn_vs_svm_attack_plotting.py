import torch
import os
from train_binary_MNIST_sigmoid_networks import Net_sigmoid
import binary_torch_svm
import ntk_methods
import numpy as np
import adversaries
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
import basic_methods
import pickle
import time

#model: the model under attack
#input_attack: the attack function for attacking the model. This function takes (model,x,label,epsilon), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
#eps: the perturbation radius for the adversarial attack
#test_set: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
#save: if None, adversarial examples are not saved. Otherwise, the examples are saved in the location specified by 'save'. The examples and the labels are saved as separate torch tensors

#for generating and saving adversarial examples to a model
def gen_adversarial_examples(model,input_attack,eps,test_set, save=None):
    data_size=len(test_set)
    ys=torch.zeros(data_size)
    x0=test_set[0][0]
    shp=list(x0.shape)
    shp[0]=data_size
    adversarial_examples=torch.zeros(shp)
    if eps==0:
        attack=adversaries.trivial_attack
    else:
        def attack(model,inputs,targets):
            return input_attack(model,inputs,targets,eps)
    for i in range(data_size):        
        x,y=test_set[i]
        example=attack(model,x,torch.tensor([y]))
        adversarial_examples[i]=example[0]
        ys[i]=y
    if save is not None:
        torch.save(adversarial_examples,save+"_x.pt")
        torch.save(ys, save+"_y.pt")
    return (adversarial_examples,ys)



#eps_list: list of perturbation radiuses at which the models were trained
#nets: list of lists of neural nets, nets[i] is a list of neural nets adversarially trained with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#svms: list of lists binary_torch_SVMs, svms[i] is a list of binary_torch_SVMs trained with the NTK corresponding to the neural net with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#test_set: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
#nets_examples: the adversarial examples for the neural nets. Nested list depth 3 with strucure: epsilon, trial, (adversarial_examples,labels)
#svms_examples: the adversarial examples for the svms. Nested list depth 3 with strucure: epsilon, trial, (adversarial_examples,labels)
#save_fig_folder: if save_fig_folder is not None, the resulting figure would be saved in save_fig_folder

#for each neural net in nets[k][i] and binary_torch_SVM in svms[k][i], computes the average classification error under a white-box attack
def white_box_attacks_plots_vs_eps(eps_list,nets,svms,nets_examples,
                                   svms_examples,save_fig_folder=None):
    net_to_net_errors=[0]*len(eps_list)
    net_to_net_sd=[0]*len(eps_list)
    svm_to_svm_errors=[0]*len(eps_list)
    svm_to_svm_sd=[0]*len(eps_list)
    for i in range(len(eps_list)):

        nets_list=nets[i]
        svms_list=svms[i]

        nets_examples_list=nets_examples[i]
        svms_examples_list=svms_examples[i]

        #errors and standard deviations of a neural net
        nn_errors_eps=[classification_error_under_attack(nets_examples_list[j],
                                                         nets_list[j],True) for j in 
                                                         range(len(nets_list))]
        nn_errors_eps=np.asarray(nn_errors_eps)
        net_to_net_errors[i]=np.mean(nn_errors_eps)
        net_to_net_sd[i]=np.std(nn_errors_eps)

        #errors and standard deviations of the svm
        svm_errors_eps=[classification_error_under_attack(svms_examples_list[j],
                                                          svms_list[j],False) for j 
                                                          in range(len(svms_list))]
        svm_errors_eps=np.asarray(svm_errors_eps)
        svm_to_svm_errors[i]=np.mean(svm_errors_eps)
        svm_to_svm_sd[i]=np.std(svm_errors_eps)
       



    #plot errors
    net_to_net_color=(0,.5,0)
    svm_to_svm_color=(0,0,0)
    net_to_net_label="neural net-to-neural net"
    svm_to_svm_label="SVM-to-SVM"
    net_to_net_marker='o'
    svm_to_svm_marker='s'

    plot_error_lines(eps_list,net_to_net_errors,net_to_net_sd,
                            net_to_net_color,net_to_net_label,net_to_net_marker)
    plot_error_lines(eps_list,svm_to_svm_errors,svm_to_svm_sd,
                            svm_to_svm_color,svm_to_svm_label,svm_to_svm_marker)

    plt.xlabel("perturbation radius")
    plt.ylabel("classification error")
    ax = plt.gca()
    ax.set_ylim([0, 0.6])

    plt.title("Error Under Attack")
    plt.legend()

    plt.savefig(os.path.join(save_fig_folder,"white_box"),bbox_inches='tight')
    plt.close()


            #saving the plot data
    variables_dict={"epsilons":eps_list, 
                    "net_mean_errors":net_to_net_errors, "net_sds":net_to_net_sd,
                    "svm_mean_errors":svm_to_svm_errors, "svms_sds":svm_to_svm_sd}
    file=open(os.path.join(save_fig_folder,"white_box_attacks_plot_data.p"),'wb')
    pickle.dump(variables_dict,file)
    file.close()



#eps_list: list of perturbation radiuses at which the models were trained
#nets: list of lists of neural nets, nets[i] is a list of neural nets adversarially trained with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#svms: list of lists binary_torch_SVMs, svms[i] is a list of binary_torch_SVMs trained with the NTK corresponding to the neural net with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#test_set: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
#nets_examples: the adversarial examples for the neural nets. Nested list depth 3 with strucure: epsilon, trial, (adversarial_examples,labels)
#svms_examples: the adversarial examples for the svms. Nested list depth 3 with strucure: epsilon, trial, (adversarial_examples,labels)
#save_fig_folder: if save_fig_folder is not None, the resulting figure would be saved in save_fig_folder

#for each neural net in nets[k][i] computes the average classification error on adversarial examples found by attacking svms[k][i] and
#for each binary_torch_SVM in svms[k][i] computes the average classification error on adversarial examples found by attacking nets[k][i]

#assumes that svms[k][i] is the binary torch svm found by training with the NTK corresponding to nets[k][i]
def grey_box_attacks_plots_vs_eps(eps_list,nets,svms,nets_examples,svms_examples,
                                  save_fig_folder=None):
    svm_to_net_errors=[0]*len(eps_list)
    svm_to_net_sd=[0]*len(eps_list)
    net_to_svm_errors=[0]*len(eps_list)
    net_to_svm_sd=[0]*len(eps_list)
    for i in range(len(eps_list)):
        nets_list=nets[i]
        svms_list=svms[i]

        nets_examples_list=nets_examples[i]
        svms_examples_list=svms_examples[i]

        #errors and standard deviations of a neural net
        nn_errors_eps=[classification_error_under_attack(svms_examples_list[j],
                                                         nets_list[j],True) for j 
                                                         in range(len(nets_list))]
        nn_errors_eps=np.asarray(nn_errors_eps)
        svm_to_net_errors[i]=np.mean(nn_errors_eps)
        svm_to_net_sd[i]=np.std(nn_errors_eps)

        #errors and standard deviations of the svm
        svm_errors_eps=[classification_error_under_attack(nets_examples_list[j],
                                                          svms_list[j],False) for j 
                                                          in range(len(nets_list))] 
        svm_errors_eps=np.asarray(svm_errors_eps)
        net_to_svm_errors[i]=np.mean(svm_errors_eps)
        net_to_svm_sd[i]=np.std(svm_errors_eps)
       



       #plot errors

    svm_to_net_color=(.85,0,0)
    net_to_svm_color=(0,0,.85)

    svm_to_net_label="SVM-to-neural net"
    net_to_svm_label="neural net-to-SVM"
    svm_to_net_marker='+'
    net_to_svm_marker='d'
    
    plot_error_lines(eps_list,svm_to_net_errors,svm_to_net_sd,
                            svm_to_net_color,svm_to_net_label,svm_to_net_marker)
    plot_error_lines(eps_list,net_to_svm_errors,net_to_svm_sd,
                            net_to_svm_color,net_to_svm_label,net_to_svm_marker)


    plt.xlabel("perturbation radius")
    plt.ylabel("classification error")
    ax = plt.gca()
    ax.set_ylim([0, 0.6])

    plt.title("Error Under Transfer Attack of Neural Nets and Associated SVMs")
    plt.legend()

    plt.savefig(os.path.join(save_fig_folder,"grey_box"),bbox_inches='tight')
    plt.close()


        #saving the plot data
    variables_dict={"epsilons":eps_list, 
                    "net_mean_errors":svm_to_net_errors, "net_sds":svm_to_net_sd,
                    "svm_mean_errors":net_to_svm_errors, "svms_sds":net_to_svm_sd}
    file=open(os.path.join(save_fig_folder,"grey_box_attacks_plot_data.p"),'wb')
    pickle.dump(variables_dict,file)
    file.close()



#eps_list: list of perturbation radiuses at which the models were trained
#nets: list of lists of neural nets, nets[i] is a list of neural nets adversarially trained with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#svms: list of lists binary_torch_SVMs, svms[i] is a list of binary_torch_SVMs trained with the NTK corresponding to the neural net with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#test_set: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
#nets_examples: the adversarial examples for the neural nets. Nested list depth 3 with strucure: epsilon, trial, (adversarial_examples,labels)
#svms_examples: the adversarial examples for the svms. Nested list depth 3 with strucure: epsilon, trial, (adversarial_examples,labels)
#save_fig_folder: if save_fig_folder is not None, the resulting figure would be saved in save_fig_folder


#for each perturbation radius in eps_list, computes the average classification error of black box attacks computed by attacking different models.
#for example, nets[k][i] would be attacked by nets[k][j] and svms[k][j] for j!=i
def black_box_attacks_plots_vs_eps(eps_list,nets,svms,nets_examples,svms_examples,
                                   save_fig_folder=None):
    net_to_net_errors=[0]*len(eps_list)
    net_to_net_sd=[0]*len(eps_list)
    svm_to_net_errors=[0]*len(eps_list)
    svm_to_net_sd=[0]*len(eps_list)
    net_to_svm_errors=[0]*len(eps_list)
    net_to_svm_sd=[0]*len(eps_list)
    svm_to_svm_errors=[0]*len(eps_list)
    svm_to_svm_sd=[0]*len(eps_list)
    for i in range(len(eps_list)):
        eps=eps_list[i]
        #errors and standard deviations of a net_vs_net
        net_to_net_errors[i],net_to_net_sd[i]=average_non_matching_attack_mean_and_stds(
            nets_examples[i],nets[i],True)
        svm_to_net_errors[i],svm_to_net_sd[i]=average_non_matching_attack_mean_and_stds(
            svms_examples[i],nets[i],True)
        net_to_svm_errors[i],net_to_svm_sd[i]=average_non_matching_attack_mean_and_stds(
            nets_examples[i],svms[i],False)
        svm_to_svm_errors[i],svm_to_svm_sd[i]=average_non_matching_attack_mean_and_stds(
            svms_examples[i],svms[i],False)
       



       #plot errors
    net_to_net_color=(0,.5,0)
    svm_to_svm_color=(0,0,0)
    svm_to_net_color=(.85,0,0)
    net_to_svm_color=(0,0,.85)
    net_to_net_label="neural net-to-neural net"
    svm_to_svm_label="SVM-to-SVM"
    svm_to_net_label="SVM-to-neural net"
    net_to_svm_label="neural net-to-SVM"
    net_to_net_marker='o'
    svm_to_svm_marker='s'
    svm_to_net_marker='+'
    net_to_svm_marker='d'
    plot_error_lines(eps_list,net_to_net_errors,net_to_net_sd,
                            net_to_net_color,net_to_net_label,net_to_net_marker)
    plot_error_lines(eps_list,svm_to_net_errors,svm_to_net_sd,
                            svm_to_net_color,svm_to_net_label,svm_to_net_marker)
    plot_error_lines(eps_list,net_to_svm_errors,net_to_svm_sd,
                            net_to_svm_color,net_to_svm_label,net_to_svm_marker)
    plot_error_lines(eps_list,svm_to_svm_errors,svm_to_svm_sd,
                            svm_to_svm_color,svm_to_svm_label,svm_to_svm_marker)

    plt.xlabel("perturbation radius")
    plt.ylabel("classification error")
    plt.title("Error Under Transfer Attacks")
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0, 0.6])

    plt.savefig(os.path.join(save_fig_folder,"black_box"),bbox_inches='tight')
    plt.close()

    #saving the plot data
    variables_dict={"epsilons":eps_list, 
                    "net_to_net_mean_error":net_to_net_errors, "net_to_net_sd":net_to_net_sd,
                    "net_to_svm_mean_error":net_to_svm_errors, "net_to_svm_sd":net_to_svm_sd, 
                    "svm_to_net_mean_error":svm_to_net_errors, "svm_to_net_sd":svm_to_net_sd,
                    "svm_to_svm_mean_error":svm_to_svm_errors, "svm_to_svm_sd":svm_to_svm_sd}
    file=open(os.path.join(save_fig_folder,"black_box_attacks_plot_data.p"),'wb')
    pickle.dump(variables_dict,file)
    file.close()


#eps_list: list of perturbation radiuses to plot on x-axis
#means: list of mean error at each perturbation radius
#stds: list of standard deviations at each perturbation radius
#color: color for plotting lines

#helper function for plotting mean error with standard deviations
def plot_error_lines(eps_list,means,stds,color,label,marker):
    plt.plot(eps_list,means,color=color,label="_"+label )#plot mean error of nets
    plt.scatter(eps_list,means,color=color,label=label ,marker=marker)
    nn_upper_sd=[means[i]+2*stds[i] for i in range(len(eps_list))]
    nn_lower_sd=[means[i]-2*stds[i] for i in range(len(eps_list))]
    plt.plot(eps_list, nn_upper_sd, label="_"+label+"_upper_sd",linestyle="--",color=color)
    plt.plot(eps_list, nn_lower_sd, label="_"+label+"nn_lower_sd",linestyle="--",color=color)


#assumes attack_models and test_models are the same length,
#computes the average error of attacking test_models[i] with adversarial examples from attack_models[j] for i !=j
#adversarial_examples: adversarial examples for the models under attack. Nested list depth 2 with strucure: trial, (adversarial_examples,labels)
#test_models: the models on which to evaluate the adversarial examples. Assumes that these are either all neural nets with two outputs or all binary_torch_SVMs
#test_model_neural_net: set to true if test_models is a list of neural nets and false if its a list of binary_torch_SVMs
def average_non_matching_attack_mean_and_stds(adversarial_examples,test_models,test_model_neural_net):
    M=len(adversarial_examples)
    mns=[]
    for test_index in range(M):
        for attack_index in range(M):
            if test_index != attack_index:
                error=classification_error_under_attack(adversarial_examples[attack_index],
                                                        test_models[test_index],
                                                        test_model_neural_net)
                mns.append(error)
    mns=np.asarray(mns)
    mean_error=np.mean(mns)
    sd=np.std(mns)
    return mean_error,sd
    



#eps: the radius of perturbation used in adversarial training
#train_set: the data set on which to compute the NTK. can be a data set object or a list of (x,label) pairs

#loads pre-trained models, extracts the associated ntks, and trains the associated svms. Returns a list of neural nets and a list of svms
def load_models(eps,train_set):
    Y=[label for (x,label) in train_set]

    trial_name_base="trial"
    parent_parent_dir="models"
    parent_dir = "binary_sigmoid_models"
    models_list=[]
    svm_list=[]
    for i in range(0,10):
        trial_name="'"+trial_name_base+str(i)+"'"
        # Directory

        path = os.path.join(parent_parent_dir,parent_dir)
        prefix=os.path.join(path,trial_name)
        prefix=os.path.join(prefix,trial_name)
        model_name=prefix+"_epsilon="+str(eps)+"model.pt"
        net=Net_sigmoid()
        state_dict=torch.load(model_name,map_location=torch.device('cpu'))
        net.load_state_dict(state_dict)
        net.eval()
        models_list.append(net)
        feature_mapping=ntk_methods.feature_mapping(net, train_set, data_set=True, 
                                                    higher_derivatives=False)
        svm=ntk_methods.svm_from_kernel_matrix(feature_mapping,Y)
        torch_svm=binary_torch_svm.binary_torch_SVM(net,svm,feature_mapping)
        svm_list.append(torch_svm)
    
    return models_list,svm_list



  



#adversarial_examples: adversarial examples for the models under attack. These are a tuple of (adversarial_examples,labels)
#test_model: the model on which to evaluate the adversarial examples. Assumes that this model is either a neural net with two outputs or a binary_torch_SVM
#test_model_neural_net: set to true if test_models is a list of neural nets and false if its a list of binary_torch_SVMs 

#computes the classification error of a model under attack. The model under attack is test_model, adversarial examples are listed in adversarial_examples
def classification_error_under_attack(adversarial_examples,test_model,test_model_neural_net):
    total=0
    correct=0

    xs=adversarial_examples[0]
    ys=adversarial_examples[1]

    data_size=ys.shape[0]
    for i in range(data_size):
        adversarial_example=xs[i,:]
        adversarial_example=adversarial_example[None]
        label=torch.tensor([ys[i].item()])
        total=total+1
        if predict(test_model,adversarial_example,test_model_neural_net)==label:
            correct=correct+1
    return 1-correct/total






#model:either a neural net or a binary_torch_SVM
#x: a tensor at which we want the model prediction
#neural_net: true if model is a neural net, false if it is a binary_torch_SVM

#function for predicting the output of a model at a point x
def predict(model, x, neural_net):
    y=model(x)[0]
    if neural_net:
        if y[0].item()>y[1].item():
            return 0
        else:
            return 1
    else:
        if y.item()<0:
            return 0
        else:
            return 1


#the identity function; to assist in attacking a binary_torch_SVM
def identity_loss_function(x,label):
    return x



def main():

    mnist_train_set= datasets.MNIST('..\data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),]))

    #the models were trained on the subset of MNIST comprised of just 1s and 7s. We extract this subset of the training set
    mnist_train_sbset=[]
    for (data, label) in mnist_train_set:
        data=data[None,:]
        if label==1:
            mnist_train_sbset.append((data,label))
        elif label==7:
            mnist_train_sbset.append((data,0))
    #We extract the 1s and 7s of the test set
    #mnist_location=os.path.join()
    mnist_test_set= datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),]))
    mnist_test_sbset=[]
    for (data, label) in mnist_test_set:
        data=data[None,:]
        if label==1:
            mnist_test_sbset.append((data,label))
        elif label==7:
            mnist_test_sbset.append((data,0))
    




    #list of perturbation radiuses for which the models were trained
    eps_list=[0,0.05,0.1,0.15,0.2,0.25, 0.3]

#uncomment these lines to test the code on a small portion of MNIST
    #mnist_train_sbset=mnist_train_sbset[0:10]
    #mnist_test_sbset=mnist_test_sbset[0:2]

    #uncomment the following line to test on a smaller number of epsilons
    #eps_list=[0,0.05,0.1]

    nets=[]
    svms=[]
    
    #loading all the models and training the SVM
    for eps in eps_list:
        nets_list,svm_list=load_models(eps,mnist_train_sbset)
        nets.append(nets_list)
        svms.append(svm_list)

    loss_function=F.cross_entropy
    p=float('inf')
    iter=7
    device='cpu'
    #models were trained with this attack at various levels of epsilon
    def input_nn_attack(model,x,y,eps):
        return adversaries.pgd_attack_p(x,y,eps,model,loss_function,p,iter,device=device,
                                        rand_init=True)

    #same attack on the svm
    def input_svm_attack(model,x,y,eps):
        return adversaries.pgd_attack_p(x,y,eps,model,identity_loss_function,p,iter,
                                        device=device,rand_init=True)
    
    

    #create folders for adversarial examples they it dosn't already exist
    examples_parent_dir=os.path.join("adversarial_examples",
                                     "binary_sigmoid_models_adversarial_examples")
    nets_parent_dir=os.path.join(examples_parent_dir,"nets")
    svms_parent_dir=os.path.join(examples_parent_dir,"svms")
    basic_methods.exist_and_create_dir(examples_parent_dir)
    basic_methods.exist_and_create_dir(nets_parent_dir)
    basic_methods.exist_and_create_dir(svms_parent_dir)
    for j in range(10):
        net_trial_folder=os.path.join(nets_parent_dir,"trial"+str(j))
        svms_trial_folder=os.path.join(svms_parent_dir, "trial"+str(j))
        basic_methods.exist_and_create_dir(net_trial_folder)
        basic_methods.exist_and_create_dir(svms_trial_folder)
    
    #set and save the random seed
    seed= int(time.time())
    torch.manual_seed(seed)
    file=open(os.path.join(examples_parent_dir,"random_seed"),'wb')
    pickle.dump({"random_seed":seed},file)
    file.close()
    
    #generate the adversarial examples
    nets_examples=[]
    svms_examples=[]
    for i in range(len(eps_list)):
        eps=eps_list[i]
        nets_eps,svms_eps=nets[i], svms[i]
        nets_examples_eps,svms_examples_eps=[],[]
        for j in range(len(nets_eps)):
            net=nets_eps[j]
            prefix="trial"+str(j)
            examples_name=os.path.join(examples_parent_dir,"nets",prefix,
                                       prefix+"_epsilon="+str(eps)+"examples")
            examples=gen_adversarial_examples(net,input_nn_attack,eps,mnist_test_sbset, 
                                              save=examples_name)
            nets_examples_eps.append(examples)
        for j in range(len(svms_eps)):
            svm=svms_eps[j]
            prefix="trial"+str(j)
            examples_name=os.path.join(examples_parent_dir,"svms",prefix,
                                       prefix+"_epsilon="+str(eps)+"examples")
            examples=gen_adversarial_examples(svm,input_svm_attack,eps,mnist_test_sbset, 
                                              save=examples_name)
            svms_examples_eps.append(examples)
        nets_examples.append(nets_examples_eps)
        svms_examples.append(svms_examples_eps)


    #create folder for plots if it doesn't already exist
    plots_parent_dir=os.path.join("plots","binary_sigmoid_models")
    basic_methods.exist_and_create_dir("plots")
    basic_methods.exist_and_create_dir(plots_parent_dir)
    
    #make plots for each of the attacks
    white_box_attacks_plots_vs_eps(eps_list,nets,svms,nets_examples,svms_examples, save_fig_folder=plots_parent_dir)
    grey_box_attacks_plots_vs_eps(eps_list,nets,svms,nets_examples,svms_examples, save_fig_folder=plots_parent_dir)
    black_box_attacks_plots_vs_eps(eps_list,nets,svms,nets_examples,svms_examples, save_fig_folder=plots_parent_dir)

    




if __name__ == "__main__":
    basic_methods.set_cwd()
    main()
