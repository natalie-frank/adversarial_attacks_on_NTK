# Description of Repository



Descriptions of files included in this repository:

In the "Limitations" section of our paper TODO HYPERLINK, we compare the robustness of neural nets and SVMs trained using the NTK feature mapping. Our experiments show that neural nets are more robust than the associated SVMs. 



<p float="left">
  <img src="plots_in_pNTK_paper/white_box.png" width="200" />
  <img src="plots_in_pNTK_paper\grey_box.png" width="200" /> 
  <img src="plots_in_pNTK_paper\black_box.png" width="200" />
</p>




We test our models against two white-box attacks and a black box attack. (Left) we test neural nets and SVMs directly attacking the models. (Center) Next, we evaluate the SVM on attacks generated from the associated neural net and the neural net on attacks generated from the associated SVM. (Right) For the black box attacks, we test: 1) neural nets on adversarial examples generated from independently trained neural nets, 2) SVMs on adversarial examples from SVMs trained with an NTK from an independently trained neural net, 3) Neural nets on adversarial examples from SVMs trained with an NTK from and independently trained neural net, 4) SVMs on adversarial examples from independently trained neural nets.  




# Setup



1. To reconstruct the environment, run
    ```
      conda install python_enironment.txt 
    ```
   in an anaconda terminal
2. Run
   ```
      git clone https://github.com/natalie-frank/adversarial_attacks_on_NTK.git
    ```
3. The file ``basic_methods.py`` has a method called  ``set_cwd``, which sets the current working directory. Re-write this method to set your current working directory to the root folder of this repository.

# How to Run this Experiment
## To run this experiment using the nets we trained:
1. Open a terminal and navigate to the directory ``adversarial_attacks_on_NTK``
2. Run 
   ```
      python3 nn_vs_svm_attack_plotting.py
   ```
   Images should appear in the directory ``plots/binary_sigmoid_models/``. Pickle files containing the means and standard deviation for each plot will also appear here.

## To train your own neural nets:
1. Open a terminal and navigate to the directory ``adversarial_attacks_on_NTK``
2. Run 
   ```
   TODO
   ```
   For trial ``i`` models and model metadata will appear in the folder ``models/binary_sigmoid_models/'trial{i}'``


# Citations
If you use this code in your work, please cite using the following BibTeX entry:

```
@inproceedings{EngelWangFranketal2023,
  title = {Robust Explanations for Deep Neural Networks via Pseudo Neural Tangent Kernel Surrogate Models},
  author = {Andrew Engel and Zhichao Wang and Natalie S. Frank and Ioana Dumitriu and Sutanay Choudhury and Anand Sarwate and Tony Chiang},
  booktitle = {Arxiv},
  year = {2023}
}
```
