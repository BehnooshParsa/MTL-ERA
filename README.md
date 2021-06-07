# MTL-ERA: Multi-Task Learning for Ergonomics Risk Assessment


In this repository, we provide the details of the implementation of the following manuscript: <br> <br>


### [A Multi-Task Learning Approach for Human Activity Segmentation and Ergonomics Risk Assessment](https://openaccess.thecvf.com/content/WACV2021/html/Parsa_A_Multi-Task_Learning_Approach_for_Human_Activity_Segmentation_and_Ergonomics_WACV_2021_paper.html)

Behnoosh Parsa, Ashis G. Banerjee; Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 2352-2362 <br> <br>


---

## Abstract

<div align="justify"> In this work, we propose a new approach to Human Activity Evaluation (HAE) in long videos using graph-based multi-task modeling. Previous works in activity evaluation either directly compute a metric using a detected skeleton or use the scene information to regress the activity score. These approaches are insufficient for accurate activity assessment since they only compute an average score over a clip, and do not consider the correlation between the joints and body dynamics. Moreover, they are highly scene-dependent which makes the generalizability of these methods questionable. We propose a novel multi-task framework for HAE that utilizes a Graph Convolutional Network backbone to embed the interconnections between human joints in the features. In this framework, we solve the Human Activity Segmentation (HAS) problem as an auxiliary task to improve activity assessment. The HAS head is powered by an encoder-Decoder Temporal Convolutional Network to semantically segment long videos into distinct activity classes, whereas, HAE uses a Long-Short-Term-Memory-based architecture. We evaluate our method on the UW-IOM and TUM Kitchen datasets and discuss the success and failure cases in these two datasets.. </div> <br>
<p align="center">
  <img width="500" src="https://github.com/BehnooshParsa/MTL-ERA/blob/master/figures/PipelineIdea.png">
</p> 
<div align="justify"> The details of the best performing multi-task learning network architecture is shown in the following picture. </div>
&nbsp;
<p align="center">
  <img width="400" src="https://github.com/BehnooshParsa/MTL-ERA/blob/master/figures/net.png">
</p> <br> 

## How to run the code
### Required Environment 
To install all the requirments for this project you can create a conda environment using the [MLTGCN_environment.yml](https://github.com/BehnooshParsa/MTL-ERA/blob/master/MLTGCN_environment.yml), by executing the following command in your terminal:

```console
foo@bar:~$ conda env create -f environment_MTLGCN.yml
```
### Running the experiments
Configration for experiments on UW/TUM dataset is in [config_files](https://github.com/BehnooshParsa/MTL-ERA/tree/master/config_files) folder. You can change the task by editting the experiment files, for example [config_UW_exp.yml](https://github.com/BehnooshParsa/MTL-ERA/blob/master/config_files/config_UW_exp.yml).
There are four tasks to choose from, ['classification', 'regression', 'MTL', 'MTL-Emb']. 

The setting for the data directories are in [config_UW_data.yml](https://github.com/BehnooshParsa/MTL-ERA/blob/master/config_files/config_UW_data.yml).

To run the experiments for UW dataset execute the [GCNEDTCN_UW.py](https://github.com/BehnooshParsa/MTL-ERA/blob/master/run/GCNEDTCN_UW.py).
