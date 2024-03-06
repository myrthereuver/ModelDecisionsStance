# Stance Modelling

This is the repository for the paper "Investigating the Robustness of Modelling Decisions for Few-Shot Cross-Topic Stance Detection: A Preregistered Study" (Published at LREC-COLING 2024).

[URL to the paper]()
[URL to the preregistration of the modelling decisions](https://osf.io/v9snk?view_only=abbd99d27b9a471fb2e4df8ffdafd2a1)

## Running the Experiments

### Preprocessing the benchmark 
In order to preprocess the benchmark into a Same Side Stance benchmark, we used the following script, in the "preprocessing" directory:
* stancebenchmark_functions.py is used to access and store the stance benchmark datasets;
* TrainingData_StanceBenchmark_intoSameSideStance.ipynb imports the functions from the .py file and preprocesses the datasets into Same Side Stance.

### bi-encoding models
The directory "bi-encoding" shows the code for our bi-encoding models.

Performing the same experiments as in our paper requires executing the notebook script "SETFIT_SSSC_experiment.ipynb" or "SETFIT_ProCon_experiment.ipynb" (depending on the task definition) with the specified datasets.

Running these models requires => python3.7 in a notebook, with the following packages and dependencies installed:

setfit
in addition to:
keras==2.9.0
Keras-Preprocessing==1.1.2
pandas==1.3.5
scikit-learn==1.0.2
scipy==1.7.0
sentence-transformers==2.2.0
sentencepiece==0.1.92
sib-clustering==0.2.0
six==1.16.0
sklearn==0.0
tensorboard==2.9.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.9.1
tensorflow-estimator==2.9.0
tensorflow-io-gcs-filesystem==0.26.0
threadpoolctl==3.1.0
tokenizers==0.12.1
torch==1.11.0


### cross-encoding models
The directory "cross-encoding" contains the code for our cross-encoding models.

Performing the same experiments as in our paper requires executing the following shell script in bash on a GPU: "roberta_seeds.sh"
which calls the python script "experiments_RQ1_multipleExperiments.py" with 5 different random seeds, 2 different versions of the benchmark (same side stance or Pro/Con), and a specified output directory for the trained model and the evaluation outcomes. 

Running these models requires => python3.7 with the following packages and dependencies installed:

keras==2.9.0
Keras-Preprocessing==1.1.2
pandas==1.3.5
scikit-learn==1.0.2
scipy==1.7.0
sentence-transformers==2.2.0
sentencepiece==0.1.92
sib-clustering==0.2.0
six==1.16.0
sklearn==0.0
tensorboard==2.9.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.9.1
tensorflow-estimator==2.9.0
tensorflow-io-gcs-filesystem==0.26.0
threadpoolctl==3.1.0
tokenizers==0.12.1
torch==1.11.0


