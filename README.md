# Classifier-driven Generative Adversarial Networks for Enhanced Antimicrobial Peptide Design

Thus study proposes a novel classifier-driven GAN (cdGAN) framework that integrates classifier predictions directly into the loss function of the generative model enabling an adaptive, end-to-end learning process that enhances AMP generation without the need for explicit data modification. Additionally, the flexible design of cdGAN allows for extension to multiple peptide attributes. To demonstrate this capability, we introduce a multi-task classifier based on the Evolutionary Scale Modeling 2 (ESM2) model, enabling cdGAN to evaluate both antimicrobial activity and peptide structural properties simultaneously. This enhancement increases the likelihood of generating viable therapeutic candidates with improved antimicrobial activity and reduced toxicity. 

## Install the dependencies
The code is tested under Windows running python 3.11.9. All required packages are enclosed in `requirements.txt`. Run:
```bash
pip install -r requirements.txt
```
## Peptide generation
To run this project, follow these steps:

### Train the classifiers and save the best model
To do so, run:  
- Single-task classifier: `train_MLP_classifier.py`
- Multi-task classifier: `train_MLP_MLT_classifier.py`
  
To run `train_MLP_classifier.py` first download `mean_embeddings_esm2_t12.csv` from [Google Drive](https://drive.google.com/drive/u/2/folders/1WijbpvpEIuInb6mI43twwP2CYRpvlxK0). The expected output is the best model saved in a `.pth` format.

### Train the generative models
For each model run the following:
- Single-task cdGAN: `cdGAN.py`
- Multi-task cdGAN: `cdGAN_mlt.py`
  
The expected output is a folder with checkpoints for each model. The optimal checkpoints utilized in this work for each model are provided on [Google Drive](https://drive.google.com/drive/u/2/folders/1WijbpvpEIuInb6mI43twwP2CYRpvlxK0).

### Generate and select valid peptides
First, select the optimal model from the previous checkpoints based on the loss plots. Save them in a folder named `checkpoint_MODEL` where MODEL = {cdGAN, cdGAN_mlt} and run:
- Single-task cdGAN: `generate_samples_cdGAN.py`
- Multi-task cdGAN: `generate_samples_cdGAN_mlt.py`

The expected output is a `.txt` file for each model with all the generated sequences. Then, for each output run `select_valid_peptides.py` to create a `.fasta` file that contains validly generated peptides.

### Evaluate the models
Use the code provided in the folder `evaluation`. The codes require the `.fasta` files created in the previous step.
- To predict the antimicrobial potency of the generated peptides, toxicity, hemolytic potency, and 3D-structure use [CAMPR4 server](https://camp.bicnirrh.res.in/predict/), [Toxinpred3.0](https://webs.iiitd.edu.in/raghava/toxinpred3/), [HemoPI](https://webs.iiitd.edu.in/raghava/hemopi/), and [Alphafold3](https://alphafoldserver.com/welcome).  
  
