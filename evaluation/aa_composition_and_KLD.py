import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

data = ["real.fasta", "random.fast", "gen.fasta"]

real = [] # list of indices selected at random
random = [] # list of indices selected at random
top_gen_data = [] # list of indices of the top scored generated data ( P(AMP>=0.8) )

# Define indices for each FASTA file 
indices = {
    "real_data": real,
    "random_protein_sequences": random,
    "generated_data": top_gen_data,
}

# Define labels for your cases
labels = [
    'Real data', 
    'Random data', 
    'Generated data']

# Create a mapping dictionary for file names to labels
file_to_label = {
    "real_data": "Real data",
    "random_protein_sequences": "Random data",
    "Generated data": "Generated data"
}

custom_colors = {
    "Real data": "#16a085",      # Vibrant Teal
    "Random data": "#2c3e50",    # Bright Sky Blue
    "Generated data": "#d35400",         # Vivid Orange
}

# Define the important amino acids
important_aas = {'C', 'D', 'E', 'K', 'R'}

# Initialize a DataFrame to hold amino acid composition data
aa_composition_data = []

# Process each FASTA file and compute amino acid compositions
for fasta_file in data:
    # Determine the key for the index lookup
    key = fasta_file.split('.')[0]
    
    # Load the sequences from the FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    
    # Extract the specified sequences based on the indices
    for idx in indices[key]:
        if idx < len(sequences):
            seq = str(sequences[idx].seq)
            
            # Count the frequency of each amino acid
            aa_count = {aa: seq.count(aa) for aa in set(seq)}
            
            # Calculate total length to get frequencies
            total_length = len(seq)
            
            # Store amino acid composition data as frequency
            for aa, count in aa_count.items():
                aa_composition_data.append({
                    "File": key,
                    "Amino Acid": aa,
                    "Frequency": count / total_length  # Convert to frequency
                })

# Convert the amino acid composition data to a DataFrame
aa_composition_df = pd.DataFrame(aa_composition_data)

# Map the file names to the corresponding labels
aa_composition_df['Label'] = aa_composition_df['File'].map(file_to_label)

# Group by Label and Amino Acid to get average frequency per model
aa_avg_composition_df = aa_composition_df.groupby(['Label', 'Amino Acid']).agg({'Frequency': 'mean'}).reset_index()

# Ensure the important amino acids list is defined
important_aas = ['C', 'D', 'E', 'K', 'R']

# Ensure that the 'Highlight' column is created based on important amino acids
aa_avg_composition_df['Highlight'] = aa_avg_composition_df['Amino Acid'].apply(lambda x: 'Important' if x in important_aas else 'Other')

# Create a color palette for highlighting important amino acids
palette = {"Important": "red", "Other": "blue"}

# Create the list of labels (models)
labels = [
    'Real data', 
    'Random data', 
    # 'wGAN', 
    'cGAN', 
    'ACGAN', 
    'cdGAN', 
    'FBGAN-ESM2', 
    'AMPGAN', 
    'HydrAMP', 
    'RLGen'
]

# Set up the subplot grid based on the number of models
n_rows = 3 # You can adjust this based on how you want to arrange the plots
n_cols = 3
fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12))  # Adjusted the figure size

# Flatten the axes array for easier indexing
axs = axs.flatten()

# Loop through each model label and plot
for i, label in enumerate(labels):
    # Subset the data for the current label (model)
    model_data = aa_avg_composition_df[aa_avg_composition_df['Label'] == label]
    
    # Plot the amino acid composition in the corresponding subplot
    sns.barplot(x='Amino Acid', y='Frequency', hue='Highlight', data=model_data, palette=palette, ax=axs[i], dodge=False)
    
    # Customize the plot
    axs[i].set_title(f"{label}", fontsize=20, fontweight='bold')
    axs[i].set_ylabel("Frequency", fontsize=18)
    axs[i].set_xlabel("Amino Acid", fontsize=18)
    axs[i].set_ylim(0, 0.25)
    # Remove the legend
    axs[i].get_legend().remove()

# Remove any extra subplots (if there are empty ones)
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Adjust layout to prevent cutting off plots
plt.ylim(0, 0.25)
plt.tight_layout()
plt.ylim(0, 0.25)
plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Increased the spacing between plots
plt.ylim(0, 0.25)
# Show the plot
plt.show()



########################################################################################

import numpy as np
from scipy.special import kl_div
from Bio import SeqIO

# Function to compute amino acid frequency distribution
def get_aa_frequencies(sequences):
    aa_counts = {}
    total_length = 0
    for seq in sequences:
        for aa in seq:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
            total_length += 1
    # Normalize to get frequencies
    for aa in aa_counts:
        aa_counts[aa] /= total_length
    return aa_counts

# Function to compute KLD between two frequency distributions
def compute_kld(dist1, dist2):
    # Align the two distributions on the same amino acids (set union)
    all_aa = set(dist1.keys()).union(dist2.keys())
    
    # Ensure both distributions have values for every amino acid
    dist1 = {aa: dist1.get(aa, 0) for aa in all_aa}
    dist2 = {aa: dist2.get(aa, 0) for aa in all_aa}
    
    # Compute KLD for each amino acid
    kld_value = 0
    for aa in all_aa:
        if dist1[aa] > 0 and dist2[aa] > 0:
            kld_value += dist1[aa] * np.log(dist1[aa] / dist2[aa])
    return kld_value
# Function to extract sequences from a FASTA file and calculate the amino acid frequency
def get_aa_frequencies_from_fasta(fasta_file, indices):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    selected_sequences = [str(sequences[idx].seq) for idx in indices if idx < len(sequences)]
    return get_aa_frequencies(selected_sequences)

# Step 1: Compute amino acid frequencies for the real data (DBAASP_filtered)
real_sequences = list(SeqIO.parse("DBAASP_filtered.fasta", "fasta"))
real_aa_frequencies = get_aa_frequencies([str(seq.seq) for seq in real_sequences])

# Step 2: Calculate the KLD between the real data and each generated model
kld_results = {}

# Loop through each model and compute the KLD
for fasta_file in data:
    model_label = file_to_label.get(fasta_file.split('.')[0], "Unknown")
    
    # Skip real data since we are comparing models with real data
    if model_label == "Real data":
        continue
    
    # Get the amino acid frequencies for the model
    model_aa_frequencies = get_aa_frequencies_from_fasta(fasta_file, indices[fasta_file.split('.')[0]])

    # Compute the KLD between the real data and this model
    kld_value = compute_kld(real_aa_frequencies, model_aa_frequencies)

    # Store the result with the model label
    kld_results[model_label] = kld_value

# Print the KLD results for each model
print("Kullback-Leibler Divergence (KLD) for each model:")
for label, kld_value in kld_results.items():
    print(f"{label}: {kld_value}")