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

# Hydrophobic amino acids set
hydrophobic_aa = {'A', 'I', 'L', 'M', 'F', 'V', 'P', 'W'}

# Initialize a DataFrame to hold properties
properties_data = []

# Process each FASTA file and compute properties
for fasta_file in data:
    # Determine the key for the index lookup
    key = fasta_file.split('.')[0]
    
    # Load the sequences from the FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    
    # Extract the specified sequences based on the indices
    for idx in indices[key]:
        if idx < len(sequences):
            seq = str(sequences[idx].seq)
            analyzer = ProteinAnalysis(seq)
            
            # Calculate properties
            charge = analyzer.charge_at_pH(7.4)  # Charge at pH 7.4
            pI = analyzer.isoelectric_point()  # Isoelectric point
            aromaticity = analyzer.aromaticity()  # Aromaticity
            hydrophobic_count = sum([seq.count(aa) for aa in hydrophobic_aa])  # Hydrophobic amino acids
            hydrophobicity_ratio = hydrophobic_count / len(seq)  # Hydrophobicity ratio
            
            # Append results to the list
            properties_data.append({
                "File": key, 
                "Charge": charge, 
                "pI": pI, 
                "Aromaticity": aromaticity, 
                "Hydrophobicity": hydrophobicity_ratio
            })

# Convert to DataFrame
properties_df = pd.DataFrame(properties_data)

# Map file names to labels
properties_df['Label'] = properties_df['File'].map(file_to_label)

# Create a color palette
palette = sns.color_palette("husl", len(labels))

palette = [custom_colors[label] for label in labels]

### 1. Charge ###
plt.figure(figsize=(16, 9))
sns.boxplot(x='Label', y="Charge", data=properties_df, palette=palette, order=labels)
plt.title('Charge', fontsize=35, fontweight='bold')
plt.xlabel("", fontsize=30, fontweight='bold')
plt.ylabel("Charge", fontsize=28, fontweight='bold')
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.tight_layout()
plt.show()

# Plot 2: Isoelectric Point (pI)
plt.figure(figsize=(16, 9))
sns.boxplot(x='Label', y='pI', data=properties_df, palette=palette, order=labels)
plt.title('Isoelectric Point (pI)', fontsize=35, fontweight='bold')
plt.xlabel("", fontsize=30, fontweight='bold')
plt.ylabel("pI", fontsize=28, fontweight='bold')
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.tight_layout()
plt.show()

# Plot 3: Aromaticity
plt.figure(figsize=(16, 9))
sns.boxplot(x='Label', y='Aromaticity', data=properties_df, palette=palette, order=labels)
plt.title('Aromaticity', fontsize=35, fontweight='bold')
plt.xlabel("", fontsize=30, fontweight='bold')
plt.ylabel("Aromaticity", fontsize=28, fontweight='bold')
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.tight_layout()
plt.show()

# Plot 4: Hydrophobicity Ratio
plt.figure(figsize=(16, 9))
sns.boxplot(x='Label', y='Hydrophobicity', data=properties_df, palette=palette, order=labels)
plt.title('Hydrophobicity Ratio', fontsize=35, fontweight='bold')
plt.xlabel("", fontsize=30, fontweight='bold')
plt.ylabel("Hydrophobicity", fontsize=28, fontweight='bold')
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.tight_layout()
plt.show()
plot_features = {
    'Charge': 'Charge',
    'Isoelectric Point (pI)': 'pI',
    'Aromaticity': 'Aromaticity',
    'Hydrophobicity Ratio': 'Hydrophobicity'
}





