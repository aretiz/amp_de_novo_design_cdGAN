import torch, os, glob
from transformers import AutoModel, AutoTokenizer
# from utils.bio_utils import *
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging
import re
import torch.optim as optim
import warnings

logging.getLogger('transformers').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)

translation_table = {
    'TTT': 'F', 'TTC': 'F',  # Phenylalanine (F)
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',  # Leucine (L)
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',  # Isoleucine (I)
    'ATG': 'M',  # Methionine (M) - Start codon
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',  # Valine (V)
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',  # Serine (S)
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',  # Proline (P)
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',  # Threonine (T)
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',  # Alanine (A)
    'TAT': 'Y', 'TAC': 'Y',  # Tyrosine (Y)
    'TAA': '', 'TAG': '', 'TGA': '', # '***': '#', # Stop codons or placeholders ('X')
    'CAT': 'H', 'CAC': 'H',  # Histidine (H)
    'CAA': 'Q', 'CAG': 'Q',  # Glutamine (Q)
    'AAT': 'N', 'AAC': 'N',  # Asparagine (N)
    'AAA': 'K', 'AAG': 'K',  # Lysine (K)
    'GAT': 'D', 'GAC': 'D',  # Aspartic Acid (D)
    'GAA': 'E', 'GAG': 'E',  # Glutamic Acid (E)
    'TGT': 'C', 'TGC': 'C',  # Cysteine (C)
    'TGG': 'W',  # Tryptophan (W)
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',  # Arginine (R)
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',  # Glycine (G)
}

class mltMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_task1, num_classes_task2, drop_out=0.2):
        super(mltMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

        # Task 1 output layer
        self.fc_task1 = nn.Linear(hidden_size, num_classes_task1)
        # Task 2 output layer
        self.fc_task2 = nn.Linear(hidden_size, num_classes_task2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        # Task 1
        out_task1 = self.fc_task1(out)
        # Task 2
        out_task2 = self.fc_task2(out)

        return out_task1, out_task2

class AMPClassifier:
    def __init__(self, hidden_dim=128, epochs=30, batch_size=16):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_epochs = epochs
        # self.model_name = "facebook/esm2_t12_35M_UR50D"

    def get_protein_mean_embeddings(self, protein_sequences):
        logging.getLogger('transformers').setLevel(logging.ERROR)

        model_name="facebook/esm2_t12_35M_UR50D"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        # model.eval()

        tokens = tokenizer(protein_sequences, return_tensors="pt", padding=True, clean_up_tokenization_spaces=True).to(self.device)

        with torch.no_grad():
            outputs = model(**tokens)

        embeddings = outputs.last_hidden_state
        mean_embeddings = torch.mean(embeddings, dim=1)
        # torch.cuda.empty_cache()
        return mean_embeddings

    def translate_dna_to_protein(self, dna_sequence, translation_table):
        protein_sequence = []
        codon = ""

        for nucleotide in dna_sequence:
            codon += nucleotide

            if len(codon) == 3:
                amino_acid = translation_table.get(codon, 'X')
                protein_sequence.append(amino_acid)
                codon = ""

        return ''.join(protein_sequence)

    def train_model(self, dna_seqs, targets_task1, targets_task2):
        prot_seqs = []
        for i in range(len(dna_seqs)):
            seq = re.sub(r'P*$', '', dna_seqs[i])
            prot = self.translate_dna_to_protein(seq, translation_table)
            prot_seqs.append(prot)

        emb = self.get_protein_mean_embeddings(prot_seqs)
        input_size = len(emb[0])

        # Initialize classifier for two tasks
        self.model = mltMLP(input_size, self.hidden_dim)
        self.model.to(self.device)

        # Two targets, one for each task
        loader = DataLoader(list(zip(emb, targets_task1, targets_task2)), batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        scaler = torch.amp.GradScaler('cuda')

        self.model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            for x, y_task1, y_task2 in loader:
                x, y_task1, y_task2 = x.to(self.device), y_task1.to(self.device), y_task2.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs_task1, outputs_task2 = self.model(x)
                    loss_task1 = criterion(outputs_task1, y_task1)
                    loss_task2 = criterion(outputs_task2, y_task2)
                    # Sum the losses from both tasks
                    loss = 0.5 * loss_task1 + 0.5 * loss_task2

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    def predict_model(self, dna_seqs):
        prot_seqs = []
        for i in range(len(dna_seqs)):
            seq = re.sub(r'P*$', '', dna_seqs[i])
            prot = self.translate_dna_to_protein(seq, translation_table)
            prot_seqs.append(prot)

        emb = self.get_protein_mean_embeddings(prot_seqs).to(self.device)
        input_size = len(emb[0])

        self.model = mltMLP(input_size, self.hidden_dim, 2, 2)
        self.model.to(self.device)

        self.model.load_state_dict(torch.load('./best_model_2_tasks.pth', weights_only=True))

        loader = DataLoader(emb, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions_task1 = []
        predictions_task2 = []

        for x in loader:
            x = x.to(self.device)
            outputs_task1, outputs_task2 = self.model(x)
            probabilities_task1 = torch.softmax(outputs_task1, dim=1)
            probabilities_task2 = torch.softmax(outputs_task2, dim=1)
            positive_class_probabilities_task1 = probabilities_task1[:, 1]
            positive_class_probabilities_task2 = probabilities_task2[:, 1]
            predictions_task1.append(positive_class_probabilities_task1)
            predictions_task2.append(positive_class_probabilities_task2)

        predictions_task1_tensor = torch.cat(predictions_task1).reshape(-1, 1)
        predictions_task2_tensor = torch.cat(predictions_task2).reshape(-1, 1)
        return {
            '0': predictions_task1_tensor,
            '1': predictions_task2_tensor
        }

