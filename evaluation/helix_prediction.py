import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import warnings

logging.getLogger('transformers').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)

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

    def get_protein_mean_embeddings(self, protein_sequences):
        logging.getLogger('transformers').setLevel(logging.ERROR)

        model_name="facebook/esm2_t12_35M_UR50D"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)

        tokens = tokenizer(protein_sequences, return_tensors="pt", padding=True, clean_up_tokenization_spaces=True).to(self.device)

        with torch.no_grad():
            outputs = model(**tokens)

        embeddings = outputs.last_hidden_state
        mean_embeddings = torch.mean(embeddings, dim=1)
        # torch.cuda.empty_cache()
        return mean_embeddings
   
    def predict_model(self, prot_seqs):
        emb = self.get_protein_mean_embeddings(prot_seqs).to(self.device)
        input_size = len(emb[0])

        self.model = mltMLP(input_size, self.hidden_dim, 2, 2)
        self.model.to(self.device)

        self.model.load_state_dict(torch.load('./best_model_2_tasks.pth'))
        
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

gen_sequences = []
with open("generated_sequences.fasta", 'r') as file: # Load generated data
    sequence = ''
    for line in file:
        line = line.strip()
        if line.startswith('>'):
            if sequence:
                gen_sequences.append(sequence)
                sequence = ''
        else:
            sequence += line
    if sequence:  
        gen_sequences.append(sequence)

analyzer = AMPClassifier()
sampled_preds = analyzer.predict_model(gen_sequences)

preds2 = sampled_preds['1']

# Check and print sequences with accuracy >= 0.95
threshold = 0.95
high_accuracy_sequences = []

for i, score in enumerate(preds2):
    if score >= threshold:
        high_accuracy_sequences.append((gen_sequences[i], score))#(specific_indices[i], specific_sequences[i], score))

# Print the result
print(f"Number of sequences with accuracy >= {threshold}: {len(high_accuracy_sequences)}")

# for idx, seq, score in high_accuracy_sequences:
    # print(f"Sequence Index: {idx}, Score: {score.item()}\nSequence: {seq}\n")

for idx, (seq, score) in enumerate(high_accuracy_sequences):
    
    print(f">Sequence {idx}\n {seq}")
