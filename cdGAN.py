import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse
from utils.torch_utils import *
from utils.utils import *
from fast_MLPClassifier import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np
from models import *
import warnings

logging.getLogger('transformers').setLevel(logging.ERROR)

# seed = 2023
# torch.manual_seed(seed)
# np.random.seed(seed)

class WGAN_LangGP():
    def __init__(self, batch_size=64, lr=0.0001, num_epochs=150, seq_len = 156, data_dir='./data/single_task_train.fasta', \
        run_name='test', hidden=512, d_steps = 10, max_examples=2000):
        self.preds_cutoff = 0.8
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = 1
        self.lamda = 10 #lambda
        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        self.sample_dir = './samples/' + run_name + "/"
        self.load_data(data_dir, max_examples) #max examples is size of discriminator training set
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()

    def build_model(self):
        self.G = Generator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        self.D = Discriminator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        # print(self.G)
        # print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.analyzer = AMPClassifier()

    def load_data(self, datadir, max_examples=1e6):
        self.data, self.charmap, self.inv_charmap = utils.language_helpers.load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
        self.labels = np.zeros(len(self.data)) #this marks at which epoch this data was added

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), self.checkpoint_dir + "G_weights_{}.pth".format(epoch))
        # torch.save(self.D.state_dict(), self.checkpoint_dir + "D_weights_{}.pth".format(epoch))

    def load_model(self, directory='', iteration=None):
        '''
            Load model parameters from most recent epoch
        '''
        # print("Loading model from directory:", directory)  # Add this line for debugging
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        list_D = glob.glob(directory + "D*.pth")

        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1  # file is not there
        if iteration is None:
            print("Loading most recently saved...")
            G_file = max(list_G, key=os.path.getctime)
            D_file = max(list_D, key=os.path.getctime)
        else:
            G_file = "G_weights_{}.pth".format(iteration)
            D_file = "D_weights_{}.pth".format(iteration)
        epoch_found = int((G_file.split('_')[-1]).split('.')[0])
        epoch_found = int((D_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_found, directory))
        self.G.load_state_dict(torch.load(G_file))
        self.D.load_state_dict(torch.load(D_file))
        return epoch_found
    
    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1)
        alpha = alpha.view(-1,1,1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda() if self.use_cuda else alpha
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda() if self.use_cuda else interpolates
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                                  if self.use_cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1).norm(2,dim=1) - 1) ** 2).mean() * self.lamda
        return gradient_penalty

    def translate_dna_to_protein(self, dna_sequence, translation_table):
        protein_sequence = []
        codon = ""

        for nucleotide in dna_sequence:
            codon += nucleotide

            # Check if we have a complete codon
            if len(codon) == 3:
                # Translate the codon or use 'X' if not found in the table
                amino_acid = translation_table.get(codon, 'X')
                protein_sequence.append(amino_acid)
                codon = ""

        return ''.join(protein_sequence)

    def train_model(self, load_dir):
        self.load_model(load_dir)
        losses_f = open(self.checkpoint_dir + "losses.txt",'a+')
        d_fake_losses, d_real_losses, grad_penalties = [],[],[]
        G_losses, D_losses, W_dist = [],[],[]

        one = torch.tensor(1.0) #torch.FloatTensor([1])
        one = one.cuda() if self.use_cuda else one
        one_neg = torch.tensor(-1) #one * -1

        table = np.arange(len(self.charmap)).reshape(-1, 1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)
        num_batches_sample = 5 #15
        n_batches = int(len(self.data)/self.batch_size)

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
            'TAA': '_', 'TAG': '_', 'TGA': '_',# '***': '#', # Stop codons or placeholders ('X')
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

        i = 0

        for epoch in range(1, self.n_epochs+1):   
            # if epoch % 2 == 0: 
            self.save_model(epoch)
            sampled_seqs = self.sample(num_batches_sample, epoch)
            sampled_preds = self.analyzer.predict_model(sampled_seqs)
            
            tmp = 0
            with open(self.sample_dir + "sampled_{}_preds.txt".format(epoch), 'w+') as f:
                for j, seq in enumerate(sampled_seqs):
                    modified_seq = re.sub(r'P', '', seq)
                    condition_met = (
                        len(modified_seq) >= 15
                        and len(modified_seq) % 3 == 0
                        and self.translate_dna_to_protein(modified_seq, translation_table).startswith('M')
                        and self.translate_dna_to_protein(modified_seq, translation_table).endswith('_')
                    )
                    if condition_met:
                        f.write(seq + '\t' + str(sampled_preds[j][0].item()) + '\n')
                        if sampled_preds[j][0] >= 0.8:
                            tmp = tmp + 1
                    else:
                        f.write(seq + '\t' + str(0.0) + '\n')
                        sampled_preds[j] = 0.0

            print(f"Epoch {epoch}: Number of samples above 0.8: {tmp}")

            for idx in range(n_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in self.data[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                real_data = torch.Tensor(data_one_hot)
                real_data = to_var(real_data)
                for p in self.D.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                for _ in range(self.d_steps): # Train D
                    self.D.zero_grad()
                    d_real_pred = self.D(real_data)
                    d_real_err = torch.mean(d_real_pred) #want to push d_real as high as possible
                    d_real_err.backward(one_neg)

                    z_input = to_var(torch.randn(self.batch_size, 128))
                    d_fake_data = self.G(z_input).detach()
                    d_fake_pred = self.D(d_fake_data)
                    d_fake_err = torch.mean(d_fake_pred) #want to push d_fake as low as possible
                    d_fake_err.backward(one)

                    gradient_penalty = self.calc_gradient_penalty(real_data.data, d_fake_data.data)
                    gradient_penalty.backward()

                    d_err = d_fake_err - d_real_err + gradient_penalty
                    self.D_optimizer.step()

                # Append things for logging
                d_fake_np, d_real_np, gp_np = (d_fake_err.data).cpu().numpy(), \
                        (d_real_err.data).cpu().numpy(), (gradient_penalty.data).cpu().numpy()
                grad_penalties.append(gp_np)
                d_real_losses.append(d_real_np)
                d_fake_losses.append(d_fake_np)
                D_losses.append(d_fake_np - d_real_np + gp_np)
                W_dist.append(d_real_np - d_fake_np)
                # Train G
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                z_input = to_var(torch.randn(self.batch_size, 128))
                g_fake_data = self.G(z_input)
                dg_fake_pred = self.D(g_fake_data)

                seqs = (g_fake_data.data).cpu().numpy()
                decoded_seqs = [decode_one_seq(seq, self.inv_charmap) for seq in seqs]

                preds  = self.analyzer.predict_model(decoded_seqs)

                for j, seq in enumerate(decoded_seqs):
                    modified_seq = re.sub(r'P', '', seq)
                    condition_met = (
                        len(modified_seq) >= 15
                        and len(modified_seq) % 3 == 0
                        and self.translate_dna_to_protein(modified_seq, translation_table).startswith('M')
                        and self.translate_dna_to_protein(modified_seq, translation_table).endswith('_')
                    )
                    if condition_met:
                        tt = 0
                    else:
                        preds[j][0] = 0.0
                
                g_err = -torch.mean(dg_fake_pred)               
                cl_err = 0.1 * torch.mean(torch.log(preds + 1e-6))
                g_err = g_err - cl_err
                g_err.backward(retain_graph=True)

                self.G_optimizer.step()
          
    def sample(self, num_batches_sample, epoch):
        decoded_seqs = []
        for i in range(num_batches_sample):
            z = to_var(torch.randn(self.batch_size, 128))
            self.G.eval()
            torch_seqs = self.G(z)
            seqs = (torch_seqs.data).cpu().numpy()
            decoded_seqs += [decode_one_seq(seq, self.inv_charmap) for seq in seqs]
        self.G.train()
        return decoded_seqs

def main():
    parser = argparse.ArgumentParser(description='FBGAN with AMP analyzer.')
    parser.add_argument("--run_name", default= "cdgan", help="Name for output files")
    parser.add_argument("--load_dir", default="./checkpoint/realProt_50aa/", help="Load pretrained GAN checkpoints")

    args = parser.parse_args()
    model = WGAN_LangGP(run_name=args.run_name)
    model.train_model(args.load_dir)

if __name__ == '__main__':
    main()
