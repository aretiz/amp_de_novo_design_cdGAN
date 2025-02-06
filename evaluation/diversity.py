# Function to calculate diversity score
def diversity_score(sequences):
    unique_sequences = set(sequences)
    diversity = len(unique_sequences) / len(sequences)
    return diversity

gen_sequences = []
with open("generated_samples.fasta", 'r') as file: # Load generated data
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

diversity = diversity_score(gen_sequences)
print(f'Diversity Score: {diversity:.2%}')
