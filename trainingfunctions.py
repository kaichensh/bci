import numpy as np
import os
from glob import glob

def train_test_split(input_output_df, n_parts, test_part, mode):
    n_motifs = len(input_output_df.motif.value_counts())
    n_steps = len(input_output_df.step.value_counts())
    if mode == 'motif_wise':
        motifs_per_part = np.ceil(n_motifs/n_parts).astype(int)
        test_motifs = np.arange(motifs_per_part*test_part, motifs_per_part*(test_part+1))
        test_input = np.array(input_output_df[input_output_df.motif.isin(test_motifs)].input.values.tolist())
        test_output = np.array(input_output_df[input_output_df.motif.isin(test_motifs)].output.values.tolist())
        train_input = np.array(input_output_df[~input_output_df.motif.isin(test_motifs)].input.values.tolist())
        train_output = np.array(input_output_df[~input_output_df.motif.isin(test_motifs)].output.values.tolist())
    elif mode == 'part_wise':
        steps_per_part = np.ceil(n_steps/n_parts).astype(int)
        test_steps = np.arange(steps_per_part*test_part, steps_per_part*(test_part+1))
        test_input = np.array(input_output_df[input_output_df.step.isin(test_steps)].input.values.tolist())
        test_output = np.array(input_output_df[input_output_df.step.isin(test_steps)].output.values.tolist())
        train_input = np.array(input_output_df[~input_output_df.step.isin(test_steps)].input.values.tolist())
        train_output = np.array(input_output_df[~input_output_df.step.isin(test_steps)].output.values.tolist())
    return train_input, train_output, test_input, test_output