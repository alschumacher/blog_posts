from seq2seq_setup import train_numeral_seq2seq
from model_evalution import evaluate_random, evaluate_accuracy, evaluate
from langs import numeral_pairs

training_config = {
    'dropout_p':0.2,
    'hidden_units':256,
    'learning_rate':0.005,
    'teacher_forcing_ratio':0.66,
    'n_iters':300000
}

encoder, decoder = train_numeral_seq2seq(**training_config)

results = evaluate_accuracy(encoder, decoder, numeral_pairs)

import torch
torch.save(encoder.state_dict(), 'encoder.torch')
torch.save(decoder.state_dict(), 'decoder.torch')
