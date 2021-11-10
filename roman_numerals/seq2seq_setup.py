import random

import torch
import torch.nn as nn
from torch import optim

from langs import arabic_numerals_lang, roman_numerals_lang, numeral_pairs
from nn_modules import Encoder, AttnDecoder, device, EOS_token, SOS_token


def train_numeral_seq2seq(dropout_p, hidden_units, learning_rate, teacher_forcing_ratio, n_iters):

    encoder = Encoder(arabic_numerals_lang.n_words, hidden_units).to(device)
    attn_decoder = AttnDecoder(hidden_units, roman_numerals_lang.n_words, dropout_p=dropout_p).to(device)

    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=learning_rate)

    training_pairs = make_training_pairs(n_iters)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor, target_tensor = training_pair[0], training_pair[1]

        loss = train_once(
            input_tensor,
            target_tensor,
            encoder,
            attn_decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
            teacher_forcing_ratio
        )

        print_loss_total += loss
        plot_loss_total += loss

        if iter % 5000 == 0:
            print_loss_avg = print_loss_total / 5000
            print_loss_total = 0
            print(iter, print_loss_avg)

    return encoder, attn_decoder


def make_tensor(lang, numeral):
    tensor = torch.tensor(
        [lang.word2index[word] for word in numeral] + [EOS_token],
        dtype=torch.long,
        device=device).view(-1, 1)
    return tensor


def make_training_pairs(n_iters):

    training_pairs = []
    for i in range(n_iters):
        selected_pair = random.choice(numeral_pairs)
        arabic_numeral_tensor = make_tensor(arabic_numerals_lang, selected_pair[0])
        roman_numeral_tensor = make_tensor(roman_numerals_lang, selected_pair[1])
        training_pairs.append(
            (arabic_numeral_tensor, roman_numeral_tensor)
        )
    return training_pairs


def train_once(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, max_length=27):

    encoder_hidden = encoder.initialize_hidden_layer()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length







