import random
import torch

from langs import arabic_numerals_lang, roman_numerals_lang
from nn_modules import device, EOS_token, SOS_token
from seq2seq_setup import make_tensor

def evaluate(encoder, decoder, sentence, max_length=27):
    with torch.no_grad():
        input_tensor = make_tensor(arabic_numerals_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initialize_hidden_layer()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(roman_numerals_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluate_random(encoder, decoder, numeral_pairs, n=10):
    for i in range(n):
        pair = random.choice(numeral_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print(pair[1] == output_sentence.replace('<EOS>', ''))
        print('')


def evaluate_accuracy(encoder, attn_decoder, numeral_pairs):
    result = {'correct':[], 'incorrect':[]}
    for pair in numeral_pairs:
        output_words, attentions = evaluate(encoder, attn_decoder, pair[0])
        output_sentence = ''.join(output_words)
        if output_sentence.replace('<EOS>', '') == pair[1]:
            result['correct'].append({'input':pair[0], 'expected':pair[1], 'model_output':output_sentence})
        else:
            result['incorrect'].append({'input':pair[0], 'expected':pair[1], 'model_output':output_sentence})
    return result