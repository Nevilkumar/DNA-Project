import os
from django.contrib.staticfiles.storage import staticfiles_storage
from pathlib import Path

def convert_to_binary(num):
    bnr = bin(num). replace('0b', '')
    x = bnr[::-1]
    while len(x) < 8:
        x += '0'
        bnr = x[::-1]

    return bnr


def convert_to_decimal(num):
    return int(num, 2)


def encoding(bin_string):
    prev0, prev1 = 'A', 'G'
    encoded_string = ''
    for ch in bin_string:
        if ch == '0':
            if prev0 == 'A':
                encoded_string += 'A'
                prev0 = 'C'
            else:
                encoded_string += 'C'
                prev0 = 'A'
        else:
            if prev1 == 'G':
                encoded_string += 'G'
                prev1 = 'T'
            else:
                encoded_string += 'T'
                prev1 = 'G'
    with open(staticfiles_storage.url('output.txt').strip("/"), 'w+') as f:
        f.write(encoded_string)


def decoding(acgt_string):
    decoded_string = ''
    for ch in acgt_string:
        if ch == 'A' or ch == 'C':
            decoded_string += '0'
        else:
            decoded_string += '1'

    return decoded_string


def input_file():
    with open(staticfiles_storage.url('input.txt').strip("/")) as f:
        content = f.read()
        print(content)
        ascii_values = [ord(ch) for ch in content]
        binary_values = [convert_to_binary(x) for x in ascii_values]
        tuple_input = tuple(''.join(binary_values))

    return tuple_input


def decoded_file(decoded_output):
    temp = [decoded_output[i:i+8] for i in range(0, len(decoded_output), 8)]
    # print(temp)
    binary_values = [''.join([str(x) for x in val]) for val in temp]
    # print(binary_values)
    decoded_text = ''.join([chr(convert_to_decimal(val))
                           for val in binary_values])
    # print(decoded_text)
    with open(staticfiles_storage.url('output.txt').strip("/"), 'w+') as f:
        f.write(decoded_text)


def v_xor(bit0, bit1):
    if(bit0 == bit1):
        return "0"
    else:
        return "1"


def v_xor(bit0, bit1):
    if(bit0 == bit1):
        return "0"
    else:
        return "1"


def viterbi_encoder():
    inputs = input_file()
    # shift register encoder
    s_reg = ["0", "0", "0"]
    obs = []
    for t in range(0, len(inputs)):
        # shifting the bits to right
        s_reg[2] = s_reg[1]
        s_reg[1] = s_reg[0]
        # inserting input
        s_reg[0] = inputs[t]
        state = s_reg[0] + s_reg[1]
        obs.append([])
        # encoded bits
        obs[t] = v_xor(v_xor(s_reg[0], s_reg[1]), s_reg[2]) +\
            v_xor(s_reg[0], s_reg[2])
        # print(s_reg,state)
    last, second_last = int(inputs[-1]), int(inputs[-2])
    obs.append(str(last ^ second_last)+str(second_last))
    obs.append(str(last)+str(last))

    DNA_sequence = ''.join(obs)
    encoding(DNA_sequence)


# inputs = ('1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','0')
# encoded_data = viterbi_encoder(inputs)
# with open('encoded_file.txt','w+') as f:
# f.write(encoded_data)
# print(encoded_data)
# obs = encoded_data
# print(encoded_data)
# obs = ("11","10","11","11","01","01","11")


start_metric = {'zero': 0, 'one': 0, 'two': 0, 'three': 0}
state_machine = {
    # current state, possible branches, branch information
    'zero': {'b1': {'out_b': "11", 'prev_st': 'one', 'input_b': 0},
             'b2': {'out_b': "00", 'prev_st': 'zero', 'input_b': 0}},
    'one': {'b1': {'out_b': "01", 'prev_st': 'three', 'input_b': 0},
            'b2': {'out_b': "10", 'prev_st': 'two', 'input_b': 0}},
    'two': {'b1': {'out_b': "11", 'prev_st': 'zero', 'input_b': 1},
            'b2': {'out_b': "00", 'prev_st': 'one', 'input_b': 1}},
    'three': {'b1': {'out_b': "10", 'prev_st': 'three', 'input_b': 1},
              'b2': {'out_b': "01", 'prev_st': 'two', 'input_b': 1}},

}


def bits_diff_num(num_1, num_2):
    count = 0
    for i in range(0, len(num_1), 1):
        if num_1[i] != num_2[i]:
            count = count+1
    return count


def viterbi():
    obs = ""
    with open(staticfiles_storage.url('input.txt').strip("/")) as f:
        obs = f.readlines()
    # Trellis structure
    sequence = ''.join(obs)
    obs = decoding(sequence)
    obs = [obs[i:i+2] for i in range(0, len(obs), 2)]

    # print(len(obs))

    V = [{}]
    for st in state_machine:
        # Calculating the probability of both initial possibilities for the first observation
        V[0][st] = {"metric": start_metric[st]}
    # for t&amp;amp;amp;amp;amp;amp;amp;amp;gt;0
    for t in range(1, len(obs)+1):
        V.append({})
        for st in state_machine:
            # Check for smallest bit difference from possible previous paths, adding with previous metric
            prev_st = state_machine[st]['b1']['prev_st']
            first_b_metric = V[(t-1)][prev_st]["metric"] + \
                bits_diff_num(state_machine[st]['b1']['out_b'], obs[t - 1])
            prev_st = state_machine[st]['b2']['prev_st']
            second_b_metric = V[(t - 1)][prev_st]["metric"] + \
                bits_diff_num(state_machine[st]['b2']['out_b'], obs[t - 1])
            # print(state_machine[st]['b1']['out_b'],obs[t - 1],t)
            if first_b_metric > second_b_metric:
                V[t][st] = {"metric": second_b_metric, "branch": 'b2'}
            else:
                V[t][st] = {"metric": first_b_metric, "branch": 'b1'}

    # print trellis nodes metric:
    # for st in state_machine:
    #     for t in range(0,len(V)):
    #         print(V[t][st]["metric"])
    #     print()
    # print()

    smaller = min(V[t][st]["metric"] for st in state_machine)
    decoded_output = []
    # traceback the path on smaller metric on last trellis column
    for st in state_machine:
        if V[len(obs)-1][st]["metric"] == smaller:
            source_state = st
            for t in range(len(obs), 0, -1):
                branch = V[t][source_state]["branch"]
                decoded_output.append(
                    state_machine[source_state][branch]['input_b'])
                source_state = state_machine[source_state][branch]['prev_st']
            # print (source_state+"\n")
    decoded_file(decoded_output[::-1][:-2])
    # print("Finish")
    # print(decoded_output)
    # return decoded_output