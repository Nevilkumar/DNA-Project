import numpy as np
from collections import defaultdict

gen_matrix = np.array([[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]])


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

    return encoded_string


def decoding(acgt_string):
    decoded_string = ''
    for ch in acgt_string:
        if ch == 'A' or ch == 'C':
            decoded_string += '0'
        else:
            decoded_string += '1'

    return decoded_string


def convert_to_binary(num):
    bnr = bin(num). replace('0b', '')
    x = bnr[::-1]
    while len(x) < 8:
        x += '0'
        bnr = x[::-1]

    return bnr


def convert_to_decimal(num):
    return int(num, 2)


def msg_codeword_mapping(msg_lst, cw):
    mapping_cw_msg = defaultdict(lambda: 0)
    for x, y in zip(msg_lst, cw):
        mapping_cw_msg[''.join(str(ch) for ch in y)] = x

    return mapping_cw_msg


def mapping_1_255():

    binary_val = []
    for i in range(1, 255):
        binary_val.append(convert_to_binary(i))

    mat = matrix_generator(binary_val, 8)
    mapping = mat@gen_matrix % 2

    mapping_dict = defaultdict(lambda: 0)
    for i in range(len(mapping)):
        mapping_dict[''.join([str(y) for y in mapping[i]])
                     ] = convert_to_binary(i)

    return mapping_dict


def create_codewords(msg):
    msg_matrix = matrix_generator(msg, 8)
    codeword_matrix = np.dot(msg_matrix, gen_matrix)
    return np.mod(codeword_matrix, 2)


def matrix_generator(lst_bin, n):
    msg_matrix = []
    for val in lst_bin:
        temp = []
        for ch in val:
            temp.append(int(ch))
        msg_matrix.append(temp)
    return np.array(msg_matrix).reshape(-1, n)


def makeSystematic(G, verbose=True):
    k, n = G.shape
    if verbose:
        print('unsystematic:')
        print(G.astype(int))
        print()

    # start with bottom row
    for i in range(k-1, 0, -1):
        if verbose:
            s = ''
        # start with most right hand bit
        for j in range(n-1, n-k-1, -1):
            # eleminate bit if it does not belong to identity matrix
            if G[i, j] == 1 and i != j-(n-k):
                if verbose:
                    s += ' + g' + str(k-n+j)
                G[i, :] = (G[i, :] + G[k-n+j, :]) % 2

        if verbose and s != '':
            print('g' + str(i) + ' = g' + str(i) + s)
            print(G.astype(int))
            print()
    return G.astype(int)


def cyclic_code_encoder():
    with open('static/input.txt') as f:
        content = f.readlines()
        cw, msg_lst = [], []
        lst_encoded_text = ''
        for item in content:
            lst_int = [ord(ch) for ch in item]
            lst_bin = [convert_to_binary(num) for num in lst_int]
            codewords = create_codewords(lst_bin)
            cw.extend(codewords)
            msg_lst.extend(lst_bin)
            lst_encoded = ''.join([''.join(str(v)
                                  for v in list(c)) for c in codewords])
            acgt_str = encoding(lst_encoded)
            lst_encoded_text += ''.join(acgt_str)

    with open('static/output.txt', 'w+') as f:
        f.write(lst_encoded_text)


def cyclic_code_decoder():
    n = 12
    with open('static/input.txt') as f:
        content = f.read()
        decoded_string = decoding(content)
        parts = [decoded_string[i:i+n]
                 for i in range(0, len(decoded_string), n)]
        mapping_cw_msg = mapping_1_255()
        decoded_output = ''.join(
            [chr(int(mapping_cw_msg[p], 2)+1) for p in parts])

    with open('static/output.txt', 'w+') as f:
        f.write(decoded_output)
