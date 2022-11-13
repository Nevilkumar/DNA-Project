from django.shortcuts import render, redirect
import numpy as np
# import sympy

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

def convert_to_binary(num):
    bnr = bin(num). replace('0b','')
    x = bnr[::-1] #this reverses an array.
    while len(x) < 8:
        x += '0'
        bnr = x[::-1]
    return bnr

def convert_to_decimal(num):
    return int(num,2)

def create_codewords(msg, codeword_bits):
    # gen_matrix = np.random.randint(2, size=(8,codeword_bits))
    # A_matrix = np.random.randint(2, size=(8,4))
    # gen_matrix = np.concatenate((np.eye(8, dtype='int32'),A_matrix),axis=1)
    gen_matrix = np.array([[1,1,0,0,0,0,0,1,0,0,0,0],[0,1,1,0,0,0,0,0,1,0,0,0],[0,0,1,1,0,0,0,0,0,1,0,0],[0,0,0,1,1,0,0,0,0,0,1,0],[0,0,0,0,1,1,0,0,0,0,0,1],
      [1,0,0,0,0,1,1,0,0,0,0,0],[0,1,0,0,0,0,1,1,0,0,0,0],[0,0,1,0,0,0,0,1,1,0,0,0]])
    msg_matrix = matrix_generator(msg, 8)
    codeword_matrix = np.dot(msg_matrix, gen_matrix)
    return np.mod(codeword_matrix,2), gen_matrix


def matrix_generator(lst_bin, n):
    msg_matrix = []
    for val in lst_bin:
        temp = []
        for ch in val:
            temp.append(int(ch))
        msg_matrix.append(temp)
    return np.array(msg_matrix).reshape(-1,n)

def cyclic_code_encoder(file, num_bits):
    with open(f'static/{file}') as f:
        content = f.readlines()
        print(content)
        cw = []
        lst_encoded_text = ''
        for item in content:
            lst_int = [ord(ch) for ch in item]
            lst_bin = [convert_to_binary(num) for num in lst_int]
            codewords, gen_matrix = create_codewords(lst_bin, codeword_bits=num_bits)
            cw.extend(codewords)
            lst_encoded = [''.join(str(v) for v in list(c)) for c in codewords]
            acgt_str = [encoding(c) for c in lst_encoded]
            lst_encoded_text += ''.join(acgt_str)    
    
    with open('static/output.txt', 'w+') as f:
        f.write(lst_encoded_text)

# Create your views here.

def home(request):
    if request.method == 'POST':
        technique = request.POST.get('technique')
        scheme = request.POST.get('scheme')
        codecFile = request.FILES.get('codecFile')
        data = codecFile.file.read().decode('utf-8')
        with open('static/input.txt', 'w+') as f:
            f.writelines(data)
        if technique == 'cyclic' and scheme == 'encoding':
            cyclic_code_encoder('input.txt', 12)
        return redirect('download')
    return render(request, 'codec/home.html')

def download(request):
    return render(request, 'codec/download.html')