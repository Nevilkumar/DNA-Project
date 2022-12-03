import code
from django.shortcuts import render, redirect
import numpy as np
from codec.codes import *
from codec.codes.convolutional import viterbi, viterbi_encoder
from codec.codes.cyclic import cyclic_code_decoder, cyclic_code_encoder
# import sympy

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
            cyclic_code_encoder()
        elif technique == 'cyclic' and scheme == 'decoding':
            cyclic_code_decoder()
        elif technique == 'convolutional' and scheme == 'encoding':
            viterbi_encoder()
        elif technique == 'convolutional' and scheme == 'decoding':
            viterbi()
        elif technique == 'rs' and scheme == 'encoding':
            viterbi()
        elif technique == 'rs' and scheme == 'decoding':
            viterbi()
        return redirect('download')
    return render(request, 'codec/home.html')


def download(request):
    return render(request, 'codec/download.html')
