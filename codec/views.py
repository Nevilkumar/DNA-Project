from django.shortcuts import render, redirect
from codec.codes import *
from codec.codes.convolutional import viterbi, viterbi_encoder
from codec.codes.cyclic import cyclic_code_decoder, cyclic_code_encoder
from codec.codes.rs import rs_decoding, rs_encoding
from django.contrib.staticfiles.storage import staticfiles_storage

# Create your views here.

def home(request):
    if request.method == 'POST':
        technique = request.POST.get('technique')
        scheme = request.POST.get('scheme')
        codecFile = request.FILES.get('codecFile')
        data = codecFile.file.read().decode('utf-8')
        with open(staticfiles_storage.url('input.txt').strip("/"), 'w+') as f:
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
            rs_encoding()
        elif technique == 'rs' and scheme == 'decoding':
            rs_decoding()
        return redirect('download')
    return render(request, 'codec/home.html')


def download(request):
    return render(request, 'codec/download.html')
