from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader, RequestContext

from .models import *
from .loadResult import load_result


def index(request):
    return render(request, 'movieRec/base.html', {})


def train_view(request):
    context = {}
    context['train_finished'] = False
    return render(request, 'movieRec/train.html', context)


n_recomm = 10
def recomm_main_view(request):
    global n_recomm
    n_recomm = request.POST.get('n_recomm')
    if n_recomm is None: n_recomm = 10
    user_list = User.objects.all()
    return render(request, 'movieRec/recomm.html', {'user_list':user_list, 'n_recomm':n_recomm})


def recomm_user(request, user_id):
    global n_recomm
    target_user = User.objects.get(id=user_id)
    viewed_list = Viewed.objects.filter(user=target_user).order_by('-rating')
    viewed_paginator = get_paginated_list(request, viewed_list)
    recomm_list = Recomm.objects.filter(user=target_user).order_by('-score')[:int(n_recomm)]
    return render(request, 'movieRec/recommResult.html', {'target_user':target_user, 'viewed_page':viewed_paginator, 'recomm_list':recomm_list})


def get_paginated_list(request, obj_list, page_size=10):
    paginator = Paginator(obj_list, page_size)
    page = request.GET.get('page')
    return paginator.get_page(page)


import os
def train_model(request):
    model = request.POST.get('model')
    if model == 'KNN': run_kNN(request)
    elif model == 'MF': run_MF(request)
    elif model == 'MF_PLSI': run_MF_PLSI(request)
    request.POST = {}
    return render(request, 'movieRec/train.html', {'train_finished':True, 'model':model})


def run_kNN(request):
    k = request.POST.get('param_k')
    os.system('cd matrixfactorization& python train.py -i data/tiny -o result/tiny -a 0 -k %s'%k)
    load_result('matrixfactorization')


def run_MF(request):
    os.system('cd matrixfactorization& python train.py -i data/tiny -o result/tiny -a 1')
    load_result('matrixfactorization')


def run_MF_PLSI(request):
    d = request.POST['param_nz']
    os.system('cd matrixfactorization& python train.py -i data/tiny -o result/tiny -a 2 -d %s'%d)
    load_result('matrixfactorization')

