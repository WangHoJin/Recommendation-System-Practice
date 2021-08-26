from django.urls import path

from . import views

urlpatterns = [
        # 같은 디렉토리의 views.py에 가서 index라는 함수를 실행시킨다.
        path('', views.index, name='index'),
        path('train', views.train_view, name='train'),
        path('train_model', views.train_model, name='train_model'),

        path('recomm', views.recomm_main_view, name='recomm'),
        path('recomm/<int:user_id>', views.recomm_user, name='recomm_result'),
        ]
