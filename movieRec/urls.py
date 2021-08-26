from django.urls import path

from . import views

urlpatterns = [
        path('', views.index, name='index'),
        path('train', views.train_view, name='train'),
        path('train_model', views.train_model, name='train_model'),

        path('recomm', views.recomm_main_view, name='recomm'),
        path('recomm/<int:user_id>', views.recomm_user, name='recomm_result'),
        ]
