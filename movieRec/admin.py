from django.contrib import admin

from .models import User, Movie, Viewed, Recomm

admin.site.register(User)
admin.site.register(Movie)
admin.site.register(Viewed)
admin.site.register(Recomm)
