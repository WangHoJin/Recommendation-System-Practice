from django.db import models



class User(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)


class Movie(models.Model):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=200)
    year = models.IntegerField(null=False)
    img = models.ImageField(blank=True, null=True)
    text = models.CharField(max_length=1000)


class Viewed(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.FloatField()


class Recomm(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    score = models.FloatField()
