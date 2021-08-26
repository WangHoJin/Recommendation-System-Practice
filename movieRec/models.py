from django.db import models

# 데이터베이스 시스템에서 SQL로 표현하면 아래와 같음
# CREATE TABLE myapp_user (
#     "id" serial NOT NULL PRIMARY KEY,
#     "name" varchar(100) NOT NULL
# );
# 만일 primary_key=True라고 표시된 컬럼이 테이블에서 없으면 Django는 자동적으로
# IntegerField인 primary key를 만들어 줌
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
    # Viewed의 user 컬럼은 User 테이블의 primary key
    # on_delete=models.CASCADE는 User 테이블에서 삭제되면 Viewed 테이블에서도 삭제가 된다는 의미
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.FloatField()


class Recomm(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    score = models.FloatField()
