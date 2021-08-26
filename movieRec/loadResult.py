import sqlite3
from os.path import join





def load_movies(cur):
    path = 'matrixFactorization/data/movielens-1m/ml-1m_movies.dat'
    cur.execute('DELETE FROM movieRec_movie')
    with open(path, encoding='latin-1') as f:
        for line in f:
            #line = line.replace('\'', '\\\'')
            token = line.strip().split('::')
            id = token[0]
            title, year = token[1].rsplit(' ', 1)
            year = year[1:-1]
            text = token[2]
            #print(id, title, year)
            #print('INSERT INTO movieRec_movie(id, title, text) VALUES(\'%s\',\'%s\',\'%s\')'%(tuple(token)) )
            try: cur.execute('INSERT INTO movieRec_movie(id, title, text, year) VALUES(?,?,?,?)', (id, title, text, year))
            except: pass


def load_viewed(cur, model_home):
    path = join(model_home, 'train_ratings.txt')
    cur.execute('DELETE FROM movieRec_viewed')
    with open(path) as f:
        for line in f:
            token = line.strip().split('::')
            cur.execute('SELECT COUNT(*) FROM movieRec_user WHERE id=?', (token[0],))
            n_rows = cur.fetchall()[0][0]
            if n_rows == 0:
                cur.execute('INSERT INTO movieRec_user(id, name) VALUES(?,?)', (token[0], 'User%05d'%(int(token[0]))))
            cur.execute('INSERT INTO movieRec_viewed(user_id, movie_id, rating) VALUES(?,?,?)', (token[0], token[1],token[2]) )

def load_recomm(cur, model_home):
    pass

def load_result(model_home):
    conn= sqlite3.connect('./db.sqlite3')
    cur = conn.cursor()
    print("LOAD MOVIEDATA")
    load_movies(cur)
    print("LOAD VIEWED DATA")
    load_viewed(cur, model_home)
    print("LOAD RECOMM DATA")
    load_recomm(cur, model_home)
    print("load_result IS FINISHED")
    conn.commit()
    conn.close()


if __name__ == '__main__':
    conn= sqlite3.connect('../db.sqlite3')
    cur = conn.cursor()
    #load_movies()
    #load_viewed(cur)
    #load_recomm()
    conn.commit()
    conn.close()


