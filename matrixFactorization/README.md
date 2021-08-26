# Matrix Factorization

### Requirements
- Python 3.7

### How to Run

`python train.py -i data/small -o result/small -a 0`: KNN

`python train.py -i data/small -o result/small -a 1`: MF

`python train.py -i data/small -o result/small -a 2`: PLSI + MF

`python train.py -i data/movielens-1m -o result/movielens-1m`

`python train.py -i data/movielens-10m -o result/movielens-10m`


### Preprocess
`python preprocess.py -i data/movielens/ml-1m_ratings.dat -o data/small -n 700`

`python preprocess.py -i data/movielens/ml-1m_ratings.dat -o data/movielens-1m`

`python preprocess.py -i data/movielens/ml-10m_ratings.dat -o data/movielens-10m`
