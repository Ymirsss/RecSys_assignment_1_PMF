from numpy import *
import random
import csv

def getData(file='ml-latest-small/ratings.csv'):
    li = []
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            # print(row)
            (userid, movieid, rating, ts) = row
            li.append([int(userid), int(movieid),float(rating)])
    return array(li)

