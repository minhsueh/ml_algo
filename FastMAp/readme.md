Goal:

Embed the objects in fastmap-data.txt into a 2D space

##########################

Data: 

fastmap-wordlist.txt: contain the words as objects

fastmap-data.txt: first two column is the ID of the words (ID: the row number correspond to fastmap-wordlist.txt), the third is the Damerau–Levenshtein distance
between two words

##########################

In FastMap class, there are five main functions: dis_cal, project_dis_cal, Farest_pair, fastmap, and plot_2d.

Note that in dis_cal function, the distance between two points for all points are not calculated once, instead,

it substrate the distance between all saved points (xi and xj) in previous iteration when dis_cal function is called.

This implementation may not be necessary in this assignment since there are only ten objects.

However, it will save time when the object number increase because calculating all distance between two points is O(n^2).

![alt text](https://github.com/minhsueh/ml_algo/blob/master/FastMAp/hw3_FastMap.png)
