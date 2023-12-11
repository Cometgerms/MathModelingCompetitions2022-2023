'''
This is the code that models the location of the expected hive and plant locations.
Used in: HiMCM Problem 3
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import random
# import pyvista as pv

TERRIAN_HEIGHT = np.array([
    [10, 15, 21, 30, 45, 35, 29, 27, 26, 24, 23, 21, 19, 15, 14, 13, 11, 10, 9, 8, 6, 4, 2, 1, 0, 0, 0, 0],
    [12, 16, 22, 34, 49, 42, 35, 32, 30, 29, 28, 27, 25, 24, 22, 21, 20, 17, 16, 14, 13, 9, 7, 6, 4, 2, 0, 0],
    [15, 18, 24, 39, 51, 47, 40, 37, 34, 31, 29, 26, 24, 22, 21, 19, 17, 16, 14, 11, 9, 8, 6, 5, 2, 0, 0, 0],
    [16, 21, 26, 41, 56, 52, 48, 42, 39, 35, 31, 29, 27, 24, 22, 21, 20, 18, 16, 14, 12, 11, 9, 8, 6, 5, 4, 2],
    [18, 25, 31, 45, 71, 55, 49, 43, 38, 33, 30, 27, 28, 26, 24, 22, 21, 18, 15, 14, 12, 12, 10, 9, 8, 7, 6, 4],
    [27, 31, 38, 51, 86, 74, 52, 44, 39, 32, 28, 24, 23, 19, 22, 20, 19, 15, 12, 14, 13, 12, 11, 9, 8, 7, 6, 5],
    [29, 34, 43, 58, 104, 87, 65, 49, 38, 29, 25, 22, 19, 15, 18, 20, 19, 18, 16, 15, 14, 13, 12, 10, 9, 8, 7, 6],
    [32, 37, 50, 72, 112, 92, 76, 56, 36, 25, 20, 18, 15, 11, 13, 19, 18, 16, 15, 12, 11, 10, 9, 7, 6, 4, 2, 1],
    [35, 40, 56, 84, 128, 103, 75, 54, 32, 21, 15, 13, 10, 6, 7, 17, 16, 14, 12, 11, 10, 8, 6, 3, 2, 1, 0, 0],
    [33, 38, 55, 76, 118, 97, 84, 77, 69, 63, 56, 48, 41, 35, 28, 21, 17, 14, 12, 10, 8, 6, 4, 2, 1, 0, 0, 0],
    [31, 37, 54, 65, 92, 74, 66, 58, 54, 47, 40, 32, 27, 22, 17, 14, 10, 9, 8, 6, 4, 2, 1, 0, 0, 0, 0, 0],
    [29, 35, 48, 52, 68, 79, 52, 45, 39, 36, 34, 32, 26, 23, 18, 13, 11, 10, 7, 6, 2, 1, 0, 0, 0, 0, 0, 0],
    [27, 33, 44, 50, 55, 70, 54, 50, 46, 42, 35, 32, 27, 24, 16, 15, 12, 11, 7, 5, 3, 2, 0, 0, 0, 0, 0, 0],
    [26, 30, 41, 45, 48, 52, 47, 42, 41, 39, 37, 34, 31, 27, 20, 16, 14, 12, 10, 7, 4, 2, 1, 0, 0, 0, 0, 0],
    [25, 24, 22, 21, 19, 17, 16, 15, 14, 12, 11, 10, 8, 6, 5, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [27, 28, 32, 30, 28, 26, 24, 22, 18, 16, 14, 10, 8, 7, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [30, 33, 37, 41, 40, 37, 33, 29, 26, 23, 20, 17, 14, 12, 9, 7, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [35, 39, 48, 45, 41, 38, 35, 32, 29, 28, 25, 21, 17, 14, 10, 8, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [29, 35, 42, 44, 41, 37, 34, 32, 30, 27, 24, 20, 16, 13, 11, 7, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [24, 29, 35, 34, 33, 32, 31, 30, 29, 28, 26, 24, 20, 16, 12, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [20, 25, 31, 29, 28, 27, 26, 24, 22, 20, 19, 17, 16, 14, 12, 10, 8, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [17, 20, 25, 24, 22, 20, 19, 16, 14, 12, 10, 8, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [15, 17, 16, 14, 13, 12, 11, 11, 9, 8, 7, 5, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [14, 15, 13, 12, 11, 9, 8, 7, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [13, 13, 14, 13, 12, 11, 10, 8, 7, 5, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [12, 12, 12, 11, 10, 9, 7, 6, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [12, 12, 11, 10, 8, 6, 7, 5, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [11, 11, 10, 9, 9, 7, 6, 5, 5, 5, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

#Temperature
TEMP_F=np.array([82,83,82.4,82.4,80.4,80.4,79.7,79.2,79.8,80.3,80.7,80,77.8,80,80.1,78.2,78.8,79.9,78.5,79.3,79.6,79.5,79.5,80.3,79.2,79.3,79,78.6,77.8,79.5,80.7,79.5,79.9,78.6,80.7,81.3,81,81.4,79.7,80.8,80.1,79.8,79.3,79.8,79,79.6,80.6,79.6,79,80.8,81.6,80.8,80.1,79.9,79.6,79.3,80.2,81.3,82.2,81.4,81.4,80.3,80.3,80.5,80.2,80.5,80.1,76,76.1,75.9,76.2,77.4,76.3,76.5,77.7,78.4,78.4,78,78.2,78.5,79.4,82.9,81.2,80.7,80.9,79.9,80,79.5,79,78.9,78.6,79,78,79.7,80.4,81.5,80.9,80.2,81.9,79.8,79.9,80.4,79.3,78.9,79.5,80.2,80.5,81,80.8,81.7,82.9,82.3,82.5,81.9,81.9,80.7,81.5,80.7,79.1,78.7,77.3,77.7,78.4,81.2,81.7,81.7,82.2,81,81.3,80.6,81.6,83.1,84.4,83.6,83.5,83.8,82.9,81.8,82,82.2,83.6,85.9,85.3,85.8,85.6,85.2,84,83.8,83.9,83.7,85.2,85.2,84.7,84.8,84.8,84.1,80.4,84.3,83.1,83.4,82.9,84.7,83.8,83.1,80.5,79.3,81.3,84.2,82.2,83.4,83.6,83.7,83.5,84.7,83,84.6,85,83.3,84.4,83.4,83.6,83.8,84.9,84.7,82,83.8,85.1,84.9,84.3,83.8,80.4,80.6,83.5,82.8,80.5,83.1,82.9,82.8,83.4,84.7,83.2,80.8,82.8,80.1,82.9,83,83.2,84.4,83.5,80.9,83.3,80,81.9,84,84.3,85.3,83.8,77.3,79.2,82.4,83.6,82.7,84.7,84.5,82.2,83.9,83.7,82.6,81,80.3,82.5,83.5,84.3,82.5,80.5,82.8,83.6,84.1,84.5,83.4,83.3,81.7,83.8,84.3,82.9,79.2,76,79.3,82,82.8,82.1,83.2,84,83.4,82.6,83.8,83.8,83.2,81.8,81.8,84.3,84.6,82.1,83.2,82.2,82.6,81.8,81.7,80.9,79.8,83.7,84.6,82.2,83.3,82.6,82.6,83.4,83.6,81.5,82.6,83.7,83.4,83.6,84.1,84.1,84.8,84.7,83.4,83.2,84.2,83.4,83.9,82.4,83.7,83.8,83.4,84.4,83.8,82.5,82.9,82.3,82.5,82.4,83.2,83.1,83.5,83.6,82.9,83.3,82.9,83.5,81.7,77.3,79,80.9,81.3,81.4,80.3,81.4,82.5,82,81.7,80.4,79.6,79.6,78.7,79.3,80.7,80.3,79.8,79.6,80,80.7,80.8,80.8,81.5,81.1,81.1,80.2,80.8,79.5,79.9,80.3,80.7,81.1,80.9,81,82.1,81.4,81.5,81.3,80.4,80.8,81,81.7,80.1,79.7,78.8,78,79.1,81.3,80.7,80.2,79.6,79.4])
TEMP_C=[]

for i in range(365):
    C=round((TEMP_F[i]*10-320)*5/9)/10
    TEMP_C.insert(i,C)

TEMP=np.array(TEMP_C)

#RAIN_HOURS_DURING_DAYLIGHT = np.array([0])
#DAYLIGHT_HOURS = np.array([8])

#Wind Speed
WIND_M=np.array([5.4,4.6,4.4,5.4,4.7,4.6,3.7,4.6,5.1,4.4,4.3,3.7,4,4.9,5.5,3.6,3.6,3.4,2.6,2.7,4,4.3,3,3.6,4.2,3.7,3.4,3.8,3.1,3.7,4.9,5.1,4.3,4.1,3.9,2.8,3.1,4.4,4.9,3.8,4,4.2,4.6,4.1,4.7,5.8,5.2,4.3,5,5.2,5.1,4.3,4.7,5.1,4.9,5.1,3.9,4.2,4.4,3.4,3,5.3,7,4.8,4.5,5.7,5.4,4.3,5.5,6.6,5.8,5,5.8,6,5.7,4.7,5.3,5.7,6.2,4.8,5.2,6.2,4.5,2.7,5.6,4.9,5.3,5.6,6.9,5.2,6.6,6.5,4.1,4.9,3.8,4,4.6,3.9,5.3,4.4,5.2,5.7,5.4,4.1,5,5.7,5.3,3.5,5,4.1,4.8,5.7,4.3,4.3,4.5,4.4,6,6.6,6.2,6.8,7.4,5.5,4,3.6,4.5,5.7,4.8,4.7,5.3,4.3,3.2,3.2,4.7,4.9,5.1,5.5,6.8,6.7,5.4,5.1,3.7,5.2,5.8,5.9,6.2,7.5,6.4,6.2,3.9,4.2,6.1,4.5,4.9,4.1,4.5,5.8,4.2,6.3,4.7,4.8,4.9,3.8,5.5,6.1,5.6,6.1,4.8,6.3,6.2,4.2,3.8,4.4,4.5,5.3,5.7,5.9,4.6,6.1,5.6,5.5,5.5,6.8,4.4,5.8,5.1,4.2,5.9,4.9,6.2,5.4,5,4.8,3.6,3.5,5.3,3.3,3.7,3.1,5.2,6.9,4.7,6.9,4.5,6.4,3.3,3.8,5,4.4,5.1,6.8,3.8,3.3,5.6,5.1,6.4,6.3,5.5,3.6,4.4,4.6,4.6,4.9,3.9,4.4,4.6,3.6,2.9,3.6,4.9,5.2,5.7,5,4.5,5.1,6,6.4,6.3,4.6,4.1,3.6,4.2,4.5,3.8,4.2,3.9,5.8,7.9,4.7,4.6,4.8,5.3,5.2,3.9,3.3,4.3,4.4,3.1,3.1,6.1,5.1,3.8,4.4,5.1,5,5.8,4.1,3.4,2.8,5.4,4.3,7,5.2,3.8,4.2,5.2,4.6,5.8,4.3,5.1,3.2,4.5,4.5,6.8,2.8,5.4,5.1,4.3,4.2,4.4,5.2,4.5,3.5,4.5,4.5,3.9,4.4,5.3,4.7,4.9,4.5,5.2,3.2,4.7,4.8,5.3,4.2,4.2,4.4,4.3,6.5,5.3,3.5,4.7,4.2,3.7,3.1,4,3.9,7,4,3.5,4.1,5.3,3.8,4.3,4.9,3.6,3.2,5.1,5.3,4.1,3.8,3.4,4.1,4,4.7,4.5,4.2,3.4,3.5,3.5,3.1,3.3,5,5.8,4.9,3.4,2.5,3.3,3.5,3.4,3.4,3.1,4.9,2.9,4,3.7,3.7,3.3,4.2,3.2,4.7,4.1,3.6,4.1])
WIND_K=[]

for i in range(365):
    K=round(WIND_M[i]*16.09)/10
    WIND_K.insert(i,K)

WIND=np.array(WIND_K)

FORAGING_BEE = np.array([44438*0.16]*365)
TPTO = 44.07



def kterrain(x,y):
    if x==0: dif_x = TERRIAN_HEIGHT[x+1][y]
    elif x == 27: dif_x = TERRIAN_HEIGHT[x-1][y]
    else: dif_x = abs(TERRIAN_HEIGHT[x-1][y]-TERRIAN_HEIGHT[x+1][y])

    if y == 0: dif_y = TERRIAN_HEIGHT[x][y+1]
    elif y == 27: dif_y = TERRIAN_HEIGHT[x][y-1]
    else: dif_y = abs(TERRIAN_HEIGHT[x][y-1]-TERRIAN_HEIGHT[x][y+1])

    k_land = max(dif_y, dif_x)
    return k_land

# for i in range(28):
#     for j in range(28):
#         if TERRIAN_HEIGHT[i][j]>=100:
#             FLO=[i,j,0]
#         else:
#             kt=kterrain(i,j)
#             if kt==0:
#                 FLO=[i,j,3000]
#             else:
#                 f=int(3000/kt)
#                 FLO=[int(i),int(j),f]
#         period.append([2+2*random.random()-1, 12+2*random.random()-1])
#     flower.insert(i,FLO)
#     FLO=[]

# print(FLOWER)
# print(FLOWERING_PERIOD)


# NUMBER_OF_FLOWER_PER_GRID = np.array([500])
HIVE = np.array([[]])
VELOCITY = 533.33 #m/min
TERRIAN = np.array([[]])
TERRIAN_AVA = np.ones((28, 28))
# print(TERRIAN_AVA)

TERRIAN_FOT = np.zeros((28, 28), dtype=int)


def PHI_SHADE(x, y):
    phishade = 0
    return phishade

def PHI_WATER(x, y):
    phiwater = 0
    return phiwater

def PHI_HEIGHT(x, y):
    phiheight = 0
    return phiheight

def INDEX_TEMP(w): #计算INDEXTEMP
    # print(789)
    if TEMP[w]<=10:
        # print(87)
        indextemp = 0
    elif (TEMP[w]>10) & (TEMP[w]<=22):
        # print(3)
        indextemp = (TEMP[w]-10)/12
    elif (TEMP[w]>22) & (TEMP[w]<=32):
        indextemp = 1
        # print(1)
    elif (TEMP[w]>32) & (TEMP[w]<=40):
        # print(4)
        indextemp = (40-TEMP[w])/8
    elif TEMP[w]>40:
        # print(5)
        indextemp = 0
    return indextemp

'''
def INDEX_RAIN(w):
    indexrain = 1 - RAIN_HOURS_DURING_DAYLIGHT[w]/DAYLIGHT_HOURS
    return indexrain
'''

def INDEX_WIND(w):
    if WIND[w]>=24.1:
        indexwind = 0
    else:
        indexwind =1
    return indexwind


def INDEX_FLIGHT(indexwind, indextemp):
    indexflight = indexwind*indextemp
    return indexflight

def delta(x1, y1, x2, y2): #得改成三维
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def deltaz(x1, y1, z1, x2, y2, z2):
    return math.sqrt(math.sqrt((x1-x2)**2 + (y1-y2)**2)+(z1-z2)**2)

def p(d, Q):
    t = math.ceil((TPTO + 11.3)/(TPTO - 2*d/VELOCITY)*Q)
    # print('123: ', t, '|', 2*d/VELOCITY)
    return t

N = 28
#Phi
for i in range(N):
    for j in range(N):
        #phi_drain
        if i==0: dif_x = TERRIAN_HEIGHT[i+1][j]
        elif i == (N-1): dif_x = TERRIAN_HEIGHT[i-1][j]
        else: dif_x = abs(TERRIAN_HEIGHT[i-1][j]-TERRIAN_HEIGHT[i+1][j])

        if j == 0: dif_y = TERRIAN_HEIGHT[i][j+1]
        elif j == (N-1): dif_y = TERRIAN_HEIGHT[i][j-1]
        else: dif_y = abs(TERRIAN_HEIGHT[i][j-1]-TERRIAN_HEIGHT[i][j+1])

        k_land = max(dif_y, dif_x)

        if TERRIAN_FOT[i][j] == 1: #有树的时候
            phi_drain = 1
        elif (k_land>=0) & (k_land<=20):
            phi_drain = 1-(abs(10-k_land)/10)
        else:
            phi_drain = 0
        TERRIAN_AVA[i][j] = phi_drain

MAX_NEED_OF_HIVE = 0
SEASON = 0
CLOSEST_HIVE_COR = [[-1, -1]]
NUMBER_OF_HIVE = 0

#flower
FLO=[]
flower=[]
period=[]
PER=[]
memo_x = []
memo_y = []

POSSIBLE_QI=[k for k in range(15, 31)]
POSSIBLE_DENSITY = [l for l in range(50, 120, 10)]
y = []
for sen in range(1, 150):
    NUMBER_OF_FLOWER = sen
    print('sen=',sen)
    flower = []
    period = []
    for i in range(NUMBER_OF_FLOWER):
        #Flower
        INDEX_P_QI = random.randint(0, 15)
        INDEX_P_D = random.randint(0, 6)
        rand_x = random.randint(0, 27)
        rand_y = random.randint(0, 27)
        Q = POSSIBLE_QI[INDEX_P_QI]*POSSIBLE_DENSITY[INDEX_P_D]
        flower.append([rand_x, rand_y, Q])
        min_p = random.randint(0, 250)
        max_p = random.randint(min_p+120, min_p+180)
        if max_p>365: max_p=365
        period.append([min_p, max_p])

    FLOWER=np.array(flower)
    FLOWERING_PERIOD=np.array(period)

    for w in range(365):
        AVA_HIVE_COR = np.array([[]])
        #可用蜂巢
        AVA_HIVE_NUM = 0
        for i in range(N):
            for j in range(N):
                if TERRIAN_AVA[i][j]>0.6:
                    t = np.array([i, j])
                    AVA_HIVE_COR = np.append(AVA_HIVE_COR, t)
                    AVA_HIVE_NUM+=1

        #单个蜂巢理论可用蜜蜂数量
        IndexWind = INDEX_WIND(w)
        IndexTemp = INDEX_TEMP(w)
        IndexFlight = INDEX_FLIGHT(IndexWind, IndexTemp)
        B = int(0.5 * FORAGING_BEE[0] * IndexFlight)
        ITER = 0
        AVA_HIVE_COR = AVA_HIVE_COR.reshape(AVA_HIVE_NUM, 2)
        # AVA_HIVE_COR = np.concatenate(AVA_HIVE_COR, [[0, 1]])
        # print(AVA_HIVE_COR)
        # print('w=                                                ', w)
        for i in FLOWER:
            if (FLOWERING_PERIOD[ITER][0]<=w) & (FLOWERING_PERIOD[ITER][1]>=w):
                MIN_DISTANCE = 10e6
                for j in AVA_HIVE_COR:
                    distance = 10 * deltaz(int(i[0]), int(i[1]), TERRIAN_HEIGHT[int(i[0])][int(i[1])], int(j[0]), int(j[1]), TERRIAN_HEIGHT[int(j[0])][int(j[1])])
                    if distance<MIN_DISTANCE:
                        MIN_DISTANCE = distance
                        CLOSEST_HIVE_CO = j
                MIN_NEEDED_NUMBER_OF_BEE = p(MIN_DISTANCE, FLOWER[ITER][2])
                NEED_NUMBER_OF_HIVE = MIN_NEEDED_NUMBER_OF_BEE/B
                NUMBER_OF_HIVE += NEED_NUMBER_OF_HIVE
            ITER+=1

        MAX_NEED_OF_HIVE = max(MAX_NEED_OF_HIVE, NUMBER_OF_HIVE)
        NUMBER_OF_HIVE = 0



    print('Lambda=', int(MAX_NEED_OF_HIVE))
    y.append(math.ceil(MAX_NEED_OF_HIVE))

print(y)