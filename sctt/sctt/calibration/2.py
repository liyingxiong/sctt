'''
Created on Apr 24, 2015

@author: Yingxiong
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.special import gamma

# lack_of_fit = np.array([[12.82251467,   10.20685425,    8.27354945,    8.14410753,  8.25679561, 11.15801952,    9.63770183,   12.01023073,   14.61838781,   16.53279184],
#                         [10.64295173,    9.56557102,    8.18743326,    7.1706128,     6.97872589,
#                             8.5895802,     9.71629218,   15.0104849,    17.90890169,   21.31461396],
#                         [9.00344145,    7.41488313,    7.12253766,    7.48455667,    8.44381611,
#                             8.99890582,   12.72356907,   15.61283174,   15.95467423,   19.34253657],
#                         [8.78547238,    7.70955343,    6.35651271,    7.41863858,    7.19215174,
#                             8.15792631,   12.14421289,   14.43344402,   16.82240274,   17.43650238],
#                         [8.57675328,    6.89792763,    7.114425,      6.34808409,    7.48044455,
#                             10.02758673,   12.06407227,   13.19910209,   21.16586551,   21.973517],
#                         [7.80149665,    6.8768933,     6.61347345,    7.71244169,    9.00331374,
#                             13.59453413,   13.84835524,   14.42014821,   18.35692521,   24.07074612],
#                         [6.94829052,    6.20632936,    6.03648067,    7.77732429,    8.31760086,
#                             9.64318099,   12.84477582,   18.47181893,   23.24005325,   25.0595838],
#                         [6.47433038,    7.82972137,    6.31514535,    7.62625205,    9.82121109,
#                             12.62384889,   12.16043557,   15.256451,     17.44007743,  20.321047347],
#                         [7.76158383,    6.36671007,    8.38782711,    7.80428504,    8.93344906,
#                             9.19167981,   13.32383411,  16.68248879,   26.68934662,   23.87758229],
#                         [6.54703951,    5.95124253,    7.1208579,     7.40794547,    8.17874811,   12.42130351,   15.46767452,   18.72039341,   25.98954232,  23.8423568461]])

# cs = np.array([[16.66666667,  17.85714286,  19.23076923,  20.83333333,  23.80952381,   26.31578947,  25.,          31.25,        35.71428571,  33.33333333],
#                [17.85714286,  20.,          19.23076923,  21.73913043,  22.72727273,
#                    25.,   27.77777778,  33.33333333,  31.25,        45.45454545],
#                [16.12903226,  19.23076923,  20.,          21.73913043,  23.80952381,
#                    26.31578947,  31.25,        33.33333333,  35.71428571,  45.45454545],
#                [17.24137931,  17.85714286,  20.83333333,  21.73913043,  23.80952381,
#                    26.31578947,  31.25,        33.33333333,  38.46153846,  41.66666667],
#                [18.51851852,  18.51851852,  20.,          22.72727273,  22.72727273,
#                    27.77777778,  31.25,        31.25,        38.46153846,  50.],
#                [17.24137931,  18.51851852,  21.73913043,  23.80952381,  25.,
#                    27.77777778,   31.25,        35.71428571,  41.66666667,  50.],
#                [16.66666667,  19.23076923,  18.51851852,  26.31578947,  25.,
#                    31.25,   33.33333333,  35.71428571,  41.66666667,  50.],
#                [16.12903226,  22.72727273,  21.73913043,  22.72727273,  25.,
#                    33.33333333,   33.33333333,  35.71428571,  38.46153846,  38.4615384615],
#                [17.85714286,  20.83333333,  21.73913043,  26.31578947,  26.31578947,
#                    26.31578947,  31.25,        38.46153846,  45.45454545,  45.45454545],
#                [17.85714286,  20.,          23.80952381,  23.80952381,  26.31578947,   29.41176471,  33.33333333,  33.33333333,  41.66666667,  45.45454545]])

lack_of_fit = np.array([[26.42848948,  28.57561277,  23.8571358,  23.63841703,
                         22.60272972,  21.31295049,  20.48185342,  18.74251193,
                         15.72704456,  14.99333278,  24.65684132,  24.22602092,
                         17.55890246,  18.28059543,  16.95782575],
                        [19.20444694,  17.0640242,  14.54365578,  13.91186435,
                         14.73698694,  15.00593991,  12.51904185,  17.29497638,
                         17.02427462,  13.48913165,  23.6223163,  17.46107529,
                         16.83068623,  18.80065816,  20.25803799],
                        [12.85580656,  12.45708493,  12.73821826,  12.46199897,
                         12.79737937,  11.28526805,  12.74763286,  14.20941447,
                         14.56373059,  15.89122952,  17.05499253,  18.68597478,
                         23.2613684,  24.70035073,  31.59359769],
                        [11.78722086,  12.94277799,  12.34385622,  13.76299058,
                         11.18211461,  14.15240899,  14.46333802,  15.74550164,
                         18.37518023,  17.81204511,  24.86727087,  24.1925094,
                         26.9775026,  32.77536613,  29.71624173],
                        [10.46418125,  12.37363601,  10.60475124,  11.6740898,
                         14.237945,  15.41397659,  14.52757911,  14.30893017,
                         17.01797298,  19.59784788,  20.49100922,  28.72476032,
                         25.9536291,  33.09490406,  35.48579687],
                        [10.13013816,   9.63731785,  10.73263899,  11.78550624,
                         11.98729448,  13.11759944,  16.74843112,  16.67699857,
                         20.45163257,  23.83460459,  19.26339835,  27.55022129,
                         33.75622622,  34.41976469,  29.98220816],
                        [9.50857789,   9.37067697,   9.12206294,  11.52046678,
                         11.92428084,  15.31081117,  16.16465792,  17.5265716,
                         18.99102307,  21.36141723,  25.15935398,  28.69752883,
                         23.8040864,  34.34673294,  47.2671262],
                        [10.60854632,  10.22004813,  10.71455733,  10.7041076,
                         12.87258809,  14.57992685,  15.1635562,  20.94381801,
                         23.31401528,  24.16082361,  25.63484897,  27.04567417,
                         35.56714713,  31.64095032,  28.96182798],
                        [8.71903686,  10.10184921,  10.84261881,  11.49313856,
                         12.14867435,  13.86885385,  18.06177965,  21.75500297,
                         25.02344864,  22.70654888,  27.13154807,  38.02104752,
                         33.97052529,  39.87165433,  38.8137727],
                        [9.22164439,  10.83248959,   9.65118708,  12.26954493,
                         13.72313383,  16.96302166,  20.59015996,  18.95502432,
                         25.12927555,  26.86332451,  33.40900502,  36.73942365,
                         24.84788097,  38.52491368,  38.46408335],
                        [8.6510603,   9.3561705,   9.68148821,   9.57699694,
                         13.8610077,  14.23339259,  16.53261869,  21.14029307,
                         20.2441627,  29.7109697,  28.29231052,  28.5969626,
                         30.53427514,  44.26199315,  43.4673862],
                        [10.36368745,   8.45250916,  10.50756123,  13.25768917,
                         14.62434143,  15.78688332,  14.7466508,  20.58550209,
                         23.68926893,  27.85509588,  34.25302123,  31.89899565,
                         36.61539487,  40.70363638,  46.14074069],
                        [8.49249071,   9.84763271,  11.1157752,  11.12682385,
                         15.49738367,  14.59648139,  14.0241624,  18.96233157,
                         21.05998244,  32.56978198,  30.32220861,  36.44078874,
                         28.73536128,  35.03788771,  38.69093079],
                        [8.21880365,   8.85633571,   9.83626094,  10.61066563,
                         16.09014805,  20.08034022,  17.90239418,  22.42474517,
                         21.62950552,  25.36597616,  32.08483105,  33.98192462,
                         31.45525735,  34.73021256,  41.04288538],
                        [8.31520361,   8.19793167,   9.72936113,   9.90068135,
                         11.66114488,  16.67023274,  17.49158088,  17.69420634,
                         22.12287309,  27.88258316,  34.62895158,  36.31659346,
                         31.17032315,  40.76257598,  39.42217479]])

# cs = np.array([[13.51351351,  14.28571429,  13.15789474,  12.19512195,
#                 11.36363636,  16.12903226,  20.83333333,  20.83333333,
#                 19.23076923,  17.85714286,  20.83333333,  31.25,
#                 31.25,  31.25,  26.31578947,  20.83333333,
#                 17.24137931,  16.66666667,  20.83333333,  29.41176471,
#                 35.71428571,  50.,  51.42857143,  55.55555556,
#                 45.45454545],
#                [14.70588235,  15.625,  14.28571429,  13.15789474,
#                 12.19512195,  16.66666667,  20.83333333,  20.83333333,
#                 19.23076923,  17.85714286,  20.83333333,  31.25,
#                 31.25,  29.41176471,  26.31578947,  22.72727273,
#                 19.23076923,  18.51851852,  23.80952381,  31.25,
#                 38.46153846,  45.45454545,  62.5,  55.55555556,  50.],
#                [15.625,  17.85714286,  16.66666667,  15.15151515,
#                 13.88888889,  17.85714286,  21.73913043,  21.73913043,
#                 20.83333333,  19.23076923,  21.73913043,  29.41176471,
#                 27.77777778,  27.77777778,  26.31578947,  23.80952381,
#                 21.73913043,  20.83333333,  25.,  29.41176471,
#                 33.33333333,  38.46153846,  45.45454545,  50.,  50.],
#                [16.12903226,  19.23076923,  17.85714286,  16.66666667,
#                 15.15151515,  18.51851852,  21.73913043,  22.72727273,
#                 20.83333333,  20.,  21.73913043,  27.77777778,
#                 27.77777778,  27.77777778,  26.31578947,  25.,
#                 23.80952381,  23.80952381,  26.31578947,  31.25,
#                 33.33333333,  38.46153846,  41.66666667,  45.45454545,
#                 45.45454545],
#                [17.24137931,  20.83333333,  19.23076923,  17.85714286,
#                 16.66666667,  18.51851852,  22.72727273,  23.80952381,
#                 21.73913043,  20.,  21.73913043,  26.31578947,
#                 26.31578947,  26.31578947,  26.31578947,  26.31578947,
#                 26.31578947,  26.31578947,  29.41176471,  31.25,
#                 33.33333333,  35.71428571,  38.46153846,  41.66666667,
#                 45.45454545],
#                [17.85714286,  21.73913043,  20.,  18.51851852,
#                 17.24137931,  19.23076923,  22.72727273,  23.80952381,
#                 22.72727273,  20.83333333,  21.73913043,  25.,
#                 25.,  26.31578947,  26.31578947,  27.77777778,
#                 27.77777778,  29.41176471,  31.25,  33.33333333,
#                 33.33333333,  35.71428571,  35.71428571,  41.66666667,
#                 45.45454545],
#                [17.85714286,  22.72727273,  20.83333333,  19.23076923,
#                 17.85714286,  19.23076923,  23.80952381,  25.,
#                 22.72727273,  20.83333333,  21.73913043,  23.80952381,
#                 25.,  26.31578947,  27.77777778,  29.41176471,
#                 31.25,  33.33333333,  35.71428571,  35.71428571,
#                 35.71428571,  35.71428571,  35.71428571,  41.66666667,  50.],
#                [16.66666667,  22.72727273,  20.,  18.51851852,
#                 17.24137931,  20.,  23.80952381,  25.,
#                 22.72727273,  20.83333333,  21.73913043,  25.,
#                 26.31578947,  27.77777778,  29.41176471,  31.25,
#                 33.33333333,  35.71428571,  35.71428571,  35.71428571,
#                 35.71428571,  35.71428571,  35.71428571,  41.66666667,
#                 45.45454545],
#                [16.12903226,  21.73913043,  19.23076923,  17.85714286,
#                 16.66666667,  20.,  23.80952381,  25.,
#                 22.72727273,  20.83333333,  21.73913043,  25.,
#                 26.31578947,  26.31578947,  27.77777778,  29.41176471,
#                 33.33333333,  35.71428571,  35.71428571,  35.71428571,
#                 35.71428571,  38.46153846,  38.46153846,  41.66666667,
#                 45.45454545],
#                [17.85714286,  21.73913043,  19.23076923,  17.85714286,
#                 16.66666667,  19.23076923,  23.80952381,  25.,
#                 22.72727273,  20.83333333,  21.73913043,  23.80952381,
#                 25.,  26.31578947,  27.77777778,  29.41176471,
#                 33.33333333,  35.71428571,  33.33333333,  33.33333333,
#                 35.71428571,  38.46153846,  41.66666667,  45.45454545,  50.],
#                [19.23076923,  21.73913043,  19.23076923,  17.85714286,
#                 16.66666667,  19.23076923,  23.80952381,  26.31578947,
#                 23.80952381,  21.73913043,  22.72727273,  25.,
#                 26.31578947,  29.41176471,  29.41176471,  31.25,
#                 31.25,  31.25,  31.25,  31.25,
#                 35.71428571,  38.46153846,  45.45454545,  50.,  50.],
#                [18.51851852,  21.73913043,  20.,  18.51851852,
#                 17.24137931,  20.,  25.,  27.77777778,
#                 25.,  22.72727273,  22.72727273,  23.80952381,
#                 25.,  27.77777778,  29.41176471,  29.41176471,
#                 31.25,  33.33333333,  35.71428571,  35.71428571,
#                 38.46153846,  41.66666667,  45.45454545,  45.45454545,  50.],
#                [17.85714286,  21.73913043,  20.,  18.51851852,
#                 17.24137931,  20.,  26.31578947,  29.41176471,
#                 26.31578947,  23.80952381,  25.,  27.77777778,
#                 29.41176471,  31.25,  31.25,  33.33333333,
#                 33.33333333,  35.71428571,  35.71428571,  38.46153846,
#                 41.66666667,  41.66666667,  41.66666667,  50.,
#                 55.55555556],
#                [16.66666667,  21.73913043,  20.,  18.51851852,
#                 17.24137931,  17.85714286,  25.,  31.25,
#                 27.77777778,  25.,  25.,  26.31578947,
#                 29.41176471,  31.25,  33.33333333,  35.71428571,
#                 35.71428571,  35.71428571,  35.71428571,  35.71428571,
#                 38.46153846,  45.45454545,  50.,  50.,  50.],
#                [16.66666667,  25.,  22.72727273,  20.83333333,
#                 20.,  20.83333333,  25.,  26.31578947,
#                 23.80952381,  21.73913043,  22.72727273,  26.31578947,
#                 29.41176471,  33.33333333,  35.71428571,  35.71428571,
#                 35.71428571,  35.71428571,  35.71428571,  35.71428571,
#                 38.46153846,  41.66666667,  41.66666667,  45.45454545,
#                 45.45454545],
#                [16.66666667,  26.31578947,  23.80952381,  21.73913043,
#                 20.83333333,  21.73913043,  25.,  26.31578947,
#                 23.80952381,  21.73913043,  22.72727273,  26.31578947,
#                 27.77777778,  31.25,  33.33333333,  33.33333333,
#                 35.71428571,  38.46153846,  38.46153846,  38.46153846,
#                 41.66666667,  45.45454545,  45.45454545,  45.45454545,
#                 45.45454545],
#                [17.85714286,  25.,  22.72727273,  20.83333333,
#                 19.23076923,  21.73913043,  26.31578947,  27.77777778,
#                 26.31578947,  23.80952381,  25.,  29.41176471,
#                 27.77777778,  26.31578947,  27.77777778,  29.41176471,
#                 33.33333333,  38.46153846,  38.46153846,  41.66666667,
#                 45.45454545,  50.,  50.,  50.,  50.],
#                [18.51851852,  22.72727273,  20.83333333,  19.23076923,
#                 18.51851852,  21.73913043,  26.31578947,  26.31578947,
#                 23.80952381,  22.72727273,  23.80952381,  27.77777778,
#                 29.41176471,  31.25,  31.25,  33.33333333,
#                 33.33333333,  33.33333333,  35.71428571,  38.46153846,
#                 41.66666667,  41.66666667,  41.66666667,  45.45454545,  50.],
#                [18.51851852,  23.80952381,  21.73913043,  20.,
#                 18.51851852,  22.72727273,  26.31578947,  25.,
#                 23.80952381,  21.73913043,  23.80952381,  27.77777778,
#                 29.41176471,  31.25,  33.33333333,  33.33333333,
#                 35.71428571,  35.71428571,  33.33333333,  33.33333333,
#                 35.71428571,  38.46153846,  41.66666667,  45.45454545,  50.],
#                [17.85714286,  23.80952381,  21.73913043,  20.,
#                 18.51851852,  23.80952381,  25.,  23.80952381,
#                 21.73913043,  20.83333333,  22.72727273,  29.41176471,
#                 31.25,  33.33333333,  35.71428571,  35.71428571,
#                 35.71428571,  35.71428571,  31.25,  27.77777778,
#                 31.25,  35.71428571,  41.66666667,  45.45454545,  50.],
#                [17.85714286,  21.73913043,  19.23076923,  17.85714286,
#                 16.66666667,  23.80952381,  23.80952381,  20.83333333,
#                 19.23076923,  17.85714286,  20.83333333,  31.25,
#                 33.33333333,  38.46153846,  38.46153846,  38.46153846,
#                 38.46153846,  35.71428571,  27.77777778,  25.,
#                 29.41176471,  35.71428571,  45.45454545,  45.45454545,
#                 45.45454545],
#                [17.85714286,  21.73913043,  19.23076923,  17.85714286,
#                 16.66666667,  25.,  22.72727273,  19.23076923,
#                 17.85714286,  16.66666667,  20.,  29.41176471,
#                 33.33333333,  35.71428571,  38.46153846,  38.46153846,
#                 38.46153846,  33.33333333,  25.,  21.73913043,
#                 26.31578947,  31.25,  38.46153846,  45.45454545,  50.],
#                [17.85714286,  20.83333333,  18.51851852,  17.24137931,
#                 16.12903226,  25.,  21.73913043,  17.85714286,
#                 16.66666667,  15.625,  19.23076923,  33.33333333,
#                 35.71428571,  41.66666667,  41.66666667,  38.46153846,
#                 38.46153846,  31.25,  20.83333333,  17.85714286,
#                 21.73913043,  27.77777778,  38.46153846,  41.66666667,
#                 45.45454545],
#                [18.51851852,  20.,  18.51851852,  16.66666667,
#                 15.625,  23.80952381,  19.23076923,  15.15151515,
#                 14.28571429,  13.51351351,  17.24137931,  33.33333333,
#                 38.46153846,  41.66666667,  45.45454545,  45.45454545,
#                 41.66666667,  31.25,  17.85714286,  14.28571429,
#                 17.85714286,  25.,  38.46153846,  41.66666667,  50.],
#                [17.85714286,  18.51851852,  17.24137931,  15.625,
#                 14.70588235,  23.80952381,  17.85714286,  13.51351351,
#                 12.82051282,  12.19512195,  16.12903226,  33.33333333,
#                 35.71428571,  41.66666667,  41.66666667,  45.45454545,
#                 45.45454545,  29.41176471,  14.28571429,  10.63829787,
#                 14.28571429,  21.73913043,  41.66666667,  45.45454545,  50.]])
#

# cs = np.array([[12.5,  11.62790698,  13.15789474,  13.51351351,
#                 14.70588235,  15.625,  15.625,  16.12903226,
#                 16.66666667,  17.24137931,  20.,  20.,
#                 18.51851852,  20.83333333,  21.73913043],
#                [12.82051282,  13.88888889,  14.28571429,  14.70588235,
#                 16.66666667,  17.85714286,  16.66666667,  20.,
#                 20.83333333,  19.23076923,  22.72727273,  22.72727273,
#                 25.,  26.31578947,  27.77777778],
#                [13.88888889,  14.28571429,  15.15151515,  16.12903226,
#                 16.66666667,  17.24137931,  19.23076923,  20.,
#                 21.73913043,  23.80952381,  25.,  25.,
#                 27.77777778,  27.77777778,  35.71428571],
#                [15.15151515,  16.12903226,  16.12903226,  18.51851852,
#                 17.24137931,  19.23076923,  22.72727273,  21.73913043,
#                 23.80952381,  26.31578947,  26.31578947,  29.41176471,
#                 33.33333333,  35.71428571,  35.71428571],
#                [14.70588235,  16.66666667,  16.12903226,  17.85714286,
#                 20.,  21.73913043,  21.73913043,  23.80952381,
#                 25.,  25.,  25.,  31.25,
#                 31.25,  38.46153846,  33.33333333],
#                [15.625,  16.66666667,  17.24137931,  17.85714286,
#                 17.85714286,  20.,  22.72727273,  20.83333333,
#                 25.,  27.77777778,  31.25,  29.41176471,
#                 35.71428571,  35.71428571,  33.33333333],
#                [14.70588235,  15.625,  16.66666667,  19.23076923,
#                 20.,  21.73913043,  21.73913043,  22.72727273,
#                 25.,  29.41176471,  29.41176471,  33.33333333,
#                 31.25,  33.33333333,  38.46153846],
#                [17.24137931,  15.625,  17.85714286,  19.23076923,
#                 19.23076923,  20.83333333,  21.73913043,  22.72727273,
#                 26.31578947,  26.31578947,  29.41176471,  33.33333333,
#                 35.71428571,  35.71428571,  41.66666667],
#                [14.70588235,  16.66666667,  17.24137931,  19.23076923,
#                 19.23076923,  20.83333333,  22.72727273,  26.31578947,
#                 26.31578947,  27.77777778,  35.71428571,  35.71428571,
#                 38.46153846,  38.46153846,  45.45454545],
#                [14.70588235,  17.85714286,  19.23076923,  20.83333333,
#                 20.,  22.72727273,  25.,  25.,
#                 31.25,  29.41176471,  33.33333333,  35.71428571,
#                 33.33333333,  38.46153846,  41.66666667],
#                [15.625,  15.625,  17.85714286,  17.85714286,
#                 21.73913043,  20.83333333,  23.80952381,  26.31578947,
#                 26.31578947,  29.41176471,  33.33333333,  35.71428571,
#                 35.71428571,  38.46153846,  45.45454545],
#                [16.12903226,  16.12903226,  19.23076923,  20.83333333,
#                 19.23076923,  21.73913043,  23.80952381,  25.,
#                 27.77777778,  29.41176471,  35.71428571,  38.46153846,
#                 41.66666667,  41.66666667,  55.55555556],
#                [15.625,  17.85714286,  16.12903226,  18.51851852,
#                 22.72727273,  21.73913043,  22.72727273,  26.31578947,
#                 25.,  31.25,  29.41176471,  33.33333333,
#                 38.46153846,  41.66666667,  45.45454545],
#                [15.15151515,  15.15151515,  17.24137931,  18.51851852,
#                 21.73913043,  25.,  22.72727273,  29.41176471,
#                 27.77777778,  31.25,  33.33333333,  35.71428571,
#                 45.45454545,  41.66666667,  45.45454545],
#                [15.625,  16.66666667,  17.24137931,  16.66666667,
#                 19.23076923,  21.73913043,  21.73913043,  26.31578947,
#                 27.77777778,  31.25,  35.71428571,  38.46153846,
#                 38.46153846,  45.45454545,  38.46153846]])
#
# lack_of_fit = cs

plt.figure(figsize=(12, 9))
im = plt.imshow(lack_of_fit, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(2.8, 3.8, 10, 100), aspect=1. / 90.)
levels = np.arange(np.amin(lack_of_fit), np.amax(
    lack_of_fit), (np.amax(lack_of_fit) - np.amin(lack_of_fit)) / 20)
CS = plt.contour(lack_of_fit, levels,
                 origin='lower',
                 linewidths=2,
                 extent=(2.8, 3.8, 10, 100))
plt.clabel(CS, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=12)
# CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.title('lack_of_fit:scm-tt')
# plt.title('average crack spacing')
plt.flag()

CBI = plt.colorbar(im, shrink=0.8)
plt.xlabel('s_m')
plt.ylabel('m_m')

m_m_arr = np.linspace(10, 100, 100)


def scale(shape):
    lp = 1.
    lc = 1000.
    sig_min = 2.72
    f = (lp / (lp + lc)) ** (1 / shape)
    return sig_min / (f * gamma(1 + 1 / shape))
s_m = scale(m_m_arr)
# print s_m

plt.plot(s_m[s_m < 3.8], m_m_arr[s_m < 3.8], 'w--')


plt.show()
