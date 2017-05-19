'''
Created on May 27, 2015

@author: Yingxiong
'''
import numpy as np
from matplotlib import pyplot as plt

# 1.0 simulation length=500 ==============================================

# s1 = np.array([0.,  2.79879581,  2.81604237,  2.83072645,  2.90325698,
#                3.04907887,  3.17181513,  3.21513467,  3.48364327,  3.62557364,
#                3.62701846,  3.75458994,  3.95441712,  4.04535365,  4.19953988,
#                4.25545303,  4.78174903,  5.26594958,  5.35621155,  5.36184803,
#                5.50477162,  5.65065944,  5.68646181,  5.70097723,  6.08816109,
#                6.21376971,  6.27052289,  6.72793539,  7.7794066,  9.21364611,
#                9.35148501])
#
# s2 = np.array([0.,  2.59521957,  2.67394236,  2.6938789,  2.73492387,
#                2.77708716,  2.79350818,  3.04512277,  3.2437171,  3.29347964,
#                3.33108606,  3.36805675,  3.75173316,  3.88603591,  3.93195512,
#                4.03794558,  4.25406454,  4.30562671,  4.31362969,  4.31411041,
#                4.44198756,  4.51012351,  5.57231047,  5.74582682,  6.02634835,
#                6.64761008,  6.70968547,  6.92350636,  7.13043723,  7.22202742,
#                7.38427048,  7.52176432,  7.5374009,  7.84985365,  7.93111723,
#                8.62557446,  8.66255616,  8.70269033,  9.16278407,  9.95010728,
#                9.9616922])
#
# s3 = np.array([0.,  2.78489494,  2.84147471,  2.8474771,  2.96430766,
#                3.08170479,  3.15190287,  3.16128908,  3.49117799,  3.56679282,
#                3.60782781,  3.63610328,  3.93164322,  3.98387018,  4.71431028,
#                4.72218391,  4.88336668,  4.97650929,  5.11696449,  5.43703152,
#                5.49237844,  5.54255786,  5.66985788,  6.3259246,  6.39946965,
#                6.77130435,  6.87744307,  9.7102612])
#
# s4 = np.array([0.,  3.43152501,  3.47266538,  3.49724426,  3.64665387,
#                3.77201022,  3.97472342,  4.23095846,  4.67427305,  4.77174431,
#                4.94302159,  5.26332321,  6.05262918,  6.56639927,  6.56871212,
#                6.69761059,  7.58232032,  7.68891009,  8.98405734])
#
# s5 = np.array([0.,  3.2952865,  3.31362066,  3.34495527,  3.37853563,
#                3.56103621,  3.88450046,  3.96515717,  4.36938056,  4.47210019,
#                4.53430594,  4.85931259,  4.87897684,  5.17988757,  6.37568605,
#                6.47611797,  6.86612823,  8.15608222,  8.34198211])

# 1.0 simulation length=500 ==============================================

# 1.5 simulation length=500 ==============================================

s1 = np.array([0.,   4.83468872,   4.86808118,   4.87353398,
               5.13759097,   5.21336996,   5.95975394,   5.97546869,
               5.97997889,   6.14054128,   6.48717463,   6.55114426,
               7.54246081,   7.63226973,   7.80741255,   7.90228004,
               8.0403079,   8.19578932,   8.59854806,   8.60004366,
               9.49929487,  11.14362578,  13.99126453,  16.18001585])

s2 = np.array([0.,   4.1194721,   4.1369008,   4.14887447,
               4.16821979,   4.28759315,   4.54458272,   4.58218043,
               4.940081,   5.21293415,   5.33213481,   5.55408918,
               5.8340442,   5.89671632,   6.06918685,   6.17428573,
               6.19092158,   6.96371973,   6.98298486,   7.41341325,
               7.48847549,   7.77329483,   7.86230364,   8.31734511,
               8.70875774,   8.83486428,   9.16136062,  10.19058874,
               10.28140424,  10.93783946,  11.17270083,  11.22704373,
               11.90069427,  12.21798419,  12.94893454,  13.07373105,
               13.23897961,  16.20283734,  16.82167166])

s3 = np.array([0.,   4.40574867,   4.47989179,   4.48538417,
               4.58100665,   4.73901973,   4.88453597,   5.0052188,
               5.6826593,   5.68995817,   5.77623055,   5.93140492,
               6.04922362,   6.24595041,   6.98714468,   7.21008542,
               7.63274789,   7.68734909,   8.59562263,   8.63546921,
               8.65579022,   8.70947283,   9.92284991,   9.95700439,
               10.48258056,  10.50787574,  11.54218,  14.73350688])

s4 = np.array([0.,   4.08194674,   4.11228441,   4.11776225,
               4.13615396,   4.34091387,   4.5266359,   4.63093963,
               4.93131035,   4.97686069,   5.0006542,   5.23098992,
               5.60659592,   5.62834657,   5.80622866,   5.80757543,
               7.08394843,   7.14781341,   7.30736785,   7.36299586,
               7.49124522,   7.53606815,   7.83717399,   8.05589283,
               8.47965298,   8.90370524,   8.94617937,   9.21335835,
               10.04211828,  11.24285034,  11.66056571,  12.94085707,
               13.39313284,  13.4754943,  13.58268248,  14.77186529])

s5 = np.array([0.,   4.252551,   4.26601877,   4.32983008,
               4.35057841,   4.46396145,   4.6794665,   5.09985213,
               5.13550962,   5.17509803,   5.21582701,   5.48734638,
               5.89610378,   6.32929938,   6.40907252,   6.46684175,
               6.49191782,   6.79751149,   6.85340977,   7.78748162,
               7.97822688,   8.26790814,   8.3227178,   8.80784022,
               11.08456338,  11.27510628,  11.41200005,  11.60445743,
               12.14694053,  12.50078468,  13.08705955,  13.62105972,
               14.07768331,  14.39449167,  16.57813071])

# 1.5 simulation length=500 ==============================================

#=========================================================================
#1% aramis

s1t = np.array([10.00872464,   5.52844059,  11.25707195,
                7.12078782,   7.5985656,   7.12078782,  12.27597275])
s2t = np.array([17.1316769,   9.72990228,  6.97622649,   8.94871857])
s3t = np.array(
    [4.40479296,   7.95527934,  17.34835118,   8.85553078,   6.56429542])
s4t = np.array(
    [15.18919693,   7.0975642,   7.97860545,  11.93039465,   9.31819916])
s5t = np.array([21.13306692,   8.62724404,  18.52434985,
                14.0797884,   7.63826555,   8.38268409,  21.98885654])


s1b = np.array(
    [6.02457869, 22.90243875,  11.25707195,   7.35146734,   8.93051279,  16.18415792])
s2b = np.array(
    [7.10445464,   9.2492387,   6.97622649,  16.79998289, 8.77070183,   8.77070183])
s3b = np.array(
    [4.40479296,  14.90546437,   5.49085621,   4.40479296, 5.97395657,  17.01455883])
s4b = np.array([19.68039936,   6.88058764,  13.92738198,
                7.0975642, 7.0975642,   7.97860545,  17.20897856])
s5b = np.array([22.92756902,   8.62724404,   8.99976055,
                10.78376281, 7.17470687,  12.65419438,   8.23534141,   5.11059319])

#=========================================================================


#=========================================================================
#1.5% aramis
#
# s1t = np.array([16.91681074,  28.24358356,  14.16653556,   7.85852962,
#                 9.26719295,  35.72474096,  15.07669087,  11.44987103,
#                 9.26719295,   9.89696334,   9.58805621])
# s2t = np.array([14.03091356,  18.79005001,  13.87798694,  15.37755049,
#                 17.00879121,  15.98687316,  15.59358517,  12.76183435])
# s3t = np.array([9.05175008,  10.76772668,   9.05175008,   9.05175008,
#                 19.71997199,  11.42490661,  11.21500315,  12.15511804,
#                 7.663163,  18.42709064,   6.37574066,  15.91778204])
# s4t = np.array([10.36301934,  18.73350336,  10.00397938,   7.78785911,
#                 11.22337517,  10.72917466,  11.89792408,  17.01613892,  10.86517866])
# s5t = np.array([18.0952497,    0.60585937,
#                 19.90342057,   0.60585937,  31.91502337, 11.28946654,  11.71460843,   0.60585937,  12.16635002])
#
# s1b = np.array([11.85898609,  19.53944724,
#                 11.44987103,  11.71533759,  14.91634299,  34.56317101,
#                 10.66047702,   7.48661655,  40.40901103])
# s2b = np.array([10.48396872,  25.14811366,  12.08379738,  11.80306615,
#                 11.3517029,  10.27731759,  14.15291021,  32.14101299,  10.74935872])
#
# s3b = np.array([16.02677067,   7.47010424,   8.7946856,  13.85198624,
#                 7.663163,   4.19339853,  26.06350928,   7.47010424,
#                 8.25237909,   7.25361537])
# s4b = np.array([16.51271353,  13.22650031,  37.56411316,   9.14751061,
#                 32.29994302,  18.90466892,   7.54397453,   9.34653494])
#
# s5b = np.array([10.81934181,   0.60585937,  35.98725918,  10.65285666,   0.60585937,   0.6058362,
#                 0.60549606,  15.01184154,   0.60585937])
#=========================================================================


s1e = np.sort(np.hstack((s1t, s1b)))
s2e = np.sort(np.hstack((s2t, s2b)))
s3e = np.sort(np.hstack((s3t, s3b)))
s4e = np.sort(np.hstack((s4t, s4b)))
s5e = np.sort(np.hstack((s5t, s5b)))


s_lst = [s1, s2, s3, s4, s5]
# strength = [14.57, 14.57, 14.57, 14.24, 14.24]
# strength = [21.86,    21.85,    21.86,    21.85,    21.86,   21.856]
# for i in range(5):
#
#     a = s_lst[i]
#
#     nc = np.arange(len(a))
#
#     cs = 500. / (nc + 1e-15)
#
#     cs[cs >= 120] = 120

#     plt.step(np.hstack((a, strength[i])), np.hstack(
#         (cs, cs[-1])), 'k')

# ss = np.array([0.,   3.89826402,   4.06468676,   4.10744562,
#                4.11799825,   4.1696913,   4.22364013,   4.57645342,
#                4.72206516,   4.95646716,   5.07794237,   5.20868716,
#                5.72339081,   5.81641709,   5.9847897,   6.0540798,
#                6.30048436,   6.38307481,   6.71177146,   7.08151078,
#                7.29753334,   7.89361343,   8.06650008,   8.17889837,
#                8.56703009,   8.7724724,   9.22723038,   9.46530432,
#                10.66724949,  10.97134591,  11.06809128,  11.5183378,
#                13.4697307,  14.40192298,  14.91430872,  15.06271271])


ss = np.array([0.,  2.95223772,  3.02028784,  3.03796958,  3.08650869,
               3.30212103,  3.31426359,  3.66214169,  3.81425602,  3.94140874,
               4.1218933,  4.26566265,  4.49377996,  4.94895295,  5.04920735,
               5.05904324,  5.09856605,  5.50542304,  6.25913179,  7.13725771,
               7.18720618,  7.29445442,  7.60365787,  8.02814324,  8.25796847,
               8.52619646,  9.1110345])

nc = np.arange(len(ss))

cs = 500. / (nc + 1e-15)

cs[cs >= 120] = 120
plt.step(np.hstack((ss, 14.55)), np.hstack(
    (cs, cs[-1])), 'k', lw=2)


s_lste = [s1e, s2e, s3e, s4e, s5e]
strength_e = [12.5, 10.38, 11.34, 12.43, 13.40]
# strength_e = [20.66, 21.29, 20.76, 20.63, 19.60]
nc_lst = []
for i in range(5):

    a = s_lste[i]

    nc = np.arange(len(a))
    nc_lst.append(len(a))

    cs = 240. / (nc + 1)

    cs[cs >= 120] = 120

    plt.step(
        np.hstack((0, a / 2, strength_e[i])), np.hstack((120, cs, cs[-1])), color='0.5')

print nc_lst
print np.mean(nc_lst)
print 240. / np.mean(nc_lst)

plt.ylim(0, 140)
# plt.xlabel('sig_c [Mpa')
# plt.ylabel('crack spacing')
# plt.plot([], [], 'k', label='simulation')
# plt.plot([], [], color='0.5', label='experiment')
# plt.legend(loc='best')


plt.show()
