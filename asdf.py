
import llm_surgeon

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

import pandas as pd
from datasets import load_dataset

import torch

device = utils.get_device()

model = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-3.2-1B-Instruct", device=device, default_padding_side='left')
ls2 = ([[128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,   2696,
             25,   6790,    220,   2366,     18,    198,  15724,   2696,     25,
            220,   1544,   5020,    220,   2366,     19,    271, 128009, 128006,
            882, 128007,    271,  41551,    757,   1268,    311,   1304,    264,
          19692, 128009],
        [128009, 128009, 128000, 128006,   9125, 128007,    271,  38766,   1303,
          33025,   2696,     25,   6790,    220,   2366,     18,    198,  15724,
           2696,     25,    220,   1544,   5020,    220,   2366,     19,    271,
         128009, 128006,    882, 128007,    271,   4438,    656,    358,   3820,
          19837, 128009],
        [128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,   2696,
             25,   6790,    220,   2366,     18,    198,  15724,   2696,     25,
            220,   1544,   5020,    220,   2366,     19,    271, 128009, 128006,
            882, 128007,    271,   3923,    374,    279,   1888,    955,    315,
          45723, 128009]], [[128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,   2696,
             25,   6790,    220,   2366,     18,    198,  15724,   2696,     25,
            220,   1544,   5020,    220,   2366,     19,    271, 128009, 128006,
            882, 128007,    271,  41551,    757,   1268,    311,   1304,    264,
          19692, 128009, 128006,  78191, 128007,    271,  43346,    264,  19692,
            649,    387,    264,   2523,    323,  42093,   3217,     13,   5810,
            596,    264,   6913,  11363,    323,   8641,    311,    636,    499,
           3940,   1473,  46847,   1473,   2520,    264,  11670,  33165,  19692,
           1473,      9,    220,     17,    220,     16,     14,     19,  26446,
            682,  59338,  20415,    198,      9,    220,     16,    220,     16,
             14,     17,  26446,  16109,   7913,  13465,    198,      9,    220,
             17,  93200,  28915,  17138,    198,      9,    220,     16,  42384,
          12290,    198,  62835,    220,     16,  10747,   7120,  62334,  14432,
             11,  90158,    198,      9,    220,     17,   3544,  19335,    198,
              9,    220,     17,  93200,  33165,   8819,    198,      9,    220,
             16,  10747,   4459,  14403,     11,    520,   3130,   9499,    271,
           2520,    264,  18414,  19692,   1473,      9,    220,     17,    220,
             16,     14,     19,  26446,    682,  59338,  20415,    198,      9,
            220,     16,    220,     16,     14,     17,  26446,  16109,   7913,
          13465,    198,      9,    220,     17,  93200,  28915,  17138,    198,
              9,    220,     16,  42384,  12290,    198,  28843,    220,     16,
          10747,   7120,  62334,  14432,     11,  90158,    198,      9,    220,
             17,   3544,  19335,    198,      9,    220,     17,  93200,  33165,
           8819,    198,      9,    220,     16,  10747,   7120,   4589,   6901,
          62888,  17138,    198,      9,    220,     16,  10747,  18768,   1355,
           4589,  18414,  24512,    320,  13099,    696,  56391,   1473,    334,
           3607,    220,     16,     25,  12362,    264,   1221,  44322,  87213,
          57277,     16,     13,   5075,  20559,    701,  24276,    311,    220,
           8652,  59572,    320,  10005,  32037,    570,  13842,    521,   1403,
            220,     24,  25224,    320,   1419,   6358,      8,   4883,  19692,
          64883,    323,   1584,    279,  93650,    449,  87204,   5684,    627,
             17,     13,    763,    264,  11298,  19763,     11,  41759,   3871,
            279,  20415,     11,  13465,     11,  28915],
        [128009, 128009, 128000, 128006,   9125, 128007,    271,  38766,   1303,
          33025,   2696,     25,   6790,    220,   2366,     18,    198,  15724,
           2696,     25,    220,   1544,   5020,    220,   2366,     19,    271,
         128009, 128006,    882, 128007,    271,   4438,    656,    358,   3820,
          19837, 128009, 128006,  78191, 128007,    271,     47,  16671,  19837,
            649,    387,    264,  17104,    323,  42093,   3217,     13,   5810,
            527,   1063,  10631,    311,   1520,    499,   5268,    279,   4832,
           6305,   1473,     16,     13,   3146,  39512,    701,  40190,  96618,
          13538,    499,   1212,  32421,     11,   1935,    264,   1427,   2212,
            701,  13863,     11,  20085,     11,    477,  12818,    311,   1518,
           1148,   4595,    315,  19837,    527,    304,  52554,  18182,     13,
           1115,    690,   3041,    499,    459,   4623,    315,   1148,    596,
           2561,    520,   2204,   3115,    315,    279,   1060,    627,     17,
             13,   3146,  38275,    279,   3280,  96618,  34496,  19837,    527,
           2561,    520,   2204,   3115,    315,    279,   1060,     13,   1789,
           3187,     11,  10683,   1481,    385,  18238,  19837,   1093,  53109,
             77,  19595,   1841,  21852,  81460,    323,  66909,    527,   4832,
            369,   4216,  10683,     11,   1418,   7474,   1481,    385,  18238,
          19837,   1093,   7160,  89770,    323,   1167,   6258,   3557,   4774,
           8369,   9282,    304,    279,   7474,    627,     18,     13,   3146,
          39787,    922,   1933,  96618,  22991,  19837,    430,   2489,    701,
          20247,     11,  13402,     11,    477,  10799,     13,   1442,    499,
           2351,   3411,    369,    264,  24364,  31257,     11,   2980,  21816,
          61741,     11,  39438,    811,     11,    477,   1069,    263,    552,
             13,   1789,    264,  10107,    323,  71414,   1427,     11,   3469,
            369,    294,   2852,    552,     11,   7160,  89770,     11,    477,
          17684,    655,     64,    294,   2852,    552,    627,     19,     13,
           3146,   4061,    279,   4787,  96618,  30379,    279,  19837,    499,
           5268,    527,    304,   1695,   3044,     13,   9925,   1002,    369,
            904,  12195,    315,  59617,     11,  79784,     11,    477,   1023,
          17190,  25384,    430,   2643,   7958,    872,  11341,    627,     20,
             13,   3146,  38053,   1148,    596,   2561,  96618,   4418,    956,
            927,   2320,    875,  19837,    311,    279],
        [128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,   2696,
             25,   6790,    220,   2366,     18,    198,  15724,   2696,     25,
            220,   1544,   5020,    220,   2366,     19,    271, 128009, 128006,
            882, 128007,    271,   3923,    374,    279,   1888,    955,    315,
          45723, 128009, 128006,  78191, 128007,    271,    791,   1888,    955,
            315,  45723,    374,  44122,    323,   3629,  14117,    389,   4443,
          19882,     11,  15481,  94436,     11,    323,   3927,  39555,     13,
           4452,     11,  79795,   1101,   5825,   3072,  13291,    382,   8538,
           5526,  45723,   4595,   2997,   1473,      9,  22591,  17604,  63552,
             25,    362,  58105,   7075,    449,    264,   4382,   3686,  37154,
          10824,    315,  25309,     11,  17604,     11,  71655,     11,  42120,
             11,  38427,     11,    323,   9955,  12843,    389,    264,  93972,
          45921,    627,      9,  86993,  41101,  45723,     25,    362,  43828,
           3072,  16850,  11002,  86993,  41101,  25309,     11,   3629,  35526,
            449,   1579,  13368,  90771,    323,   5016,  17615,  28559,    627,
              9,  31941,   3880,    392,   2788,  45723,     25,    362,  61733,
            477,   7363,   2269,   4588,  45723,  52614,    449,    264,   9235,
          10485,     11,    902,    649,    923,    264,   9257,     11,  94760,
          17615,    323,    264,  95333,  10651,    627,      9,    480,  51802,
          45723,     25,    362,  63174,   3072,  16850,  15193,  14293,     11,
          11782,  90771,     11,    323,   5016,  17615,  21542,     11,   1778,
            439,    490,  13511,   5707,    477,  41452,  20673,    627,      9,
          86523,   4791,  11549,  45723,     25,    362,  16736,     11,  58372,
           3072,  14948,    555,  11670,   3778,  11884,  97467,    271,   2746,
            499,   2351,   3411,    311,  13488,   2204,  45723,   9404,     11,
           2980,   4560,   1473,     16,     13,   3146,     33,  10599,   1557,
          45921,  96618,  25483,    264,  90030,    323,  28682,  10651,    311,
            279,  45723,    627,     17,     13,   3146,     34,  47704,   1534,
          38427,  96618,  27687,    323,  94760,     11,  59811,   1534,  45697,
            923,    264,   8149,    315,  17615,    311,    279,  45723,    627,
             18,     13,   3146,   9470,  17570,    278,  17604,  96618,  11220,
           2641,    311,  39143,    323,   5825,    459,   1358,    315,  17615,
           2671,     11,   1778,    439,    523,  78716]], [[128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,   2696,
             25,   6790,    220,   2366,     18,    198,  15724,   2696,     25,
            220,   1544,   5020,    220,   2366,     19,    271, 128009, 128006,
            882, 128007,    271,  41551,    757,   1268,    311,   1304,    264,
          19692, 128009],
        [128009, 128009, 128000, 128006,   9125, 128007,    271,  38766,   1303,
          33025,   2696,     25,   6790,    220,   2366,     18,    198,  15724,
           2696,     25,    220,   1544,   5020,    220,   2366,     19,    271,
         128009, 128006,    882, 128007,    271,   4438,    656,    358,   3820,
          19837, 128009],
        [128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,   2696,
             25,   6790,    220,   2366,     18,    198,  15724,   2696,     25,
            220,   1544,   5020,    220,   2366,     19,    271, 128009, 128006,
            882, 128007,    271,   3923,    374,    279,   1888,    955,    315,
          45723, 128009]], [[128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,   2696,
             25,   6790,    220,   2366,     18,    198,  15724,   2696,     25,
            220,   1544,   5020,    220,   2366,     19,    271, 128009, 128006,
            882, 128007,    271,  41551,    757,   1268,    311,   1304,    264,
          19692, 128009, 128006,  78191, 128007,    271,  43346,    264,  19692,
            649,    387,    264,  31439,   1920,    422,    499,    617,    279,
           1314,  14293,    323,   1833,    264,  95797,  11363,     13,   5810,
            596,    264,  44899,     11,  19692,  11363,    369,  47950,   1473,
            334,  46847,     25,  57277,   2520,    264,   6913,  33165,  19692,
           1473,     12,    220,     17,    220,     16,     14,     19,  26446,
            682,  59338,  20415,    198,     12,    220,     16,  39020,  28915,
          17138,    198,     12,    220,     16,  39020,  28915,  39962,    198,
             12,    220,     16,  39020,  12290,    198,     12,    220,     16,
          10747,   7120,  62334,  14432,     11,    520,   3130,   9499,    198,
             12,    220,     16,    220,     18,     14,     19,  26446,  16109,
           7913,  13465,    198,     12,    220,     18,   3544,  19335,    520,
           3130,   9499,    198,     12,    220,     17,  39020,  33165,   8819,
            198,     12,    220,     16,  10747,   4459,  14403,     11,    520,
           3130,   9499,    271,   2520,    264,  21147,    323,  18406,  19692,
             11,   2980,   7999,    279,   2768,   1473,     12,    220,     16,
          39020,  28915,  17138,    198,     12,    220,     16,  39020,  28915,
          39962,    198,     12,    220,     16,  39020,  12290,    198,     12,
            220,     16,     14,     17,  10747,   1579,  22867,  33165,   8819,
            198,     12,  12536,     25,  38525,  31049,     11,  18414,  24512,
             11,    477,  32720,  14098,    369,   3779,  17615,    323,  10651,
            271,    334,  56391,     25,  57277,     16,     13,   3146,   4808,
          20559,    701,  24276,  68063,   5075,  20559,    701,  24276,    311,
            220,   8652,  59572,    320,  10005,  32037,    570,   7557,   2771,
            499,    617,    701,  28915,  64883,   5644,    382,     17,     13,
           3146,  59183,   9235,  14293,  68063,  47912,    220,     17,    220,
             16,     14,     19,  26446,    315,  20415,     11,    220,     16,
          39020,    315,  28915,  17138,     11,    220,     16,  39020,    315,
          28915,  39962,     11,    323,    220,     16],
        [128009, 128009, 128000, 128006,   9125, 128007,    271,  38766,   1303,
          33025,   2696,     25,   6790,    220,   2366,     18,    198,  15724,
           2696,     25,    220,   1544,   5020,    220,   2366,     19,    271,
         128009, 128006,    882, 128007,    271,   4438,    656,    358,   3820,
          19837, 128009, 128006,  78191, 128007,    271,  96144,    279,   1314,
          19837,    369,    264,   3230,  13402,    477,  27204,    649,    387,
            264,   2766,  22798,     13,   5810,    527,   1063,   7504,    311,
           1520,    499,   3820,  19837,   1473,     16,     13,   3146,  36438,
            279,  13402,  96618,  13538,    499,   1212,  15389,    369,  19837,
             11,   8417,    279,   7580,    315,    279,  27204,     13,   8886,
            499,   6968,    264,  92713,    369,    264,   3361,   4423,     11,
            264,  13306,     11,    477,    264,  13560,     30,   1115,    690,
           1520,    499,  10491,    389,    279,    955,    315,  19837,    323,
            872,  12472,    627,     17,     13,   3146,   4110,    264,   1933,
          13155,  96618,  22991,    264,   1933,  13155,    477,   7057,    430,
           9248,    279,  13402,     13,   1472,    649,   1101,   2980,    279,
           4443,  19882,    315,    279,  22458,    477,    279,   8146,    315,
            279,   1537,    814,   2351,   7231,    499,    627,     18,     13,
           3146,  38275,    279,   3280,  96618,   7365,  19837,    505,    264,
           2254,  70240,    380,    477,  56226,    430,    374,    304,   3280,
             13,  14598,    278,  19837,   2586,    449,    810,  15193,   7729,
            323,   2731,   4367,    627,     19,     13,   3146,  10596,    369,
           8205,    320,    333,  12096,    304,  20155,  33395,     25,   1442,
            499,   2351,   3339,    264,   3544,  12472,    315,  19837,    369,
            264,   3361,  13402,     11,   2980,  12096,    264,   6651,    315,
          19837,    304,   2204,   8146,    323,   4595,     13,   1115,   1648,
             11,    499,    649,   9526,    449,   2204,   1933,  28559,   2085,
          15061,    279,   6201,    627,     20,     13,   3146,   4061,    279,
          99316,  96618,  22991,  19837,    430,    527,   7878,     11,    628,
           1538,     11,    323,    617,  17832,  75596,    291,  11141,     13,
            435,  15808,    477,  22843,    279,  19837,    311,   6106,    814,
           2351,   7878,    627,    720,    220,    482,   3816,  61741,    198,
            220,    482,   8219,  89770,    198,    220],
        [128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,   2696,
             25,   6790,    220,   2366,     18,    198,  15724,   2696,     25,
            220,   1544,   5020,    220,   2366,     19,    271, 128009, 128006,
            882, 128007,    271,   3923,    374,    279,   1888,    955,    315,
          45723, 128009, 128006,  78191, 128007,    271,    791,   1888,    955,
            315,  45723,    374,  44122,    323,   3629,  14117,    389,   4443,
          19882,     11,  15481,  94436,     11,    323,  58441,  32006,     13,
           5810,    527,   1063,   5526,   4595,    315,  63452,    430,   1274,
           3629,   4774,   1473,     16,     13,   3146,  64431,  45419,  63552,
          96618,    362,  58105,   7075,  40901,    449,  50459,  17604,     11,
          11383,    523,  78716,    477,   3778,    627,     17,     13,   3146,
             33,  23184,  45419,  63552,  96618,    362,  27744,    389,    279,
          11670,     11,   7999,  73624,  41452,    323,  50459,  17604,    627,
             18,     13,   3146,   7530,     12,   1951,  15512,  52871,  96618,
            362,  42415,  45723,    449,   1403,  84167,    552,     11,   3629,
          40901,    449,  17604,     11,  41452,     11,    323,   1023,  90771,
            627,     19,     13,   3146,   1016,    875,   7813,    332,  52871,
          96618,    362,  56153,  45723,    449,    264,  12314,    281,  23758,
             11,   3629,    505,  16763,  79100,  25309,    477,    682,  15502,
            830,  84167,    552,    627,  89561,  30205,  19579,    539,     25,
           7410,  17677,    649,   1304,    264,  19295,  85359,  52871,    810,
          14113,  65915,   5413,   8125,  26676,   3225,  23354,  26564,    331,
          52614,     72,  52517, 110580,   1273,  21099,   1367,  18922,   8783,
          10294,   2520,   3146,   1395,    830,  33218,  98319,   2980,    264,
           1473,     16,     13,   3146,     39,  37069,   1122,  52871,  96618,
          10335,  16618,  14966,  70750,  28109,  41452,     11,  13824,     11,
            323,  78182,    389,    264,  93972,  45921,    627,     17,     13,
           3146,   5028,   1394,    258,      6,  46377,  52871,  96618,  39247,
          23283,  11059,  46377,  19737,     11,  75491,    295,     11,    323,
            523,  78716,  17604,    389,    264,  93972,  45921,    627,     18,
             13,   3146,  26070,  31382,  52871,  96618,    328,  75730,  50059,
             11,  25349,  26128,  17604,     11,  68346,     11,    323,  74314,
          20037,    304,    264,  73624,  16831,   6374]])    

ls3 = []

for ten in ls2:
    ls3.append(model.tokenizer.batch_decode(ten))
    
ls4 = []
for i in range(len(ls3)):
    l = 0
    if i % 2 == 0:
        l = len(ls3[i][0])
    else:
        for prompt in ls3[i]:
            ls4.append(prompt)

print(len(ls4))