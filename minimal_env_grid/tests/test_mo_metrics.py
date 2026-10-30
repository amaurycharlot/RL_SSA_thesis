"""Block 01 baseline, captured with Python 3.9.13 / NumPy 2.0.2 / SB3 2.7.1.

Frozen expectations are intentional: tests never regenerate their own baseline.
Run with the project's training interpreter using unittest (no pytest required).
"""
from pathlib import Path
import json
import sys
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "agents/train/train_ppo_combined.py").exists():
    PROJECT_ROOT = Path(r"C:\Users\amaur\Documents\TUD Thesis\minimal_env_grid")
sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
from scripts import plot_combined_threshold_tradeoff as metrics

FIELDS = json.loads(r'''
[
  "detections",
  "mean_positional_error",
  "mean_final_positional_error",
  "search_time",
  "mean_observation_frequency",
  "mean_known_observation_frequency"
]
''')

AGGREGATE_FIELDS = json.loads(r'''
[
  "n_seeds",
  "detections_mean",
  "detections_std",
  "positional_error_mean",
  "positional_error_std",
  "final_positional_error_mean",
  "final_positional_error_std",
  "search_time_mean",
  "search_time_std",
  "observation_frequency_mean",
  "observation_frequency_std",
  "known_observation_frequency_mean",
  "known_observation_frequency_std"
]
''')

BASELINE = json.loads(r'''
{
  "combined_10kn_0uk": {
    "per_run": {
      "heuristic_combined_threshold0.002_seed1": [
        0.0,
        0.5560203790664673,
        0.39873969554901123,
        0.0,
        0.16999999999999998,
        0.16999999999999998
      ],
      "heuristic_combined_threshold0.002_seed2": [
        0.0,
        0.6311081647872925,
        0.3031788468360901,
        0.0,
        0.1645,
        0.1645
      ],
      "heuristic_combined_threshold0.002_seed3": [
        0.0,
        0.5885979533195496,
        0.32926028966903687,
        0.0,
        0.1455,
        0.1455
      ],
      "heuristic_combined_threshold0.002_seed4": [
        0.0,
        0.6165657639503479,
        0.3588142693042755,
        0.0,
        0.13799999999999998,
        0.13799999999999998
      ],
      "heuristic_combined_threshold0.002_seed5": [
        0.0,
        0.5917490720748901,
        0.3629528880119324,
        0.0,
        0.14500000000000002,
        0.14500000000000002
      ],
      "heuristic_combined_threshold0.01_seed1": [
        0.0,
        0.5560203790664673,
        0.39873969554901123,
        0.0,
        0.16999999999999998,
        0.16999999999999998
      ],
      "heuristic_combined_threshold0.01_seed2": [
        0.0,
        0.6311081647872925,
        0.3031788468360901,
        0.0,
        0.1645,
        0.1645
      ],
      "heuristic_combined_threshold0.01_seed3": [
        0.0,
        0.5885979533195496,
        0.32926028966903687,
        0.0,
        0.1455,
        0.1455
      ],
      "heuristic_combined_threshold0.01_seed4": [
        0.0,
        0.6165657639503479,
        0.3588142693042755,
        0.0,
        0.13799999999999998,
        0.13799999999999998
      ],
      "heuristic_combined_threshold0.01_seed5": [
        0.0,
        0.5917490720748901,
        0.3629528880119324,
        0.0,
        0.14500000000000002,
        0.14500000000000002
      ],
      "heuristic_combined_threshold0.04_seed1": [
        0.0,
        0.5560203790664673,
        0.39873969554901123,
        0.0,
        0.16999999999999998,
        0.16999999999999998
      ],
      "heuristic_combined_threshold0.04_seed2": [
        0.0,
        0.6311081647872925,
        0.3031788468360901,
        0.0,
        0.1645,
        0.1645
      ],
      "heuristic_combined_threshold0.04_seed3": [
        0.0,
        0.5885979533195496,
        0.32926028966903687,
        0.0,
        0.1455,
        0.1455
      ],
      "heuristic_combined_threshold0.04_seed4": [
        0.0,
        0.6165657639503479,
        0.3588142693042755,
        0.0,
        0.13799999999999998,
        0.13799999999999998
      ],
      "heuristic_combined_threshold0.04_seed5": [
        0.0,
        0.5917490720748901,
        0.3629528880119324,
        0.0,
        0.14500000000000002,
        0.14500000000000002
      ],
      "heuristic_combined_threshold0.2_seed1": [
        0.0,
        0.553390383720398,
        0.3893064856529236,
        33.5,
        0.146,
        0.146
      ],
      "heuristic_combined_threshold0.2_seed2": [
        0.0,
        0.6450836658477783,
        0.3282031714916229,
        38.5,
        0.13699999999999998,
        0.13699999999999998
      ],
      "heuristic_combined_threshold0.2_seed3": [
        0.0,
        0.5976952910423279,
        0.34773844480514526,
        34.0,
        0.12450000000000001,
        0.12450000000000001
      ],
      "heuristic_combined_threshold0.2_seed4": [
        0.0,
        0.6238777041435242,
        0.36967456340789795,
        31.5,
        0.12449999999999999,
        0.12449999999999999
      ],
      "heuristic_combined_threshold0.2_seed5": [
        0.0,
        0.5898635983467102,
        0.4044893682003021,
        34.0,
        0.125,
        0.125
      ],
      "heuristic_combined_threshold1.0_seed1": [
        0.0,
        0.655626654624939,
        0.5472244620323181,
        70.5,
        0.10200000000000001,
        0.10200000000000001
      ],
      "heuristic_combined_threshold1.0_seed2": [
        0.0,
        0.7269889712333679,
        0.625590980052948,
        70.0,
        0.10550000000000001,
        0.10550000000000001
      ],
      "heuristic_combined_threshold1.0_seed3": [
        0.0,
        0.629610538482666,
        0.4000501036643982,
        71.5,
        0.10300000000000001,
        0.10300000000000001
      ],
      "heuristic_combined_threshold1.0_seed4": [
        0.0,
        0.6758758425712585,
        0.3706734776496887,
        71.5,
        0.10400000000000001,
        0.10400000000000001
      ],
      "heuristic_combined_threshold1.0_seed5": [
        0.0,
        0.5925413966178894,
        0.4311439096927643,
        71.0,
        0.1095,
        0.1095
      ],
      "heuristic_combined_threshold5.0_seed1": [
        0.0,
        0.6482657790184021,
        0.4065576195716858,
        88.0,
        0.0915,
        0.0915
      ],
      "heuristic_combined_threshold5.0_seed2": [
        0.0,
        0.6813702583312988,
        0.43378791213035583,
        88.0,
        0.095,
        0.095
      ],
      "heuristic_combined_threshold5.0_seed3": [
        0.0,
        0.6739583611488342,
        0.4324589669704437,
        88.5,
        0.0985,
        0.0985
      ],
      "heuristic_combined_threshold5.0_seed4": [
        0.0,
        0.6673986911773682,
        0.5582322478294373,
        89.5,
        0.095,
        0.095
      ],
      "heuristic_combined_threshold5.0_seed5": [
        0.0,
        0.7368450164794922,
        0.4994433522224426,
        89.5,
        0.1,
        0.1
      ],
      "heuristic_combined_threshold50.0_seed1": [
        0.0,
        0.7304676175117493,
        0.5201495289802551,
        96.5,
        0.0915,
        0.0915
      ],
      "heuristic_combined_threshold50.0_seed2": [
        0.0,
        0.7280817031860352,
        0.4318501949310303,
        97.0,
        0.0925,
        0.0925
      ],
      "heuristic_combined_threshold50.0_seed3": [
        0.0,
        0.7737026810646057,
        0.554601788520813,
        96.0,
        0.0935,
        0.0935
      ],
      "heuristic_combined_threshold50.0_seed4": [
        0.0,
        0.7497602105140686,
        0.44437557458877563,
        96.5,
        0.0935,
        0.0935
      ],
      "heuristic_combined_threshold50.0_seed5": [
        0.0,
        0.7870817184448242,
        0.38041064143180847,
        96.5,
        0.086,
        0.086
      ]
    },
    "input_sha256": {
      "heuristic_combined_threshold0.002_seed1": "d81ce7d63b9647037401de14e3d2150c52d7c7a0aadc7d466203d9554d37a48b",
      "heuristic_combined_threshold0.002_seed2": "d5f15198a2dc8cbf46900c5d51205ae4730d4d36c167ad6afebf5e22d94da78f",
      "heuristic_combined_threshold0.002_seed3": "af04091aa53a22d67639d65b28387a94e5f8a90c4672debbdf9882b4d37e4549",
      "heuristic_combined_threshold0.002_seed4": "5640d542f1adbc45403ae6aaece1dbbaaacc4f2389e0e9b5470108104631e31c",
      "heuristic_combined_threshold0.002_seed5": "452bcc3d69bcbf11683d2bf4b2c87e5f01db6249caf6daad81eaf4c5a4177f93",
      "heuristic_combined_threshold0.01_seed1": "aa0128c840787fa4182f97897229094e1a2c79504b3c6129a7243ae03ee92ab9",
      "heuristic_combined_threshold0.01_seed2": "a7b72b19faf3bbfbbb9288d8ae3bf118f8174f7dba6eaddb524effbf12614062",
      "heuristic_combined_threshold0.01_seed3": "65231d0d5d31c29db4ea617a9f12ebdd5da718de1acf7a44243764f4821aa291",
      "heuristic_combined_threshold0.01_seed4": "1f4344d7c43ef70e0c49dc8ad7859be75f75608e2d71eae064d3d335f17f1c97",
      "heuristic_combined_threshold0.01_seed5": "6a1c2af3dd259e6bdeccf17f3b695930ced861d19088e3b4223ea19aaeea81ad",
      "heuristic_combined_threshold0.04_seed1": "345fc6d9be2827b6875811942a4f95f3f0ec9930575d824ec15ce66181c71971",
      "heuristic_combined_threshold0.04_seed2": "f97cd1ffe053ab54397b842ca2fe04ff01321528d7f707cf86d69e6b0f88fb52",
      "heuristic_combined_threshold0.04_seed3": "f2659b04ced333893c6237d3f2e30a5685d3e5a407e13ceaf297afffea4be75c",
      "heuristic_combined_threshold0.04_seed4": "632e315e988d568488d739c0da44b891c2e85a987c58ab7a9f48c7aff8b81835",
      "heuristic_combined_threshold0.04_seed5": "ce09fde87f3c86a5051aa32287e708ad61d72257e221f6a089c1c6d928f164df",
      "heuristic_combined_threshold0.2_seed1": "314e8616eba9359a107916b5e0fb2486c9ca6395b54c660b4fe3382966633234",
      "heuristic_combined_threshold0.2_seed2": "faefc20bab6b13706a309d97735c732a0dd72c80b6b81a8da6bd35c10607c21e",
      "heuristic_combined_threshold0.2_seed3": "8e35131ce11229cfc03c0b0732f7cd7edc97456ffee456f700e9d88562caf5fe",
      "heuristic_combined_threshold0.2_seed4": "c64dbdf45c6578109e791856d8cd0cb362219ee9c22f079ed5ffc4d6d7834fd2",
      "heuristic_combined_threshold0.2_seed5": "6f998a15093af1ad849afde5a0e6a0515a8b8bb224e1c7af1d2cf4b22eb6f2b7",
      "heuristic_combined_threshold1.0_seed1": "aab61e17bcd1245fc8c7dc04b47246d109a7c2bf4b08b16460969cddae0c0f51",
      "heuristic_combined_threshold1.0_seed2": "253c332087adc7bb09e1500d88c6c0be85571b27c7827320679ebd8dd41e4074",
      "heuristic_combined_threshold1.0_seed3": "252a0d753cb2b97bb808e661edd9ea82d0353186ae52d345ddb501288823a863",
      "heuristic_combined_threshold1.0_seed4": "aa3e8ad339e04782e48a6083bac07d69852010769fcf9528da9833ec89601c22",
      "heuristic_combined_threshold1.0_seed5": "d81ab6914861bcb0b647a42c1b96c30b42ecdadb6a7d0bfe44c891e4840c4264",
      "heuristic_combined_threshold5.0_seed1": "34a654d18ce90afb68f2d8a13a33ddd74c6488d73b65de82e00093f55875d949",
      "heuristic_combined_threshold5.0_seed2": "a7ec30704896e27885fc2ddfc578474d42ac53f2d8dfc26f09f6c4eb71a9a0cf",
      "heuristic_combined_threshold5.0_seed3": "15deb695be7288d67b464802b1d18d4028b1619d48061cfaa389d860dd810e66",
      "heuristic_combined_threshold5.0_seed4": "e44c4d1cc12bfe2fcea2d442aee6a60edaa75ca7270a4a034916e3095fb249e1",
      "heuristic_combined_threshold5.0_seed5": "ac45ae10f0dcb2c9e48af7af751d6c2bf256bcd216be7560b4e43c7f5b4bdbf2",
      "heuristic_combined_threshold50.0_seed1": "e09c45451b108c56af944b9c362b3a493ac8d19034ce512c2729577980cdd502",
      "heuristic_combined_threshold50.0_seed2": "cbe9e62bb5929ee29b57fa69ae3bd5eab2c289a44877ede2b214c137147d5c60",
      "heuristic_combined_threshold50.0_seed3": "707eec0d839445be78b57c1ddf023389b50d9b57e7d9791632f8091e3242f514",
      "heuristic_combined_threshold50.0_seed4": "0c02e6a02468d76b13ebfb3ff87d51a926fd813be83d4143bf5821c619197a52",
      "heuristic_combined_threshold50.0_seed5": "f99d8baf5044165ae50045594152a17d872b1917b82c5c6df20182b621527ac0"
    },
    "aggregated": {
      "0.002": [
        5,
        0.0,
        0.0,
        0.5968082666397094,
        0.028824894536068082,
        0.3505891978740692,
        0.03619960901365093,
        0.0,
        0.0,
        0.1526,
        0.013835642377569608,
        0.1526,
        0.013835642377569608
      ],
      "0.01": [
        5,
        0.0,
        0.0,
        0.5968082666397094,
        0.028824894536068082,
        0.3505891978740692,
        0.03619960901365093,
        0.0,
        0.0,
        0.1526,
        0.013835642377569608,
        0.1526,
        0.013835642377569608
      ],
      "0.04": [
        5,
        0.0,
        0.0,
        0.5968082666397094,
        0.028824894536068082,
        0.3505891978740692,
        0.03619960901365093,
        0.0,
        0.0,
        0.1526,
        0.013835642377569608,
        0.1526,
        0.013835642377569608
      ],
      "0.2": [
        5,
        0.0,
        0.0,
        0.6019821286201477,
        0.03486958618801597,
        0.36788240671157835,
        0.030750989912824104,
        34.3,
        2.5641762809916173,
        0.1314,
        0.009755767524905454,
        0.1314,
        0.009755767524905454
      ],
      "1.0": [
        5,
        0.0,
        0.0,
        0.6561286807060241,
        0.050393297595565295,
        0.4749365866184235,
        0.10760908763476275,
        70.9,
        0.6519202405202649,
        0.1048,
        0.0029283100928692613,
        0.1048,
        0.0029283100928692613
      ],
      "5.0": [
        5,
        0.0,
        0.0,
        0.6815676212310791,
        0.03325395273568138,
        0.466096019744873,
        0.06188944849726801,
        88.7,
        0.758287544405155,
        0.096,
        0.0033354160160315866,
        0.096,
        0.0033354160160315866
      ],
      "50.0": [
        5,
        0.0,
        0.0,
        0.7538187861442566,
        0.026105260457826084,
        0.4662775456905365,
        0.07025209342173971,
        96.5,
        0.3535533905932738,
        0.0914,
        0.0031304951684997086,
        0.0914,
        0.0031304951684997086
      ]
    }
  },
  "combined_10kn_10uk": {
    "per_run": {
      "heuristic_combined_threshold0.002_seed1": [
        10.0,
        0.6954863667488098,
        0.41997671127319336,
        0.0,
        0.12058474379138037,
        0.1145
      ],
      "heuristic_combined_threshold0.002_seed2": [
        10.0,
        0.7398333549499512,
        0.36592185497283936,
        0.0,
        0.11539021289365614,
        0.1115
      ],
      "heuristic_combined_threshold0.002_seed3": [
        10.0,
        0.7576505541801453,
        0.3564862608909607,
        0.0,
        0.11845881088491621,
        0.11750000000000001
      ],
      "heuristic_combined_threshold0.002_seed4": [
        10.0,
        0.7134180068969727,
        0.3274948298931122,
        0.0,
        0.12218402414094613,
        0.11699999999999999
      ],
      "heuristic_combined_threshold0.002_seed5": [
        10.0,
        0.7181126475334167,
        0.3668519854545593,
        0.0,
        0.1154024407690655,
        0.11200000000000002
      ],
      "heuristic_combined_threshold0.01_seed1": [
        10.0,
        0.6954863667488098,
        0.41997671127319336,
        0.0,
        0.12058474379138037,
        0.1145
      ],
      "heuristic_combined_threshold0.01_seed2": [
        10.0,
        0.7398333549499512,
        0.36592185497283936,
        0.0,
        0.11539021289365614,
        0.1115
      ],
      "heuristic_combined_threshold0.01_seed3": [
        10.0,
        0.7576505541801453,
        0.3564862608909607,
        0.0,
        0.11845881088491621,
        0.11750000000000001
      ],
      "heuristic_combined_threshold0.01_seed4": [
        10.0,
        0.7134180068969727,
        0.3274948298931122,
        0.0,
        0.12218402414094613,
        0.11699999999999999
      ],
      "heuristic_combined_threshold0.01_seed5": [
        10.0,
        0.7181126475334167,
        0.3668519854545593,
        0.0,
        0.1154024407690655,
        0.11200000000000002
      ],
      "heuristic_combined_threshold0.04_seed1": [
        10.0,
        0.6954863667488098,
        0.41997671127319336,
        0.0,
        0.12058474379138037,
        0.1145
      ],
      "heuristic_combined_threshold0.04_seed2": [
        10.0,
        0.7398333549499512,
        0.36592185497283936,
        0.0,
        0.11539021289365614,
        0.1115
      ],
      "heuristic_combined_threshold0.04_seed3": [
        10.0,
        0.7576505541801453,
        0.3564862608909607,
        0.0,
        0.11845881088491621,
        0.11750000000000001
      ],
      "heuristic_combined_threshold0.04_seed4": [
        10.0,
        0.7134180068969727,
        0.3274948298931122,
        0.0,
        0.12218402414094613,
        0.11699999999999999
      ],
      "heuristic_combined_threshold0.04_seed5": [
        10.0,
        0.7181126475334167,
        0.3668519854545593,
        0.0,
        0.1154024407690655,
        0.11200000000000002
      ],
      "heuristic_combined_threshold0.2_seed1": [
        10.0,
        0.6954863667488098,
        0.41997671127319336,
        0.0,
        0.12058474379138037,
        0.1145
      ],
      "heuristic_combined_threshold0.2_seed2": [
        10.0,
        0.7398333549499512,
        0.36592185497283936,
        0.0,
        0.11539021289365614,
        0.1115
      ],
      "heuristic_combined_threshold0.2_seed3": [
        10.0,
        0.7570816874504089,
        0.32475075125694275,
        0.5,
        0.11794974352740326,
        0.11699999999999999
      ],
      "heuristic_combined_threshold0.2_seed4": [
        10.0,
        0.7134180068969727,
        0.3274948298931122,
        0.0,
        0.12218402414094613,
        0.11699999999999999
      ],
      "heuristic_combined_threshold0.2_seed5": [
        10.0,
        0.7181126475334167,
        0.3668519854545593,
        0.0,
        0.1154024407690655,
        0.11200000000000002
      ],
      "heuristic_combined_threshold1.0_seed1": [
        10.0,
        0.6979469060897827,
        0.43342000246047974,
        60.0,
        0.10250423994971034,
        0.094
      ],
      "heuristic_combined_threshold1.0_seed2": [
        10.0,
        0.7774596214294434,
        0.43561068177223206,
        60.0,
        0.10265541529145208,
        0.097
      ],
      "heuristic_combined_threshold1.0_seed3": [
        10.0,
        0.7766871452331543,
        0.47992080450057983,
        66.0,
        0.09780194500684743,
        0.094
      ],
      "heuristic_combined_threshold1.0_seed4": [
        10.0,
        0.7429631352424622,
        0.4409680962562561,
        60.5,
        0.10395493743037673,
        0.1015
      ],
      "heuristic_combined_threshold1.0_seed5": [
        10.0,
        0.7025373578071594,
        0.3204525411128998,
        63.5,
        0.10355615552210692,
        0.099
      ],
      "heuristic_combined_threshold5.0_seed1": [
        10.0,
        0.8180714249610901,
        0.4981216788291931,
        81.0,
        0.09432197381789029,
        0.089
      ],
      "heuristic_combined_threshold5.0_seed2": [
        10.0,
        0.8249434232711792,
        0.5156112909317017,
        81.5,
        0.09641482801272386,
        0.093
      ],
      "heuristic_combined_threshold5.0_seed3": [
        10.0,
        0.8300595283508301,
        0.49324721097946167,
        82.0,
        0.0967514226667339,
        0.094
      ],
      "heuristic_combined_threshold5.0_seed4": [
        10.0,
        0.7717389464378357,
        0.4224678874015808,
        80.5,
        0.09846440208788323,
        0.098
      ],
      "heuristic_combined_threshold5.0_seed5": [
        10.0,
        0.8115357160568237,
        0.4169600009918213,
        80.0,
        0.09478324133954882,
        0.0945
      ],
      "heuristic_combined_threshold50.0_seed1": [
        10.0,
        0.7861015200614929,
        0.4135800004005432,
        92.5,
        0.09521821057572745,
        0.095
      ],
      "heuristic_combined_threshold50.0_seed2": [
        10.0,
        0.8349326848983765,
        0.3866393566131592,
        93.5,
        0.094316919867386,
        0.097
      ],
      "heuristic_combined_threshold50.0_seed3": [
        10.0,
        0.8746841549873352,
        0.47326111793518066,
        94.0,
        0.09150759843646175,
        0.0915
      ],
      "heuristic_combined_threshold50.0_seed4": [
        10.0,
        0.8493608236312866,
        0.40890222787857056,
        93.0,
        0.09415855240265025,
        0.0925
      ],
      "heuristic_combined_threshold50.0_seed5": [
        10.0,
        0.8262163996696472,
        0.4688727855682373,
        93.0,
        0.09245650773297671,
        0.0925
      ]
    },
    "input_sha256": {
      "heuristic_combined_threshold0.002_seed1": "bd104f4fe9fb7d9ee565729588c7f42d040646ace8117b2fcdcb3b2d38303fd5",
      "heuristic_combined_threshold0.002_seed2": "294ea88a75baf6ed48c71c531440307dae0c141485ad1d171c81ca0dbc13b7dc",
      "heuristic_combined_threshold0.002_seed3": "be0286d68dd4d755c6f4133f9add5e2d340ebf5d736ba885805dd866fc8dd313",
      "heuristic_combined_threshold0.002_seed4": "fb08ef7b8d92800623ad40f46aac5863afbfe2865162bde58fa48d1e69eb5216",
      "heuristic_combined_threshold0.002_seed5": "b29946b914caf89371c93093f3b836c3f9d3991dbdf48ac8638c666643344a66",
      "heuristic_combined_threshold0.01_seed1": "bd104f4fe9fb7d9ee565729588c7f42d040646ace8117b2fcdcb3b2d38303fd5",
      "heuristic_combined_threshold0.01_seed2": "294ea88a75baf6ed48c71c531440307dae0c141485ad1d171c81ca0dbc13b7dc",
      "heuristic_combined_threshold0.01_seed3": "be0286d68dd4d755c6f4133f9add5e2d340ebf5d736ba885805dd866fc8dd313",
      "heuristic_combined_threshold0.01_seed4": "fb08ef7b8d92800623ad40f46aac5863afbfe2865162bde58fa48d1e69eb5216",
      "heuristic_combined_threshold0.01_seed5": "b29946b914caf89371c93093f3b836c3f9d3991dbdf48ac8638c666643344a66",
      "heuristic_combined_threshold0.04_seed1": "bd104f4fe9fb7d9ee565729588c7f42d040646ace8117b2fcdcb3b2d38303fd5",
      "heuristic_combined_threshold0.04_seed2": "294ea88a75baf6ed48c71c531440307dae0c141485ad1d171c81ca0dbc13b7dc",
      "heuristic_combined_threshold0.04_seed3": "be0286d68dd4d755c6f4133f9add5e2d340ebf5d736ba885805dd866fc8dd313",
      "heuristic_combined_threshold0.04_seed4": "fb08ef7b8d92800623ad40f46aac5863afbfe2865162bde58fa48d1e69eb5216",
      "heuristic_combined_threshold0.04_seed5": "b29946b914caf89371c93093f3b836c3f9d3991dbdf48ac8638c666643344a66",
      "heuristic_combined_threshold0.2_seed1": "bd104f4fe9fb7d9ee565729588c7f42d040646ace8117b2fcdcb3b2d38303fd5",
      "heuristic_combined_threshold0.2_seed2": "294ea88a75baf6ed48c71c531440307dae0c141485ad1d171c81ca0dbc13b7dc",
      "heuristic_combined_threshold0.2_seed3": "bc19a28c0ab27796ef7c0f716d7abf2fff7f23335bd14677631021f27a694cde",
      "heuristic_combined_threshold0.2_seed4": "fb08ef7b8d92800623ad40f46aac5863afbfe2865162bde58fa48d1e69eb5216",
      "heuristic_combined_threshold0.2_seed5": "b29946b914caf89371c93093f3b836c3f9d3991dbdf48ac8638c666643344a66",
      "heuristic_combined_threshold1.0_seed1": "6d368c962f09faaf622937fc4c93a8778bfcc95bc997a8a3cecbb438a506b934",
      "heuristic_combined_threshold1.0_seed2": "8b00cfe1111b2a8ba2c62c919c0f4b00a91da61966e85b7fc259f4159aa8e09e",
      "heuristic_combined_threshold1.0_seed3": "a1e7280c756f681bae1a5cc475d27a4a47f4d2f7b7151c07f98f8a57a24e7d14",
      "heuristic_combined_threshold1.0_seed4": "2cc73fc3fcc5250559f6b5666245e9746dc19d9a9304d241441e041c7729e855",
      "heuristic_combined_threshold1.0_seed5": "6c89a11cb8d5b70324154ba8229009e782fc1834757d649c76ea3fed69b1d331",
      "heuristic_combined_threshold5.0_seed1": "eec5a26cadda97ca6e218ccfbf73aca3216011934cf411ca9b49a631e2c258a5",
      "heuristic_combined_threshold5.0_seed2": "12692321c6f21d4fc8369e4fcc353f53ef1e5e9a1b8962b73adddf7a345ae679",
      "heuristic_combined_threshold5.0_seed3": "59928c5d1ef6d9710fb3424340f34b07773259c401da0ce14ebfd2419ea67383",
      "heuristic_combined_threshold5.0_seed4": "d507699ca9aeed539a15fc946d06678e170e5a9a20b78b214c1eee7f17a9c993",
      "heuristic_combined_threshold5.0_seed5": "bc3afbfd51481233bda9e3bb3f2dc7025f5943a330c47ed511ad9f44b3fa9a08",
      "heuristic_combined_threshold50.0_seed1": "84a0bf2e3d9a733c250de6b9f02f6538ea2e3ebe87504369a8cf776e7b1788bf",
      "heuristic_combined_threshold50.0_seed2": "485253193113845cfeb6c8c8917d4d98bd29819c1914629756868ac07aa92184",
      "heuristic_combined_threshold50.0_seed3": "af80ea02b6ec6b8e5649d9f1df02ada6c9a24789bc2a0198d1d1fb0e1f47d224",
      "heuristic_combined_threshold50.0_seed4": "628358496dfe4f31f0b95dacfcd2f79c757f5b68e665ec8580424904e9958db5",
      "heuristic_combined_threshold50.0_seed5": "477a7167aa867c79896a06dd2f8ad48fccf0369b419a584cdb23ba01eabae9cb"
    },
    "aggregated": {
      "0.002": [
        5,
        10.0,
        0.0,
        0.7249001860618591,
        0.02417989882709339,
        0.367346328496933,
        0.033460101988848946,
        0.0,
        0.0,
        0.11840404649599287,
        0.003047109316682291,
        0.1145,
        0.002761340254296811
      ],
      "0.01": [
        5,
        10.0,
        0.0,
        0.7249001860618591,
        0.02417989882709339,
        0.367346328496933,
        0.033460101988848946,
        0.0,
        0.0,
        0.11840404649599287,
        0.003047109316682291,
        0.1145,
        0.002761340254296811
      ],
      "0.04": [
        5,
        10.0,
        0.0,
        0.7249001860618591,
        0.02417989882709339,
        0.367346328496933,
        0.033460101988848946,
        0.0,
        0.0,
        0.11840404649599287,
        0.003047109316682291,
        0.1145,
        0.002761340254296811
      ],
      "0.2": [
        5,
        10.0,
        0.0,
        0.7247864127159118,
        0.023987849669262918,
        0.3609992265701294,
        0.03864365186570819,
        0.1,
        0.22360679774997902,
        0.11830223302449028,
        0.003053320441966673,
        0.11439999999999999,
        0.002631539473388146
      ],
      "1.0": [
        5,
        10.0,
        0.0,
        0.7395188331604003,
        0.038499041200118184,
        0.4220744252204895,
        0.05987932467668495,
        62.0,
        2.669269563007828,
        0.1020945386400987,
        0.002475187136864587,
        0.0971,
        0.0032480763537823458
      ],
      "5.0": [
        5,
        10.0,
        0.0,
        0.8112698078155518,
        0.023178886198832408,
        0.4692816138267517,
        0.046047840145708924,
        81.0,
        0.7905694150420949,
        0.09614717358495602,
        0.0016581593062843627,
        0.0937,
        0.0032326459750489254
      ],
      "50.0": [
        5,
        10.0,
        0.0,
        0.8342591166496277,
        0.032582734607903646,
        0.43025109767913816,
        0.0386561130446018,
        93.2,
        0.570087712549569,
        0.09353155780304043,
        0.001508878519494809,
        0.0937,
        0.0022527760652137642
      ]
    }
  }
}
''')


class ExistingRolloutMetricsBaseline(unittest.TestCase):
    # Params: self, dataset; returns: loaded.
    def load_frozen_runs(self, dataset):
        root = PROJECT_ROOT / "rollouts" / dataset
        if not root.exists():
            self.skipTest("External baseline rollouts are unavailable: " + str(root))
        found = {run.run_dir.name: run for run in metrics.discover_runs(root)}
        expected = BASELINE[dataset]
        loaded = []
        for name, values in expected["per_run"].items():
            with self.subTest(dataset=dataset, run=name):
                self.assertIn(name, found, "A captured rollout is missing")
                run = found[name]
                digest = hashlib.sha256()
                for path in sorted(run.run_dir.iterdir()):
                    if path.suffix == ".npy" or path.name == "meta.json":
                        digest.update(path.name.encode())
                        digest.update(b"\0")
                        digest.update(path.read_bytes())
                self.assertEqual(digest.hexdigest(), expected["input_sha256"][name],
                                 "Rollout inputs changed; review the baseline, do not regenerate silently")
                result = metrics.load_run_metrics(run, 100.0, "catalogue")
                np.testing.assert_allclose([getattr(result, f) for f in FIELDS], values,
                                           rtol=1e-7, atol=1e-9, equal_nan=True)
                loaded.append(result)
        return loaded

    # Params: self, dataset; returns: None.
    def check_dataset(self, dataset):
        loaded = self.load_frozen_runs(dataset)
        self.assertEqual(len(loaded), 35)
        rows = metrics.aggregate_by_threshold(loaded)
        self.assertEqual(len(rows), 7)
        for row in rows:
            with self.subTest(dataset=dataset, threshold=row.threshold):
                np.testing.assert_allclose(
                    [getattr(row, field) for field in AGGREGATE_FIELDS],
                    BASELINE[dataset]["aggregated"][str(row.threshold)],
                    rtol=1e-7, atol=1e-9, equal_nan=True)

    # Params: self; returns: None.
    def test_10_known_0_unknown_metrics_and_aggregation(self):
        self.check_dataset("combined_10kn_0uk")

    # Params: self; returns: None.
    def test_10_known_10_unknown_metrics_and_aggregation(self):
        self.check_dataset("combined_10kn_10uk")


from agents.mo import metrics as shared_metrics


class SharedPositionMetrics(unittest.TestCase):
    # Params: self; returns: None.
    def test_catalogue_origin_absence_and_nonfinite_covariance(self):
        est = np.zeros((1, 6, 9), np.float32)
        est[0, 0, 4:6] = [1, 2]  # valid estimate at origin
        est[0, 2, :2] = [7, 8]    # position alone is not a track
        est[0, 3, 4:6] = [0, 2]
        est[0, 4, 4:6] = [1, np.nan]
        est[0, 5, 4:6] = [np.inf, 1]
        np.testing.assert_array_equal(shared_metrics.catalogue_mask(est),
                                      [[True, False, False, True, False, False]])
        np.testing.assert_array_equal(shared_metrics.catalogue_mask(est),
                                      metrics.catalogue_mask(est))
        self.assertEqual(shared_metrics.mean_covariance_trace(est, 'catalogue', 1), 2.5)

    # Params: self; returns: None.
    def test_legacy_position_only_fallback(self):
        est = np.array([[[0, 0], [1, 0], [np.nan, 1]]], np.float32)
        np.testing.assert_array_equal(shared_metrics.catalogue_mask(est),
                                      [[False, True, False]])

    # Params: self; returns: None.
    def test_wrapping_and_half_world_convention(self):
        estimate = np.array([[99., 1.], [50., 150.], [201., -201.]])
        truth = np.array([[1., 99.], [0., 0.], [0., 0.]])
        np.testing.assert_array_equal(shared_metrics.wrapped_difference(estimate, truth, 100),
                                      [[-2, 2], [-50, -50], [1, -1]])

    # Params: self; returns: None.
    def test_pair_weighting_final_only_and_initial_target_subset(self):
        truth = np.zeros((2, 3, 4), np.float32)
        est = np.zeros((2, 3, 9), np.float32)
        est[:, 0, :2] = [[3, 4], [0, 2]]
        est[:, 0, 4:6] = [[1, 2], [2, 3]]
        est[1, 1, :2] = [99, 0]
        est[1, 1, 4:6] = [3, 4]
        est[:, 2, :2] = 30  # absent despite a nonzero estimate
        meta = {'known_targets': 1}
        args = (truth, est, 100, 'catalogue', meta)
        self.assertEqual(shared_metrics.compute_mean_positional_error(*args), float(np.float32(8 / 3)))
        self.assertEqual(shared_metrics.compute_mean_final_positional_error(*args), 1.5)
        self.assertEqual(shared_metrics.compute_mean_positional_error(truth, est, 100, 'initial', meta), 3.5)
        self.assertEqual(shared_metrics.compute_mean_final_positional_error(truth, est, 100, 'initial', meta), 2)
        self.assertEqual(shared_metrics.mean_covariance_trace(est, 'catalogue', 1), 5)
        self.assertEqual(shared_metrics.mean_covariance_trace(est, 'initial', 1), 4)
        # Frozen trace arrays are accepted without mutation.
        truth.setflags(write=False)
        est.setflags(write=False)
        self.assertEqual(shared_metrics.compute_mean_final_positional_error(*args), 1.5)
        self.assertEqual(shared_metrics.mean_covariance_trace(est, 'catalogue', 1), 5)

    # Params: self; returns: None.
    def test_no_valid_pairs_and_empty_final_are_explicit(self):
        truth = np.zeros((2, 1, 4), np.float32)
        est = np.zeros((2, 1, 9), np.float32)
        for function in (shared_metrics.compute_mean_positional_error,
                         shared_metrics.compute_mean_final_positional_error):
            with self.assertRaisesRegex(ValueError, 'No valid catalogue'):
                function(truth, est, 100, 'catalogue', {})
        self.assertIsNone(shared_metrics.mean_covariance_trace(est, 'catalogue', 1))
        self.assertIsNone(shared_metrics.mean_covariance_trace(est[:0], 'catalogue', 1))
        est[:, :, 4:6] = 1
        self.assertIsNone(shared_metrics.mean_covariance_trace(est, 'initial', 0))
        with self.assertRaisesRegex(ValueError, 'no final timestep'):
            shared_metrics.compute_mean_final_positional_error(truth[:0], est[:0], 100, 'catalogue', {})
        with self.assertRaisesRegex(ValueError, 'align'):
            shared_metrics.compute_mean_final_positional_error(truth[:1], est, 100, 'catalogue', {})

    # Params: self; returns: None.
    def test_nonfinite_positions_are_excluded_from_error_only(self):
        truth = np.zeros((1, 2, 4), np.float32)
        est = np.zeros((1, 2, 9), np.float32)
        est[:, :, 4:6] = 1
        est[0, 0, :2] = [3, 4]
        est[0, 1, 0] = np.nan
        self.assertEqual(shared_metrics.compute_mean_positional_error(truth, est, 100, 'catalogue', {}), 5)
        self.assertEqual(shared_metrics.mean_covariance_trace(est, 'catalogue', 0), 2)

    # Params: self; returns: None.
    def test_initial_catalogue_metadata_aliases(self):
        self.assertIsNone(shared_metrics.initial_catalogue_size({}))
        for key in shared_metrics.INITIAL_CATALOGUE_META_KEYS:
            self.assertEqual(shared_metrics.initial_catalogue_size({key: 2}), 2)
            self.assertEqual(shared_metrics.initial_catalogue_size({key: [0, 1]}), 2)
        with self.assertRaises(KeyError):
            shared_metrics.compute_mean_positional_error(np.zeros((1, 1, 4)),
                np.zeros((1, 1, 9)), 100, 'initial', {})

    # Params: self; returns: None.
    def test_covariance_input_contracts(self):
        est = np.ones((1, 2, 9), np.float32)
        for array, target_set, count, error in (
                (est[:, :, :6], 'catalogue', 1, ValueError),
                (est[0], 'catalogue', 1, ValueError),
                (est.astype(int), 'catalogue', 1, TypeError),
                (est, 'all', 1, ValueError), (est, 'initial', -1, ValueError),
                (est, 'initial', 3, ValueError), (est, 'initial', True, TypeError),
                (est, 'initial', 1.5, TypeError)):
            with self.subTest(target_set=target_set, count=count, shape=array.shape):
                with self.assertRaises(error):
                    shared_metrics.mean_covariance_trace(array, target_set, count)

    # Params: self; returns: None.
    def test_shared_module_has_no_plotting_or_training_imports(self):
        import subprocess
        code = ('import sys; import agents.mo.metrics; '
                'assert not any(n == "matplotlib" or n.startswith("matplotlib.") '
                'or n == "stable_baselines3" or n.startswith("stable_baselines3.") '
                'or n == "scripts" or n.startswith("scripts.") for n in sys.modules)')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=str(PROJECT_ROOT), check=True)

    # Params: self; returns: None.
    def test_shared_position_metrics_match_all_captured_rollouts(self):
        for dataset, expected in BASELINE.items():
            root = PROJECT_ROOT / 'rollouts' / dataset
            if not root.exists():
                self.skipTest('External baseline rollouts are unavailable: ' + str(root))
            for name, values in expected['per_run'].items():
                with self.subTest(dataset=dataset, run=name):
                    path = root / name
                    truth = np.load(path / 'true_states.npy', allow_pickle=False)
                    est = np.load(path / 'est_states.npy', allow_pickle=False)
                    meta = metrics.load_meta(path)
                    self.assertEqual(shared_metrics.compute_mean_positional_error(
                        truth, est, 100, 'catalogue', meta), values[1])
                    self.assertEqual(shared_metrics.compute_mean_final_positional_error(
                        truth, est, 100, 'catalogue', meta), values[2])
                    np.testing.assert_array_equal(shared_metrics.catalogue_mask(est), metrics.catalogue_mask(est))
                    count = shared_metrics.initial_catalogue_size(meta)
                    for target_set in ('catalogue', 'initial'):
                        self.assertEqual(shared_metrics.compute_mean_positional_error(
                            truth, est, 100, target_set, meta), metrics.compute_mean_positional_error(
                            truth, est, 100, target_set, meta))
                        self.assertEqual(shared_metrics.compute_mean_final_positional_error(
                            truth, est, 100, target_set, meta), metrics.compute_mean_final_positional_error(
                            truth, est, 100, target_set, meta))
                        selected = est if target_set == 'catalogue' else est[:, :count]
                        mask = metrics.catalogue_mask(selected)
                        expected_cov = float(np.mean((selected[:, :, 4] + selected[:, :, 5])[mask]))
                        self.assertEqual(shared_metrics.mean_covariance_trace(est, target_set, count), expected_cov)


from agents.mo.types import EpisodeTrace, EnvConfig, ExperimentConfig, TargetObservationMetrics


# Params: known, observed, valid; returns: EpisodeTrace.
def episode_trace(known=1, observed=None, valid=True):
    observed = (np.array([[True, True, False], [False, True, False], [True, False, True]])
                if observed is None else np.asarray(observed, dtype=bool))
    d, n = observed.shape
    est = np.zeros((d, n, 9), np.float32)
    if valid:
        est[:, :known, 4:6] = 1
        for target in range(known, n):
            ids = np.flatnonzero(observed[:, target])
            if len(ids):
                est[ids[0]:, target, 4:6] = 1
    return EpisodeTrace(initial_observation=np.zeros((2, 3, 3), np.float32),
        observations=np.ones((d, 2, 3, 3), np.float32), vector_rewards=np.ones((d, 2), np.float32),
        true_states=np.zeros((d, n, 4), np.float32), estimated_states=est,
        actions=np.zeros((d, 2), np.int64), observed_targets=observed,
        elapsed_steps=np.array([2, 5, 3][:d], np.int64), coverage_values=np.array([.2, .4, .8][:d]),
        terminated=np.arange(d) == d - 1, truncated=np.zeros(d, bool), seed=1, initial_known_count=known)


class SharedEpisodeMetrics(unittest.TestCase):
    # Params: self; returns: None.
    def test_unique_ids_first_action_repeated_and_never_detected(self):
        observed = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
        self.assertEqual(shared_metrics.count_unique_new_detections(observed, 1), 2)
        self.assertEqual(shared_metrics.count_unique_new_detections(observed, 0), 3)
        self.assertEqual(shared_metrics.count_unique_new_detections(observed, 4), 0)
        self.assertEqual(shared_metrics.count_unique_new_detections(np.zeros((0, 3), bool), 1), 0)
        for mask, count in (([1, 0], 0), ([[2]], 0), ([[1]], -1), ([[1]], 2)):
            with self.assertRaises(ValueError):
                shared_metrics.count_unique_new_detections(mask, count)

    # Params: self; returns: None.
    def test_frequency_slew_discovery_zero_known_and_missing_unknown(self):
        observed = np.array([[1, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 1, 0]])
        result = shared_metrics.compute_observation_frequency_per_target(observed, [2, 5, 3], 2)
        self.assertEqual([item.to_dict() for item in result], [
            dict(target_id=0, initially_known=True, observation_count=2, counting_start_timestep=0,
                 eligible_timesteps=10, observation_frequency=.2),
            dict(target_id=1, initially_known=True, observation_count=0, counting_start_timestep=0,
                 eligible_timesteps=10, observation_frequency=0),
            dict(target_id=2, initially_known=False, observation_count=2, counting_start_timestep=2,
                 eligible_timesteps=9, observation_frequency=2/9),
            dict(target_id=3, initially_known=False, observation_count=1, counting_start_timestep=10,
                 eligible_timesteps=1, observation_frequency=1),
            dict(target_id=4, initially_known=False, observation_count=0, counting_start_timestep=None,
                 eligible_timesteps=None, observation_frequency=None)])
        summary = shared_metrics.summarize_observation_frequencies(result)
        self.assertEqual(summary['mean_known_observation_frequency'], .1)
        self.assertEqual(summary['mean_observation_frequency'], np.mean([.2, 0, 2/9, 1]))
        self.assertEqual(result, shared_metrics.compute_observation_frequency_per_target(
            observed, [2, 5, 3], np.int64(2)))

    # Params: self; returns: None.
    def test_frequency_missing_subsets_and_equal_target_weighting(self):
        unknown = shared_metrics.compute_observation_frequency_per_target([[0, 0]], [3], 0)
        self.assertEqual(shared_metrics.summarize_observation_frequencies(unknown),
            {'mean_observation_frequency': None, 'mean_known_observation_frequency': None})
        self.assertEqual(shared_metrics.summarize_observation_frequencies([]),
            {'mean_observation_frequency': None, 'mean_known_observation_frequency': None})
        result = shared_metrics.compute_observation_frequency_per_target([[0, 0], [1, 1]], [9, 1], 1)
        self.assertEqual(shared_metrics.summarize_observation_frequencies(result)['mean_observation_frequency'], .55)
        with self.assertRaises(ValueError):
            shared_metrics.summarize_observation_frequencies([result[0], result[0]])

    # Params: self; returns: None.
    def test_frequency_invalid_alignment_masks_durations_and_counts(self):
        for mask, elapsed, count in (([[0]], [], 0), ([[0]], [0], 0), ([[0]], [1.5], 0),
                ([[0]], [np.nan], 0), ([[2]], [1], 0), ([], [], 0), ([[0]], [1], -1)):
            with self.assertRaises(ValueError):
                shared_metrics.compute_observation_frequency_per_target(mask, elapsed, count)
        for count in (True, .5):
            with self.assertRaises(TypeError):
                shared_metrics.compute_observation_frequency_per_target([[0]], [1], count)

    # Params: self; returns: None.
    def test_coverage_is_post_action_mean_and_final(self):
        self.assertEqual(shared_metrics.summarize_coverage([.2, .4, .8]),
                         {'mean_coverage': np.mean([.2, .4, .8]), 'final_coverage': .8})
        for values in ([], [[.2]], [np.nan], [-1]):
            with self.assertRaises(ValueError):
                shared_metrics.summarize_coverage(values)

    # Params: self; returns: None.
    def test_search_time_requires_explicit_modes_and_weights_durations(self):
        self.assertIsNone(shared_metrics.search_time_percentage(None, [2, 5, 3]))
        self.assertEqual(shared_metrics.search_time_percentage(['search', 'track', 'search'], [2, 5, 3]), 50)
        self.assertEqual(shared_metrics.search_time_percentage(['track'], [5]), 0)
        self.assertEqual(shared_metrics.search_time_percentage(['search'], [5]), 100)
        for modes, elapsed in ((['combined'], [1]), (['search'], [0]), ([], []),
                               (['search'], [1, 2]), (['track'], [np.inf])):
            with self.assertRaises(ValueError):
                shared_metrics.search_time_percentage(modes, elapsed)

    # Params: self; returns: None.
    def test_episode_assembly_uses_trace_metrics_and_keeps_ppo_mode_missing(self):
        trace = episode_trace()
        env = EnvConfig(known_targets=1, unknown_targets=2, grid_shape=(2, 3))
        result = shared_metrics.summarize_episode(trace, env, {'error_target_set': 'catalogue'})
        self.assertEqual(result.mean_positional_error, 0)
        self.assertEqual(result.mean_final_positional_error, 0)
        self.assertEqual(result.mean_covariance_trace, 2)
        self.assertEqual(result.unique_new_detections, 2)
        self.assertEqual(result.decisions, 3)
        self.assertEqual(result.elapsed_timesteps, 10)
        self.assertEqual(result.final_coverage, .8)
        self.assertEqual(result.mean_coverage, np.mean([.2, .4, .8]))
        self.assertEqual(result.mean_known_observation_frequency, .2)
        self.assertIsNone(result.search_time_percentage)
        self.assertEqual(set(result.unavailable_reasons), {'search_time_percentage'})
        self.assertEqual(len(result.target_observation_metrics), 3)
        self.assertIsNone(result.to_dict()['search_time_percentage'])
        config = ExperimentConfig(env=env)
        self.assertEqual(shared_metrics.summarize_episode(trace, env, config), result)

    # Params: self; returns: None.
    def test_episode_unavailable_values_have_reasons_and_zero_stays_valid(self):
        trace = episode_trace(known=0, observed=np.zeros((3, 3), bool), valid=False)
        result = shared_metrics.summarize_episode(trace,
            EnvConfig(known_targets=0, unknown_targets=3, grid_shape=(2, 3)), {'error_target_set': 'catalogue'})
        for name in ('mean_positional_error', 'mean_final_positional_error', 'mean_covariance_trace',
                     'mean_observation_frequency', 'mean_known_observation_frequency', 'search_time_percentage'):
            self.assertIsNone(getattr(result, name))
            self.assertTrue(result.unavailable_reasons[name])
        self.assertEqual(result.unique_new_detections, 0)
        known = episode_trace(known=1, observed=np.zeros((3, 3), bool))
        result = shared_metrics.summarize_episode(known,
            EnvConfig(known_targets=1, unknown_targets=2, grid_shape=(2, 3)), {'error_target_set': 'initial'})
        self.assertEqual(result.mean_known_observation_frequency, 0)
        self.assertEqual(result.mean_observation_frequency, 0)

    # Params: self; returns: None.
    def test_episode_final_missing_does_not_erase_valid_earlier_error(self):
        from dataclasses import replace
        trace = episode_trace()
        est = trace.estimated_states.copy()
        est[-1] = 0
        result = shared_metrics.summarize_episode(replace(trace, estimated_states=est),
            EnvConfig(known_targets=1, unknown_targets=2, grid_shape=(2, 3)), {'error_target_set': 'catalogue'})
        self.assertEqual(result.mean_positional_error, 0)
        self.assertIsNone(result.mean_final_positional_error)
        self.assertIn('mean_final_positional_error', result.unavailable_reasons)

    # Params: self; returns: None.
    def test_episode_config_mismatches_raise(self):
        trace = episode_trace()
        env = EnvConfig(known_targets=1, unknown_targets=2, grid_shape=(2, 3))
        for bad in (EnvConfig(known_targets=2, unknown_targets=1, grid_shape=(2, 3)),
                    EnvConfig(known_targets=1, unknown_targets=3, grid_shape=(2, 3)), EnvConfig()):
            with self.assertRaises(ValueError):
                shared_metrics.summarize_episode(trace, bad, {'error_target_set': 'catalogue'})
        for config in ({'error_target_set': 'all'}, {'error_target_set': 'catalogue', 'world_size': 0},
                       {'error_target_set': 'catalogue', 'guess_mode': True}):
            with self.assertRaises(ValueError):
                shared_metrics.summarize_episode(trace, env, config)

    # Params: self; returns: None.
    def test_finite_summaries_preserve_legacy_missing_and_sd_behavior(self):
        for values in ([], [np.nan, np.inf], [1], [1, 3, np.nan]):
            np.testing.assert_allclose(shared_metrics.finite_mean_and_std(values),
                metrics.finite_mean_and_std(values), equal_nan=True)

    # Params: self; returns: None.
    def test_frequency_and_detections_match_all_70_rollouts(self):
        for dataset, expected in BASELINE.items():
            root = PROJECT_ROOT / 'rollouts' / dataset
            if not root.exists():
                self.skipTest('External baseline rollouts unavailable')
            for name, values in expected['per_run'].items():
                with self.subTest(dataset=dataset, run=name):
                    path = root / name
                    truth = np.load(path / 'true_states.npy', allow_pickle=False)
                    est = np.load(path / 'est_states.npy', allow_pickle=False)
                    meta = metrics.load_meta(path)
                    known = shared_metrics.initial_catalogue_size(meta)
                    observed = metrics.load_target_observations(path, truth, meta, 100)
                    elapsed = np.load(path / 'elapsed_steps.npy', allow_pickle=False)
                    records = shared_metrics.compute_observation_frequency_per_target(observed, elapsed, known)
                    legacy = metrics.compute_observation_frequency_per_target(observed, elapsed, known)
                    self.assertEqual([r.to_dict() for r in records], [vars(r) for r in legacy])
                    summary = shared_metrics.summarize_observation_frequencies(records)
                    self.assertEqual(summary['mean_observation_frequency'], values[4])
                    self.assertEqual(summary['mean_known_observation_frequency'], values[5])
                    detections = shared_metrics.count_unique_new_detections(observed, known)
                    self.assertEqual(detections, values[0])
                    self.assertEqual(detections, np.count_nonzero(shared_metrics.catalogue_mask(est)[-1, known:]))


class SharedPlotDelegation(unittest.TestCase):
    # Params: self; returns: None.
    def test_plotting_public_helpers_are_shared_imports(self):
        for name in ('initial_catalogue_size', 'catalogue_mask', 'wrapped_difference',
                     'compute_mean_positional_error', 'compute_mean_final_positional_error',
                     'compute_observation_frequency_per_target', 'finite_mean_and_std', 'sample_std'):
            self.assertIs(getattr(metrics, name), getattr(shared_metrics, name))
        self.assertIs(metrics.TargetObservationMetrics, TargetObservationMetrics)

    # Params: self; returns: None.
    def test_plot_search_adapter_delegates_reconstructed_and_explicit_modes(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            est = np.zeros((2, 1, 9), np.float32)
            np.save(path / 'policy_modes.npy', np.array(['search', 'track']))
            np.save(path / 'elapsed_steps.npy', [2, 3])
            with patch.object(metrics, 'search_time_percentage', wraps=shared_metrics.search_time_percentage) as helper:
                self.assertEqual(metrics.compute_search_time(path, 1, est, {}), 40)
                np.testing.assert_array_equal(helper.call_args[0][0], ['search', 'track'])
                helper.assert_called_once()
            (path / 'policy_modes.npy').unlink()
            # Legacy post-action uncertainty reconstruction retains its initial value.
            np.save(path / 'observations.npy', np.zeros((2, 2, 2, 3), np.float32))
            with patch.object(metrics, 'search_time_percentage', wraps=shared_metrics.search_time_percentage) as helper:
                self.assertEqual(metrics.compute_search_time(path, 0, est, {'known_targets': 0}), 100)
                helper.assert_called_once()
            (path / 'elapsed_steps.npy').unlink()
            self.assertEqual(metrics.compute_search_time(path, 0, est, {'known_targets': 0}), 100)

    # Params: self; returns: None.
    def test_direct_script_and_module_cli_keep_the_same_help(self):
        import subprocess
        direct = subprocess.run([sys.executable, '-B', 'scripts/plot_combined_threshold_tradeoff.py', '--help'],
            cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True)
        module = subprocess.run([sys.executable, '-B', '-m', 'scripts.plot_combined_threshold_tradeoff', '--help'],
            cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True)
        self.assertEqual(direct.stdout, module.stdout)


if __name__ == "__main__":
    unittest.main()
