"""Block 01 baseline, captured with Python 3.9.13 / NumPy 2.0.2 / SB3 2.7.1.

Frozen expectations are intentional: tests never regenerate their own baseline.
Run with the project's training interpreter using unittest (no pytest required).
"""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
from dataclasses import replace
import os
from unittest.mock import Mock, patch
from agents.mo import EnvConfig, ExperimentConfig, ExecutionConfig, NormalizationConfig
from agents.mo.config import make_seed_plan
from agents.mo import envs as mo_envs

import hashlib
import gymnasium as gym
from agents.mo.wrappers import LinearScalarizationWrapper as MOScalarizationWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "agents/train/train_ppo_combined.py").exists():
    PROJECT_ROOT = Path(r"C:\Users\amaur\Documents\TUD Thesis\minimal_env_grid")
sys.path.insert(0, str(PROJECT_ROOT))

from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv
from agents.train.train_ppo_combined import LinearScalarizationWrapper

CASES = json.loads(r'''
{
  "slew_default": {
    "config": {
      "known_targets": 10,
      "unknown_targets": 10,
      "grid_shape": [
        10,
        10
      ],
      "timesteps": 200,
      "measurement_noise_std": 1.0,
      "reward_type": "R6",
      "search_reward": "R2",
      "zero_noise": false,
      "mode": "combined",
      "dwell_steps": 1,
      "slew_alpha": 1.0,
      "uncertainty_norm_scale": 100.0
    },
    "seed": 123,
    "actions": [
      [
        5,
        5
      ],
      [
        5,
        6
      ],
      [
        0,
        0
      ],
      [
        9,
        9
      ],
      [
        0,
        9
      ],
      [
        9,
        0
      ]
    ],
    "initial": {
      "obs_sums": [
        2.380462646484375,
        0.0,
        1.0
      ],
      "obs_samples": [
        [
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0
        ]
      ],
      "true_state_samples": [
        [
          68.23518371582031,
          5.382102012634277,
          -0.559280276298523,
          -0.6312564015388489
        ],
        [
          7.148680686950684,
          58.3238410949707,
          -0.5241872072219849,
          0.5299273133277893
        ],
        [
          61.01448440551758,
          74.38790893554688,
          -0.14033935964107513,
          -0.39439573884010315
        ]
      ]
    },
    "steps": [
      {
        "reward": [
          0.00217129266820848,
          0.9911110401153564
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 2,
        "observed_target_ids": [
          12,
          18
        ],
        "elapsed_steps": 1,
        "total_steps": 1,
        "sensor_pointing": [
          55.0,
          55.0
        ],
        "obs_sums": [
          3.787515163421631,
          8.919999122619629,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            1.0,
            1.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            67.7081527709961,
            4.723681926727295,
            -0.527031660079956,
            -0.6584198474884033
          ],
          [
            6.598663330078125,
            58.826908111572266,
            -0.5500174164772034,
            0.5030665993690491
          ],
          [
            60.8940315246582,
            73.98699188232422,
            -0.1204524040222168,
            -0.4009169042110443
          ]
        ],
        "estimated_state_samples": [
          [
            67.6919174194336,
            4.7371296882629395,
            -0.527031660079956,
            -0.6584198474884033,
            1.9997917413711548,
            1.9997917413711548,
            1.0,
            1.0,
            5.9995832443237305
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ]
        ]
      },
      {
        "reward": [
          0.0023961784318089485,
          0.33000001311302185
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 2,
        "observed_target_ids": [
          15,
          19
        ],
        "elapsed_steps": 2,
        "total_steps": 3,
        "sensor_pointing": [
          55.0,
          65.0
        ],
        "obs_sums": [
          7.231960773468018,
          11.600423812866211,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.9900000095367432,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            66.79081726074219,
            3.308190107345581,
            -0.4586663842201233,
            -0.7077458500862122
          ],
          [
            5.4036784172058105,
            59.71819305419922,
            -0.5974924564361572,
            0.44564324617385864
          ],
          [
            60.734378814697266,
            73.16511535644531,
            -0.0798257365822792,
            -0.41093915700912476
          ]
        ],
        "estimated_state_samples": [
          [
            66.70539855957031,
            3.3698244094848633,
            -0.4586663842201233,
            -0.7077458500862122,
            9.983138084411621,
            9.983138084411621,
            1.0,
            1.0,
            21.966276168823242
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.46150588989258,
            73.58756256103516,
            0.0,
            0.0,
            0.5,
            0.5,
            1.0,
            1.0,
            3.0
          ]
        ]
      },
      {
        "reward": [
          0.020442238077521324,
          0.9911110401153564
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 2,
        "observed_target_ids": [
          6,
          11
        ],
        "elapsed_steps": 8,
        "total_steps": 11,
        "sensor_pointing": [
          5.0,
          5.0
        ],
        "obs_sums": [
          13.387029647827148,
          16.615957260131836,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            1.0,
            1.0
          ],
          [
            0.0,
            0.6567862629890442,
            0.0
          ],
          [
            0.0,
            0.9900000095367432,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            65.61601257324219,
            96.66426849365234,
            -0.14685049653053284,
            -0.8304901719093323
          ],
          [
            99.61273193359375,
            61.1405143737793,
            -0.7238686680793762,
            0.1777900904417038
          ],
          [
            61.4264030456543,
            69.888427734375,
            0.08650287240743637,
            -0.4095856249332428
          ]
        ],
        "estimated_state_samples": [
          [
            64.2505111694336,
            97.13350677490234,
            -0.14685049653053284,
            -0.8304901719093323,
            118.98038482666016,
            118.98038482666016,
            1.0,
            1.0,
            239.9607696533203
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.46150588989258,
            73.58756256103516,
            0.0,
            0.0,
            63.65120315551758,
            63.65120315551758,
            1.0,
            1.0,
            129.30239868164062
          ]
        ]
      },
      {
        "reward": [
          0.014416643418371677,
          0.5499999523162842
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 1,
        "observed_target_ids": [
          2
        ],
        "elapsed_steps": 3,
        "total_steps": 14,
        "sensor_pointing": [
          95.0,
          95.0
        ],
        "obs_sums": [
          13.202202796936035,
          19.762327194213867,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.9900000095367432,
            0.0
          ],
          [
            0.0,
            0.5631121397018433,
            0.0
          ],
          [
            0.0,
            1.0,
            1.0
          ]
        ],
        "true_state_samples": [
          [
            65.55272674560547,
            94.13494110107422,
            -0.02109462395310402,
            -0.8431097269058228
          ],
          [
            97.38580322265625,
            61.34337615966797,
            -0.742309033870697,
            0.06762012094259262
          ],
          [
            61.86661911010742,
            68.71224975585938,
            0.14673924446105957,
            -0.3920595943927765
          ]
        ],
        "estimated_state_samples": [
          [
            63.99811935424805,
            94.61839294433594,
            -0.02109462395310402,
            -0.8431097269058228,
            189.12625122070312,
            189.12625122070312,
            1.0,
            1.0,
            380.25250244140625
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.46150588989258,
            73.58756256103516,
            0.0,
            0.0,
            118.48037719726562,
            118.48037719726562,
            1.0,
            1.0,
            238.96075439453125
          ]
        ]
      },
      {
        "reward": [
          0.003317601978778839,
          0.1413775086402893
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 1,
        "observed_target_ids": [
          6
        ],
        "elapsed_steps": 2,
        "total_steps": 16,
        "sensor_pointing": [
          5.0,
          95.0
        ],
        "obs_sums": [
          13.321006774902344,
          19.853534698486328,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.9900000095367432,
            0.0
          ],
          [
            0.0,
            0.508208692073822,
            0.0
          ],
          [
            0.0,
            0.9900000095367432,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            65.67909240722656,
            92.45293426513672,
            0.06318128854036331,
            -0.8410036563873291
          ],
          [
            95.89510345458984,
            61.32972717285156,
            -0.7453513145446777,
            -0.0068249450996518135
          ],
          [
            62.23691177368164,
            67.96134948730469,
            0.18514680862426758,
            -0.3754514455795288
          ]
        ],
        "estimated_state_samples": [
          [
            64.04023742675781,
            92.93287658691406,
            0.06318128854036331,
            -0.8410036563873291,
            243.63462829589844,
            243.63462829589844,
            1.0,
            1.0,
            489.2692565917969
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.46150588989258,
            73.58756256103516,
            0.0,
            0.0,
            163.63294982910156,
            163.63294982910156,
            1.0,
            1.0,
            329.2658996582031
          ]
        ]
      },
      {
        "reward": [
          0.002910234499722719,
          0.1812182068824768
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 0,
        "observed_target_ids": [],
        "elapsed_steps": 3,
        "total_steps": 19,
        "sensor_pointing": [
          95.0,
          5.0
        ],
        "obs_sums": [
          12.584653854370117,
          19.692487716674805,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.9900000095367432,
            0.0
          ],
          [
            0.0,
            0.43572545051574707,
            0.0
          ],
          [
            0.0,
            0.9900000095367432,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            66.24354553222656,
            89.98657989501953,
            0.1881498396396637,
            -0.8221184015274048
          ],
          [
            93.68721771240234,
            60.9753303527832,
            -0.7359619140625,
            -0.11813221871852875
          ],
          [
            62.954437255859375,
            66.93064880371094,
            0.23917457461357117,
            -0.34356755018234253
          ]
        ],
        "estimated_state_samples": [
          [
            64.4179458618164,
            90.43350219726562,
            0.1881498396396637,
            -0.8221184015274048,
            335.6535339355469,
            335.6535339355469,
            1.0,
            1.0,
            673.3070678710938
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.46150588989258,
            73.58756256103516,
            0.0,
            0.0,
            243.13462829589844,
            243.13462829589844,
            1.0,
            1.0,
            488.2692565917969
          ]
        ]
      }
    ]
  },
  "no_slew_termination": {
    "config": {
      "known_targets": 10,
      "unknown_targets": 10,
      "grid_shape": [
        10,
        10
      ],
      "timesteps": 6,
      "measurement_noise_std": 1.0,
      "reward_type": "R6",
      "search_reward": "R2",
      "zero_noise": false,
      "mode": "combined",
      "dwell_steps": 1,
      "slew_alpha": 0.0,
      "uncertainty_norm_scale": 100.0
    },
    "seed": 123,
    "actions": [
      [
        5,
        5
      ],
      [
        5,
        6
      ],
      [
        0,
        0
      ],
      [
        9,
        9
      ],
      [
        0,
        9
      ],
      [
        9,
        0
      ]
    ],
    "initial": {
      "obs_sums": [
        2.380462646484375,
        0.0,
        1.0
      ],
      "obs_samples": [
        [
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0
        ]
      ],
      "true_state_samples": [
        [
          68.23518371582031,
          5.382102012634277,
          -0.559280276298523,
          -0.6312564015388489
        ],
        [
          7.148680686950684,
          58.3238410949707,
          -0.5241872072219849,
          0.5299273133277893
        ],
        [
          61.01448440551758,
          74.38790893554688,
          -0.14033935964107513,
          -0.39439573884010315
        ]
      ]
    },
    "steps": [
      {
        "reward": [
          0.00217129266820848,
          0.9911110401153564
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 2,
        "observed_target_ids": [
          12,
          18
        ],
        "elapsed_steps": 1,
        "total_steps": 1,
        "sensor_pointing": [
          55.0,
          55.0
        ],
        "obs_sums": [
          3.787515163421631,
          8.919999122619629,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            1.0,
            1.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            67.7081527709961,
            4.723681926727295,
            -0.527031660079956,
            -0.6584198474884033
          ],
          [
            6.598663330078125,
            58.826908111572266,
            -0.5500174164772034,
            0.5030665993690491
          ],
          [
            60.8940315246582,
            73.98699188232422,
            -0.1204524040222168,
            -0.4009169042110443
          ]
        ],
        "estimated_state_samples": [
          [
            67.6919174194336,
            4.7371296882629395,
            -0.527031660079956,
            -0.6584198474884033,
            1.9997917413711548,
            1.9997917413711548,
            1.0,
            1.0,
            5.9995832443237305
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ]
        ]
      },
      {
        "reward": [
          0.0024641137570142746,
          0.33000001311302185
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 2,
        "observed_target_ids": [
          15,
          19
        ],
        "elapsed_steps": 1,
        "total_steps": 2,
        "sensor_pointing": [
          55.0,
          65.0
        ],
        "obs_sums": [
          6.095522403717041,
          11.741498947143555,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.9900000095367432,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            67.21468353271484,
            4.0397443771362305,
            -0.49346572160720825,
            -0.6839376091957092
          ],
          [
            6.024190425872803,
            59.301856994628906,
            -0.5744729042053223,
            0.4749484956264496
          ],
          [
            60.793766021728516,
            73.58055877685547,
            -0.10026437789201736,
            -0.4064359664916992
          ]
        ],
        "estimated_state_samples": [
          [
            67.18156433105469,
            4.0658111572265625,
            -0.49346572160720825,
            -0.6839376091957092,
            4.996667861938477,
            4.996667861938477,
            1.0,
            1.0,
            11.993335723876953
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.52089309692383,
            74.00300598144531,
            0.0,
            0.0,
            0.5,
            0.5,
            1.0,
            1.0,
            3.0
          ]
        ]
      },
      {
        "reward": [
          0.08561058342456818,
          0.9911110401153564
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 1,
        "observed_target_ids": [
          6
        ],
        "elapsed_steps": 1,
        "total_steps": 3,
        "sensor_pointing": [
          5.0,
          5.0
        ],
        "obs_sums": [
          7.097438335418701,
          20.074424743652344,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            1.0,
            1.0
          ],
          [
            0.0,
            0.940500020980835,
            0.0
          ],
          [
            0.0,
            0.9900000095367432,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            66.75601959228516,
            3.331998586654663,
            -0.4586663842201233,
            -0.7077458500862122
          ],
          [
            5.426697731018066,
            59.747501373291016,
            -0.597492516040802,
            0.44564324617385864
          ],
          [
            60.71393966674805,
            73.16961669921875,
            -0.0798257440328598,
            -0.41093915700912476
          ]
        ],
        "estimated_state_samples": [
          [
            66.70539855957031,
            3.3698244094848633,
            -0.4586663842201233,
            -0.7077458500862122,
            9.983138084411621,
            9.983138084411621,
            1.0,
            1.0,
            21.966276168823242
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.52089309692383,
            74.00300598144531,
            0.0,
            0.0,
            1.4997917413711548,
            1.4997917413711548,
            1.0,
            1.0,
            4.9995832443237305
          ]
        ]
      },
      {
        "reward": [
          0.09172491729259491,
          0.5499999523162842
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 1,
        "observed_target_ids": [
          2
        ],
        "elapsed_steps": 1,
        "total_steps": 4,
        "sensor_pointing": [
          95.0,
          95.0
        ],
        "obs_sums": [
          8.996997833251953,
          24.219202041625977,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.9900000095367432,
            0.0
          ],
          [
            0.0,
            0.8934749960899353,
            0.0
          ],
          [
            0.0,
            1.0,
            1.0
          ]
        ],
        "true_state_samples": [
          [
            66.33329772949219,
            2.6022133827209473,
            -0.42272061109542847,
            -0.729785144329071
          ],
          [
            4.807679176330566,
            60.16272735595703,
            -0.6190186738967896,
            0.4152241349220276
          ],
          [
            60.65475082397461,
            72.75520324707031,
            -0.05918758362531662,
            -0.4144152104854584
          ]
        ],
        "estimated_state_samples": [
          [
            66.26461029052734,
            2.650909185409546,
            -0.42272061109542847,
            -0.729785144329071,
            16.94673728942871,
            16.94673728942871,
            1.0,
            1.0,
            35.89347457885742
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.52089309692383,
            74.00300598144531,
            0.0,
            0.0,
            4.496667861938477,
            4.496667861938477,
            1.0,
            1.0,
            10.993335723876953
          ]
        ]
      },
      {
        "reward": [
          0.010597837157547474,
          0.12099999934434891
        ],
        "terminated": false,
        "truncated": false,
        "num_measurements": 1,
        "observed_target_ids": [
          6
        ],
        "elapsed_steps": 1,
        "total_steps": 5,
        "sensor_pointing": [
          5.0,
          95.0
        ],
        "obs_sums": [
          10.193346977233887,
          24.488794326782227,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.9900000095367432,
            0.0
          ],
          [
            0.0,
            0.8488012552261353,
            0.0
          ],
          [
            0.0,
            0.9900000095367432,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            65.94757843017578,
            1.8522130250930786,
            -0.3857182562351227,
            -0.7500003576278687
          ],
          [
            4.168681621551514,
            60.54649353027344,
            -0.6389976143836975,
            0.38376718759536743
          ],
          [
            60.61634826660156,
            72.33834838867188,
            -0.038401488214731216,
            -0.41685545444488525
          ]
        ],
        "estimated_state_samples": [
          [
            65.86030578613281,
            1.9108622074127197,
            -0.3857182562351227,
            -0.7500003576278687,
            25.87006187438965,
            25.87006187438965,
            1.0,
            1.0,
            53.7401237487793
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.52089309692383,
            74.00300598144531,
            0.0,
            0.0,
            9.483138084411621,
            9.483138084411621,
            1.0,
            1.0,
            20.966276168823242
          ]
        ]
      },
      {
        "reward": [
          0.003214447060599923,
          0.1424499899148941
        ],
        "terminated": true,
        "truncated": false,
        "num_measurements": 0,
        "observed_target_ids": [],
        "elapsed_steps": 1,
        "total_steps": 6,
        "sensor_pointing": [
          95.0,
          5.0
        ],
        "obs_sums": [
          11.337287902832031,
          24.928298950195312,
          1.0
        ],
        "obs_samples": [
          [
            0.0,
            0.9900000095367432,
            0.0
          ],
          [
            0.0,
            0.806361198425293,
            0.0
          ],
          [
            0.0,
            0.9900000095367432,
            0.0
          ]
        ],
        "true_state_samples": [
          [
            65.59982299804688,
            1.0838720798492432,
            -0.34775182604789734,
            -0.7683409452438354
          ],
          [
            3.5113022327423096,
            60.89784622192383,
            -0.6573793888092041,
            0.3513510227203369
          ],
          [
            60.59882736206055,
            71.92009735107422,
            -0.017519406974315643,
            -0.4182537794113159
          ]
        ],
        "estimated_state_samples": [
          [
            65.49349212646484,
            1.1515333652496338,
            -0.34775182604789734,
            -0.7683409452438354,
            36.73080825805664,
            36.73080825805664,
            1.0,
            1.0,
            75.46161651611328
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            60.52089309692383,
            74.00300598144531,
            0.0,
            0.0,
            16.44673728942871,
            16.44673728942871,
            1.0,
            1.0,
            34.89347457885742
          ]
        ]
      }
    ]
  }
}
''')

STATE_IDS = json.loads(r'''
[
  0,
  10,
  19
]
''')

CELLS = json.loads(r'''
[
  [
    0,
    0
  ],
  [
    5,
    5
  ],
  [
    9,
    9
  ]
]
''')


class ExistingEnvironmentBaseline(unittest.TestCase):
    # Params: self, actual, expected; returns: None.
    def assert_numeric(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)

    # Params: self, name; returns: None.
    def check_case(self, name):
        case = CASES[name]
        env = MinimalSSAEnv(**case["config"])
        self.addCleanup(env.close)
        obs, info = env.reset(seed=case["seed"])
        self.assertEqual(obs.shape, (10, 10, 3))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(env.observation_space.shape, (10, 10, 3))
        np.testing.assert_array_equal(env.action_space.nvec, [10, 10])
        self.assert_numeric(obs.sum(axis=(0, 1)), case["initial"]["obs_sums"])
        self.assert_numeric([obs[x, y] for x, y in CELLS], case["initial"]["obs_samples"])
        self.assert_numeric(info["true_states"][STATE_IDS], case["initial"]["true_state_samples"])
        for action, expected in zip(case["actions"], case["steps"]):
            with self.subTest(case=name, action=action, total_steps=expected["total_steps"]):
                obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.int64))
                self.assertEqual(reward.shape, (2,))
                self.assertEqual(reward.dtype, np.float32)
                self.assert_numeric(reward, expected["reward"])
                self.assertEqual(bool(terminated), expected["terminated"])
                self.assertEqual(bool(truncated), expected["truncated"])
                for key in ("num_measurements", "observed_target_ids", "elapsed_steps", "total_steps"):
                    self.assertEqual(info[key], expected[key])
                for key in ("sensor_pointing",):
                    self.assert_numeric(info[key], expected[key])
                self.assert_numeric(obs.sum(axis=(0, 1)), expected["obs_sums"])
                self.assert_numeric([obs[x, y] for x, y in CELLS], expected["obs_samples"])
                self.assert_numeric(info["true_states"][STATE_IDS], expected["true_state_samples"])
                self.assert_numeric(info["estimated_states"][STATE_IDS], expected["estimated_state_samples"])

    # Params: self; returns: None.
    def test_fixed_actions_with_slew(self):
        self.check_case("slew_default")

    # Params: self; returns: None.
    def test_fixed_actions_through_episode_termination(self):
        self.check_case("no_slew_termination")

    # Params: self; returns: None.
    def test_scalarization_preserves_observations_and_diagnostics(self):
        case = CASES["slew_default"]
        weights = np.asarray([0.85, 0.15], dtype=np.float32)
        raw = MinimalSSAEnv(**case["config"])
        wrapped = LinearScalarizationWrapper(MinimalSSAEnv(**case["config"]), weights)
        self.addCleanup(raw.close)
        self.addCleanup(wrapped.close)
        raw_obs, raw_info = raw.reset(seed=case["seed"])
        obs, info = wrapped.reset(seed=case["seed"])
        np.testing.assert_array_equal(raw_obs, obs)
        np.testing.assert_array_equal(raw_info["true_states"], info["true_states"])
        for action in case["actions"]:
            raw_obs, vector, done, truncated, raw_info = raw.step(np.asarray(action))
            obs, scalar, wrapped_done, wrapped_truncated, info = wrapped.step(np.asarray(action))
            np.testing.assert_array_equal(raw_obs, obs)
            self.assertEqual((done, truncated), (wrapped_done, wrapped_truncated))
            self.assertIsInstance(scalar, float)
            self.assertEqual(scalar, float(np.dot(weights, vector)))
            np.testing.assert_array_equal(info["vector_reward"], vector)
            self.assertEqual(info["tracking_reward"], float(vector[0]))
            self.assertEqual(info["search_reward"], float(vector[1]))
            self.assertEqual(info["scalarized_reward"], scalar)
            for key, value in raw_info.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, info[key])
                else:
                    self.assertEqual(value, info[key])

    # Params: self; returns: None.
    def test_seeded_reset_and_steps_repeat_independently_of_global_rng(self):
        case = CASES["slew_default"]
        env = MinimalSSAEnv(**case["config"])
        self.addCleanup(env.close)
        global_state = np.random.get_state()
        self.addCleanup(np.random.set_state, global_state)
        traces = []
        for repetition in range(2):
            np.random.seed(987 + repetition)
            obs, info = env.reset(seed=case["seed"])
            trace = [obs.copy(), info["true_states"].copy()]
            for action in case["actions"]:
                np.random.normal(size=37)
                obs, reward, done, truncated, info = env.step(np.asarray(action))
                trace.extend([obs.copy(), reward.copy(), info["true_states"].copy(),
                              info["estimated_states"].copy()])
            traces.append(trace)
        for first, second in zip(*traces):
            np.testing.assert_array_equal(first, second)



SCALAR_MODE_BASELINE = json.loads(r'''
{
  "track": {
    "config": {
      "mode": "track",
      "known_targets": 10,
      "unknown_targets": 0,
      "reward_type": "R6",
      "search_reward": "R2",
      "timesteps": 200
    },
    "seed": 123,
    "actions": [
      [
        5,
        5
      ],
      [
        5,
        6
      ],
      [
        0,
        0
      ],
      [
        9,
        9
      ],
      [
        0,
        9
      ],
      [
        9,
        0
      ]
    ],
    "initial": "a668a9dfe608fc538ed3228402be945ba908b4dfe4820f312fb4df5f20dbb052",
    "steps": [
      {
        "fingerprint": "eaffa1a0aea6c07a052e14864b8749a69ee629ce5c97fad9d699220d4c3da4cb",
        "reward": 0.0021712927290104445,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 1,
        "total_steps": 1,
        "observed_target_ids": []
      },
      {
        "fingerprint": "9359238f4c06d479a867854748e79fcf8aaeacc4689e11148df30b596400900d",
        "reward": 0.002612038749506592,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 2,
        "total_steps": 3,
        "observed_target_ids": []
      },
      {
        "fingerprint": "3a5d2a5801130452e244a49b4ba56b702515f5b2187574b3e2ae1a0f7327e4be",
        "reward": 0.026011866818460776,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 8,
        "total_steps": 11,
        "observed_target_ids": [
          6
        ]
      },
      {
        "fingerprint": "eba2476e303dd4c40ad426420ef664606e98aec3478f257f1cf243e38e60d6e7",
        "reward": 0.019271104488835455,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 3,
        "total_steps": 14,
        "observed_target_ids": [
          2
        ]
      },
      {
        "fingerprint": "b57849743a88f07161230592bb8c52ba3c2a1c28cfc9f8ca2ca2695d5f2be6dc",
        "reward": 0.004647848549473607,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 2,
        "total_steps": 16,
        "observed_target_ids": [
          6
        ]
      },
      {
        "fingerprint": "152d5a6c9c344b8a23dcfde446cb6f2972d492d3500bd85a0b08603d356ecca9",
        "reward": 0.0041621004948767205,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 3,
        "total_steps": 19,
        "observed_target_ids": []
      }
    ]
  },
  "search": {
    "config": {
      "mode": "search",
      "known_targets": 0,
      "unknown_targets": 10,
      "reward_type": "R6",
      "search_reward": "R2",
      "timesteps": 200
    },
    "seed": 123,
    "actions": [
      [
        5,
        5
      ],
      [
        5,
        6
      ],
      [
        0,
        0
      ],
      [
        9,
        9
      ],
      [
        0,
        9
      ],
      [
        9,
        0
      ]
    ],
    "initial": "c020d776fc730384dd5db6d4cbdfb66b590800ec74cfd451e4fd507943739f2d",
    "steps": [
      {
        "fingerprint": "b9f0e514c1530c4a2add745869e4c084ccec06c882c7180c95e9a458a36c6606",
        "reward": 0.9911110401153564,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 1,
        "total_steps": 1,
        "observed_target_ids": []
      },
      {
        "fingerprint": "a5f3ae7e99049e75dfb6e2d95b9b2193ff43caeffa27001aa776793c97ef84cd",
        "reward": 0.33000001311302185,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 2,
        "total_steps": 3,
        "observed_target_ids": []
      },
      {
        "fingerprint": "ad80751d6e098b6d19fa228e51ec3807f177e47f9d058b08d48ecc67416a7928",
        "reward": 0.9911110401153564,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 8,
        "total_steps": 11,
        "observed_target_ids": [
          6
        ]
      },
      {
        "fingerprint": "8811b50f6cc8a23704a059d13fbd14eb988ab9182f32d7fa9d754ff0b0172174",
        "reward": 0.5499999523162842,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 3,
        "total_steps": 14,
        "observed_target_ids": [
          2
        ]
      },
      {
        "fingerprint": "bf453490d41cea4165f6c892e7ae8cd7365477c6da0051e96ffc4891cb0e91b3",
        "reward": 0.1413775086402893,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 2,
        "total_steps": 16,
        "observed_target_ids": [
          6
        ]
      },
      {
        "fingerprint": "3b28850e87799c878a24e3f3a0a9f6aa3a69bd4ee130a48d083b628962955c8a",
        "reward": 0.1812182068824768,
        "terminated": false,
        "truncated": false,
        "elapsed_steps": 3,
        "total_steps": 19,
        "observed_target_ids": []
      }
    ]
  }
}
''')

class SyntheticRewardEnv(gym.Env):
    # Params: self, reward; returns: None.
    def __init__(self, reward=None):
        self.observation_space = gym.spaces.Box(-1., 1., (2, 2, 3), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2, 2])
        self.observation = np.zeros((2, 2, 3), np.float32)
        self.reward = np.array([2, 6], np.float32) if reward is None else reward
        self.info = {"retained": np.array([1, 2])}
        self.calls = 0

    # Params: self, action; returns: (self.observation, self.reward, bool, bool, self.info).
    def step(self, action):
        self.calls += 1
        self.last_action = action
        return self.observation, self.reward, True, False, self.info

    # Params: self, seed, options; returns: (self.observation, self.info).
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation, self.info


# Params: obs, info; returns: digest.hexdigest result.
def scalar_fingerprint(obs, info):
    digest = hashlib.sha256()
    for value in (obs, info["true_states"], info.get("estimated_states"), info.get("sensor_pointing")):
        if value is not None:
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


class FixedObjectiveScalarization(unittest.TestCase):
    # Params: self; returns: None.
    def test_scaled_example_is_175_and_only_one_raw_step_is_executed(self):
        env = SyntheticRewardEnv()
        wrapper = MOScalarizationWrapper(env, [.25, .75], [2, 3])
        action = np.array([1, 1])
        obs, reward, terminated, truncated, info = wrapper.step(action)
        self.assertEqual(reward, 1.75)
        self.assertIs(type(reward), float)
        self.assertIs(obs, env.observation)
        self.assertIs(env.last_action, action)
        self.assertEqual((terminated, truncated), (True, False))
        self.assertEqual(env.calls, 1)
        self.assertIsNot(info, env.info)
        self.assertEqual(set(env.info), {"retained"})
        self.assertIs(info["retained"], env.info["retained"])
        np.testing.assert_array_equal(info["vector_reward"], [2, 6])
        np.testing.assert_array_equal(info["vector_reward_raw"], [2, 6])
        np.testing.assert_array_equal(info["vector_reward_scaled"], [1, 2])
        self.assertEqual(info["tracking_reward"], 2.)
        self.assertEqual(info["search_reward"], 6.)
        self.assertEqual(info["scalarized_reward"], 1.75)
        info["vector_reward"][0] = 999
        self.assertEqual(env.reward[0], 2)
        self.assertEqual(info["vector_reward_raw"][0], 2)

    # Params: self; returns: None.
    def test_preferences_and_scales_are_fixed_independent_immutable_copies(self):
        weights, scales = np.array([.25, .75], np.float32), np.array([2, 3], np.float32)
        wrapper = MOScalarizationWrapper(SyntheticRewardEnv(), weights, scales)
        weights[:] = [.5, .5]
        scales[:] = [10, 10]
        self.assertEqual(wrapper.step(np.array([0, 0]))[1], 1.75)
        for value in (wrapper.weights, wrapper.scales):
            with self.assertRaises(ValueError):
                value[0] = 99
            with self.assertRaises(ValueError):
                value.setflags(write=True)
        with self.assertRaises(AttributeError):
            wrapper.weights = np.array([.5, .5])
        with self.assertRaises(AttributeError):
            wrapper.scales = np.array([1, 1])

    # Params: self; returns: None.
    def test_invalid_weights_scales_and_conversion_underflow_are_rejected(self):
        for weights in ([1], [1, 0, 0], [-.1, 1.1], [-1e-60, 1.], [.5, .6], [np.nan, 1.], [True, 0]):
            with self.subTest(weights=weights), self.assertRaises((TypeError, ValueError)):
                MOScalarizationWrapper(SyntheticRewardEnv(), weights)
        for scales in ([1], [0, 1], [-1, 1], [np.inf, 1], [1e-60, 1], [1e100, 1], [True, 1]):
            with self.subTest(scales=scales), self.assertRaises((TypeError, ValueError)):
                MOScalarizationWrapper(SyntheticRewardEnv(), [.5, .5], scales)

    # Params: self; returns: None.
    def test_malformed_nonfinite_wrong_precision_rewards_and_scaled_overflow_fail(self):
        for reward in (np.array([1], np.float32), np.zeros((1, 2), np.float32),
                       np.array([np.nan, 1], np.float32), np.array([np.inf, 1], np.float32),
                       np.array([1, 2], np.float64), np.array(1., np.float32)):
            env = SyntheticRewardEnv(reward)
            with self.subTest(reward=reward), self.assertRaises((TypeError, ValueError)):
                MOScalarizationWrapper(env, [.5, .5]).step(np.array([0, 0]))
            self.assertEqual(env.calls, 1)
        with self.assertRaises(ValueError):
            MOScalarizationWrapper(SyntheticRewardEnv(), [.5, .5], [1e-40, 1]).step(np.array([0, 0]))

    # Params: self; returns: None.
    def test_double_scalarization_is_rejected_even_through_an_intervening_wrapper(self):
        for wrapped in (MOScalarizationWrapper(SyntheticRewardEnv(), [.5, .5]),
                        LinearScalarizationWrapper(SyntheticRewardEnv(), [.5, .5])):
            with self.assertRaises(ValueError):
                MOScalarizationWrapper(gym.Wrapper(wrapped), [.5, .5])

    # Params: self; returns: None.
    def test_identity_scaling_matches_the_live_legacy_wrapper_exactly(self):
        case = CASES["slew_default"]
        legacy = LinearScalarizationWrapper(MinimalSSAEnv(**case["config"]), [.85, .15])
        shared = MOScalarizationWrapper(MinimalSSAEnv(**case["config"]), [.85, .15])
        self.addCleanup(legacy.close)
        self.addCleanup(shared.close)
        obs, info = legacy.reset(seed=case["seed"])
        other_obs, other_info = shared.reset(seed=case["seed"])
        np.testing.assert_array_equal(obs, other_obs)
        for action in case["actions"]:
            obs, reward, done, truncated, info = legacy.step(np.array(action))
            other_obs, other_reward, other_done, other_truncated, other_info = shared.step(np.array(action))
            np.testing.assert_array_equal(obs, other_obs)
            self.assertEqual((reward, done, truncated), (other_reward, other_done, other_truncated))
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, other_info[key])
                else:
                    self.assertEqual(value, other_info[key])

    # Params: self; returns: None.
    def test_combined_metadata_diagnostics_and_catalogue_discoveries(self):
        case = CASES["slew_default"]
        env = MinimalSSAEnv(**case["config"])
        self.addCleanup(env.close)
        self.assertEqual(env.reward_dim, 2)
        self.assertEqual(env.reward_order, ("tracking", "search"))
        self.assertEqual(env.reward_space.shape, (2,))
        self.assertEqual(env.reward_space.dtype, np.float32)
        self.assertTrue(np.isneginf(env.reward_space.low).all())
        self.assertTrue(np.isposinf(env.reward_space.high).all())
        env.reset(seed=case["seed"])
        for action in case["actions"]:
            previous_ids = set(env.catalogue.tracks)
            _, reward, _, _, info = env.step(np.array(action))
            self.assertTrue(env.reward_space.contains(reward))
            self.assertTrue(np.isfinite(reward).all())
            self.assertEqual(info["coverage_mean"], float(np.mean(env.catalogue.get_observation_grid_copy())))
            self.assertEqual(info["new_detections"], len(set(env.catalogue.tracks) - previous_ids))
            self.assertIs(type(info["new_detections"]), int)

    # Params: self; returns: None.
    def test_scalar_track_and_search_modes_match_prechange_baselines(self):
        for mode, case in SCALAR_MODE_BASELINE.items():
            with self.subTest(mode=mode):
                env = MinimalSSAEnv(**case["config"])
                self.addCleanup(env.close)
                self.assertFalse(hasattr(env, "reward_space"))
                self.assertFalse(hasattr(env, "reward_dim"))
                obs, info = env.reset(seed=case["seed"])
                self.assertEqual(scalar_fingerprint(obs, info), case["initial"])
                for action, expected in zip(case["actions"], case["steps"]):
                    obs, reward, done, truncated, info = env.step(np.array(action))
                    self.assertEqual(np.asarray(reward).shape, ())
                    self.assertEqual(float(reward), expected["reward"])
                    self.assertEqual(scalar_fingerprint(obs, info), expected["fingerprint"])
                    self.assertEqual((bool(done), bool(truncated)), (expected["terminated"], expected["truncated"]))
                    self.assertNotIn("coverage_mean", info)
                    self.assertNotIn("new_detections", info)
                    for key in ("elapsed_steps", "total_steps", "observed_target_ids"):
                        self.assertEqual(info[key], expected[key])

    # Params: self; returns: None.
    def test_sb3_environment_check_uses_scalarized_combined_environment(self):
        from stable_baselines3.common.env_checker import check_env
        env = MOScalarizationWrapper(MinimalSSAEnv(**CASES["no_slew_termination"]["config"]), [.85, .15])
        self.addCleanup(env.close)
        check_env(env, warn=True)



def numerical_thread_snapshot():
    """Read installed pool sizes without changing them or installing packages."""
    try:
        from threadpoolctl import threadpool_info
    except ModuleNotFoundError:
        import ctypes
        site = Path(np.__file__).resolve().parent.parent
        paths = [path for name in ("numpy.libs", "scipy.libs")
                 for path in (site / name).glob("*.dll") if "openblas" in path.name.lower()]
        counts = []
        for path in paths:
            library = ctypes.CDLL(str(path))
            for symbol in ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
                           "openblas_get_num_threads", "openblas_get_num_threads64_",
                           "openblas_get_num_threads_64_"):
                try:
                    getter = getattr(library, symbol)
                except AttributeError:
                    continue
                getter.restype = ctypes.c_int
                getter.argtypes = []
                counts.append((str(path), getter()))
                break
            else:
                raise RuntimeError("No read-only thread-count API found for " + str(path))
        if len(counts) != 2:
            raise RuntimeError("Expected the installed NumPy and SciPy OpenBLAS libraries")
        return sorted(counts)
    return sorted((row["filepath"], row["num_threads"]) for row in threadpool_info())


class SharedEnvironmentFactories(unittest.TestCase):
    # Params: self, parallel; returns: ExperimentConfig.
    def config(self, parallel=False):
        return ExperimentConfig(env=EnvConfig(episode_steps=3, slew_alpha=0.),
                                execution=ExecutionConfig(subproc=parallel, worker_threads=1),
                                eval_episodes=3, final_episodes=4)

    # Params: self; returns: None.
    def test_raw_factory_forwards_physical_settings_and_only_resets_with_seed(self):
        cfg = EnvConfig(known_targets=2, unknown_targets=3, grid_shape=(7, 9), episode_steps=11,
                        measurement_noise_std=.7, dwell_steps=3, slew_alpha=.4,
                        uncertainty_norm_scale=17.)
        fake = Mock()
        with patch.object(mo_envs, "MinimalSSAEnv", return_value=fake) as constructor:
            self.assertIs(mo_envs.make_raw_env(cfg, None), fake)
            fake.reset.assert_not_called()
            mo_envs.make_raw_env(cfg, 123)
            fake.reset.assert_called_once_with(seed=123)
            self.assertEqual(constructor.call_args.kwargs, dict(
                known_targets=2, unknown_targets=3, grid_shape=(7, 9), timesteps=11,
                measurement_noise_std=.7, zero_noise=False, reward_type="R6", search_reward="R2",
                dwell_steps=3, slew_alpha=.4, uncertainty_norm_scale=17., mode="combined", simple_obs=False))
        raw = mo_envs.make_raw_env(cfg, 123)
        self.addCleanup(raw.close)
        self.assertEqual(raw.step(np.array([3, 4]))[1].shape, (2,))

    # Params: self; returns: None.
    def test_legacy_names_are_reexports_and_factory_keeps_identity_behavior(self):
        from agents.train import train_ppo_combined as legacy
        self.assertIs(legacy.LinearScalarizationWrapper, MOScalarizationWrapper)
        self.assertIs(legacy.TrainingSubprocVecEnv, mo_envs.TrainingSubprocVecEnv)
        self.assertIs(legacy.worker_thread_environment, mo_envs.worker_thread_environment)
        wrapped = legacy.create_combined_env(legacy.build_parser().parse_args([]), 123)
        self.addCleanup(wrapped.close)
        self.assertEqual(wrapped.weights.tolist(), np.asarray([.85, .15], np.float32).tolist())
        np.testing.assert_array_equal(wrapped.scales, [1, 1])

    # Params: self; returns: None.
    def test_factory_rank_seed_and_direct_main_call_do_not_limit_learner(self):
        import torch
        config = self.config()
        plan = make_seed_plan(config, 0)
        before = torch.get_num_threads()
        factory = mo_envs.make_env_factory(config, [.85, .15], 3, plan.base_worker_seed, 1)
        env = factory()
        reference = mo_envs.make_raw_env(config.env, plan.worker_seeds[3])
        self.addCleanup(env.close)
        self.addCleanup(reference.close)
        np.testing.assert_array_equal(env.unwrapped._get_true_states_array(), reference._get_true_states_array())
        self.assertEqual(env.unwrapped.np_random_seed, plan.worker_seeds[3])
        self.assertEqual(torch.get_num_threads(), before)

    # Params: self; returns: None.
    def test_thread_environment_restores_existing_and_missing_values_on_failure(self):
        keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        old = {key: os.environ.get(key) for key in keys}
        with mo_envs.worker_thread_environment(1):
            self.assertTrue(all(os.environ[key] == "1" for key in keys))
        self.assertEqual({key: os.environ.get(key) for key in keys}, old)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with mo_envs.worker_thread_environment(2):
                raise RuntimeError("injected")
        self.assertEqual({key: os.environ.get(key) for key in keys}, old)

    # Params: self; returns: None.
    def test_sequential_stack_has_one_monitor_and_scalar_training_rewards(self):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from stable_baselines3.common.monitor import Monitor
        config = self.config()
        with mo_envs.make_train_vec_env(config, [.85, .15], make_seed_plan(config, 0)) as handle:
            self.assertIsInstance(handle.vec_env, VecMonitor)
            self.assertIsInstance(handle.raw_vec_env, DummyVecEnv)
            self.assertIs(handle.vec_env.venv, handle.raw_vec_env)
            self.assertIsNone(handle.vec_normalize)
            self.assertEqual(len(handle.raw_vec_env.envs), 8)
            for env in handle.raw_vec_env.envs:
                self.assertIsInstance(env, MOScalarizationWrapper)
                self.assertNotIsInstance(env.env, Monitor)
            handle.vec_env.reset()
            self.assertEqual(handle.vec_env.step(np.tile([5, 5], (8, 1)))[1].shape, (8,))
        handle.close()

    # Params: self; returns: None.
    def test_failures_close_partial_raw_and_sequential_children(self):
        raw = Mock()
        raw.reset.side_effect = RuntimeError("reset failed")
        with patch.object(mo_envs, "MinimalSSAEnv", return_value=raw), self.assertRaises(RuntimeError):
            mo_envs.make_raw_env(EnvConfig(), 123)
        raw.close.assert_called_once_with()
        raw = Mock()
        with patch.object(mo_envs, "make_raw_env", return_value=raw), \
             patch.object(mo_envs, "LinearScalarizationWrapper", side_effect=RuntimeError("wrap failed")), \
             self.assertRaises(RuntimeError):
            mo_envs.make_scalar_env(EnvConfig(), [.5, .5], 123)
        raw.close.assert_called_once_with()
        child = SyntheticRewardEnv()
        child.close = Mock()
        # Params: none; returns: never (raises).
        def fail():
            raise RuntimeError("second factory failed")
        with self.assertRaises(RuntimeError):
            mo_envs._make_dummy_vec_env([lambda: child, fail])
        child.close.assert_called_once_with()

    # Params: self; returns: None.
    def test_monitor_failure_closes_registered_vector_environment(self):
        original = mo_envs._make_dummy_vec_env
        created = []
        # Params: factories; returns: raw.
        def tracked(factories):
            raw = original(factories)
            raw.close = Mock(wraps=raw.close)
            created.append(raw)
            return raw
        config = self.config()
        with patch.object(mo_envs, "_make_dummy_vec_env", side_effect=tracked), \
             patch.object(mo_envs, "VecMonitor", side_effect=RuntimeError("monitor failed")), \
             self.assertRaises(RuntimeError):
            mo_envs.make_train_vec_env(config, [.5, .5], make_seed_plan(config, 0))
        created[0].close.assert_called_once_with()

    # Params: self; returns: None.
    def test_invalid_plans_fail_before_allocation(self):
        config = self.config()
        plan = make_seed_plan(config, 0)
        with patch.object(mo_envs, "_make_dummy_vec_env") as allocate:
            with self.assertRaises(ValueError):
                mo_envs.make_train_vec_env(config, [.5, .5], replace(plan, run_seed=999))
            allocate.assert_not_called()
        with self.assertRaises(ValueError):
            mo_envs.make_env_factory(config, [.5, .5], 8, plan.base_worker_seed)
        with self.assertRaises(ValueError):
            mo_envs.make_env_factory(config, [.5, .5], 0, 2**32 - 1)

    # Params: self; returns: None.
    def test_partial_spawn_startup_releases_processes_and_pipes(self):
        from stable_baselines3.common.vec_env import SubprocVecEnv
        process, remote = Mock(), Mock()
        process.pid = 123
        process.is_alive.return_value = True
        # Params: instance, *args, **kwargs; returns: never (raises).
        def fail_start(instance, *args, **kwargs):
            instance.processes = [process]
            instance.remotes = [remote]
            instance.work_remotes = []
            raise RuntimeError("partial start failure")
        with patch.object(SubprocVecEnv, "__init__", fail_start), self.assertRaises(RuntimeError):
            mo_envs.TrainingSubprocVecEnv([])
        process.terminate.assert_called_once_with()
        process.join.assert_called_once_with(timeout=5)
        remote.close.assert_called_once_with()

    # Params: self; returns: None.
    def test_eight_spawn_workers_match_sequential_reset_and_repeat_with_worker_only_limits(self):
        import multiprocessing as mp
        import torch
        config = self.config()
        plan = make_seed_plan(config, 0)
        thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                       "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        parent_env = {key: os.environ.get(key) for key in thread_keys}
        parent_torch = torch.get_num_threads()
        parent_pools = numerical_thread_snapshot()
        parent_pid = os.getpid()
        original = mo_envs.make_env_factory
        # Params: *args, **kwargs; returns: probe.
        def probe_factory(*args, **kwargs):
            factory = original(*args, **kwargs)
            # Params: none; returns: env.
            def probe():
                env = factory()
                import torch
                env._block12_runtime = {"pid": os.getpid(), "torch_threads": torch.get_num_threads(),
                    "limits": {key: os.environ.get(key) for key in thread_keys},
                    "pools": numerical_thread_snapshot(),
                    "seed": env.unwrapped.np_random_seed}
                return env
            return probe
        # Params: parallel, broken; returns: (initial, trace, states).
        def collect(parallel, broken=False):
            selected = replace(config, execution=ExecutionConfig(subproc=parallel, worker_threads=1))
            with patch.object(mo_envs, "make_env_factory", side_effect=probe_factory):
                handle = mo_envs.make_train_vec_env(selected, [.85, .15], plan)
            workers = tuple(handle.workers)
            try:
                states = handle.vec_env.get_attr("_block12_runtime")
                initial = handle.vec_env.reset().copy()
                trace = []
                for action in ([5, 5], [5, 6], [0, 0], [9, 9], [0, 9], [9, 0]):
                    obs, reward, done, infos = handle.vec_env.step(np.tile(action, (8, 1)))
                    trace.append((obs.copy(), reward.copy(), done.copy(), infos))
                if broken:
                    workers[0].terminate()
                    workers[0].join(timeout=5)
            finally:
                handle.close()
                handle.close()
            for worker in workers:
                self.assertFalse(worker.is_alive())
            if parallel:
                self.assertTrue(all(remote.closed for remote in handle.raw_vec_env.remotes))
                self.assertFalse({worker.pid for worker in workers} & {child.pid for child in mp.active_children()})
            return initial, trace, states
        sequential = collect(False)
        parallel = collect(True, broken=True)
        repeat = collect(True)
        self.assertEqual(len({state["pid"] for state in parallel[2]}), 8)
        for rank, state in enumerate(parallel[2]):
            self.assertNotEqual(state["pid"], parent_pid)
            self.assertEqual(state["seed"], plan.worker_seeds[rank])
            self.assertEqual(state["torch_threads"], 1)
            self.assertTrue(all(value == "1" for value in state["limits"].values()))
            self.assertTrue(all(threads == 1 for _, threads in state["pools"]))
        for other in (parallel, repeat):
            np.testing.assert_array_equal(sequential[0], other[0])
            for left, right in zip(sequential[1], other[1]):
                for a, b in zip(left[:3], right[:3]):
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
                for a, b in zip(left[3], right[3]):
                    for key in ("true_states", "estimated_states", "sensor_pointing", "vector_reward_raw",
                                "vector_reward_scaled", "coverage_mean"):
                        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-7)
                    for key in ("elapsed_steps", "total_steps", "observed_target_ids", "new_detections"):
                        self.assertEqual(a[key], b[key])
                    if "terminal_observation" in a:
                        np.testing.assert_allclose(a["terminal_observation"], b["terminal_observation"], rtol=1e-6, atol=1e-7)
            for left, right in zip(parallel[1], repeat[1]):
                for a, b in zip(left[:3], right[:3]):
                    np.testing.assert_array_equal(a, b)
        self.assertTrue(sequential[1][2][2].all())
        self.assertTrue(sequential[1][5][2].all())
        self.assertEqual({key: os.environ.get(key) for key in thread_keys}, parent_env)
        self.assertEqual(torch.get_num_threads(), parent_torch)
        self.assertEqual(numerical_thread_snapshot(), parent_pools)
        print("Block 12 pools:", {"parent_torch": parent_torch, "parent_blas": parent_pools,
                                  "worker_torch": parallel[2][0]["torch_threads"],
                                  "worker_blas": parallel[2][0]["pools"]})



class PerPolicyNormalization(unittest.TestCase):
    # Params: self, norm_obs, norm_reward, parallel, n_envs; returns: ExperimentConfig.
    def config(self, norm_obs=True, norm_reward=True, parallel=False, n_envs=2):
        from agents.mo import PPOConfig
        return ExperimentConfig(
            env=EnvConfig(episode_steps=3, slew_alpha=0.),
            ppo=PPOConfig(n_envs=n_envs, gamma=.9),
            execution=ExecutionConfig(subproc=parallel, worker_threads=1),
            normalization=NormalizationConfig(norm_obs=norm_obs, norm_reward=norm_reward,
                                               epsilon=1e-6, clip_obs=1.5, clip_reward=2.5),
            eval_episodes=3, final_episodes=4,
        )

    # Params: self; returns: None.
    def test_enabled_flags_and_all_normalization_parameters_are_explicit(self):
        from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
        for norm_obs, norm_reward in ((True, False), (False, True), (True, True)):
            config = self.config(norm_obs, norm_reward)
            with self.subTest(norm_obs=norm_obs, norm_reward=norm_reward), \
                 patch.object(mo_envs, "VecNormalize", wraps=VecNormalize) as constructor, \
                 mo_envs.make_train_vec_env(config, [.85, .15], make_seed_plan(config, 0)) as handle:
                self.assertIs(handle.vec_env, handle.vec_normalize)
                self.assertIsInstance(handle.vec_env, VecNormalize)
                self.assertIsInstance(handle.vec_env.venv, VecMonitor)
                self.assertIs(handle.vec_env.venv.venv, handle.raw_vec_env)
                self.assertEqual(constructor.call_args.kwargs, dict(training=True,
                    norm_obs=norm_obs, norm_reward=norm_reward, gamma=.9, epsilon=1e-6,
                    clip_obs=1.5, clip_reward=2.5, norm_obs_keys=None))
                for name, expected in (("training", True), ("norm_obs", norm_obs), ("norm_reward", norm_reward),
                                       ("gamma", .9), ("epsilon", 1e-6), ("clip_obs", 1.5), ("clip_reward", 2.5)):
                    self.assertEqual(getattr(handle.vec_normalize, name), expected)

    # Params: self; returns: None.
    def test_two_policies_own_independent_rms_and_return_buffers(self):
        config = self.config()
        plan = make_seed_plan(config, 0)
        with mo_envs.make_train_vec_env(config, [.85, .15], plan) as first, \
             mo_envs.make_train_vec_env(config, [.15, .85], plan) as second:
            a, b = first.vec_normalize, second.vec_normalize
            self.assertIsNot(a.obs_rms, b.obs_rms)
            self.assertIsNot(a.ret_rms, b.ret_rms)
            self.assertFalse(np.shares_memory(a.obs_rms.mean, b.obs_rms.mean))
            self.assertFalse(np.shares_memory(a.obs_rms.var, b.obs_rms.var))
            self.assertFalse(np.shares_memory(a.returns, b.returns))
            second_count = (b.obs_rms.count, b.ret_rms.count)
            second_mean = b.obs_rms.mean.copy()
            first.vec_env.reset()
            first.vec_env.step(np.tile([5, 5], (2, 1)))
            self.assertGreater(a.obs_rms.count, b.obs_rms.count)
            self.assertGreater(a.ret_rms.count, b.ret_rms.count)
            self.assertEqual((b.obs_rms.count, b.ret_rms.count), second_count)
            np.testing.assert_array_equal(b.obs_rms.mean, second_mean)
            a.obs_rms.mean.flat[0] = 123
            a.ret_rms.mean = np.float64(123)
            self.assertNotEqual(b.obs_rms.mean.flat[0], 123)
            self.assertNotEqual(float(b.ret_rms.mean), 123)

    # Params: self; returns: None.
    def test_disabled_normalization_allocates_no_normalizer_and_preserves_values(self):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        config = self.config(False, False)
        plan = make_seed_plan(config, 0)
        factories = [mo_envs.make_env_factory(config, [.85, .15], rank, plan.base_worker_seed)
                     for rank in range(2)]
        reference = VecMonitor(DummyVecEnv(factories))
        reference.seed(plan.base_worker_seed)
        self.addCleanup(reference.close)
        with patch.object(mo_envs, "VecNormalize") as constructor, \
             mo_envs.make_train_vec_env(config, [.85, .15], plan) as handle:
            constructor.assert_not_called()
            self.assertIsNone(handle.vec_normalize)
            np.testing.assert_array_equal(handle.vec_env.reset(), reference.reset())
            for action in ([5, 5], [5, 6], [0, 0], [9, 9]):
                actions = np.tile(action, (2, 1))
                actual, expected = handle.vec_env.step(actions), reference.step(actions)
                for a, b in zip(actual[:3], expected[:3]):
                    np.testing.assert_array_equal(a, b)
                for a, b in zip(actual[3], expected[3]):
                    np.testing.assert_array_equal(a["vector_reward_raw"], b["vector_reward_raw"])
                    np.testing.assert_array_equal(a["true_states"], b["true_states"])

    # Params: self; returns: None.
    def test_normalized_outputs_preserve_raw_objective_info_and_scalar_original_reward(self):
        for norm_obs, norm_reward in ((True, False), (False, True), (True, True)):
            config = self.config(norm_obs, norm_reward)
            with self.subTest(norm_obs=norm_obs, norm_reward=norm_reward), \
                 mo_envs.make_train_vec_env(config, [.85, .15], make_seed_plan(config, 0)) as handle:
                normalizer = handle.vec_normalize
                obs = handle.vec_env.reset()
                self.assertEqual(obs.shape, (2, 10, 10, 3))
                self.assertEqual(obs.dtype, np.float32)
                self.assertTrue(np.isfinite(obs).all())
                if not norm_obs:
                    np.testing.assert_array_equal(obs, normalizer.get_original_obs())
                for action in ([5, 5], [5, 6], [0, 0]):
                    obs, reward, done, info = handle.vec_env.step(np.tile(action, (2, 1)))
                    raw_scalar = np.array([item["scalarized_reward"] for item in info], np.float32)
                    np.testing.assert_array_equal(normalizer.get_original_reward(), raw_scalar)
                    if not norm_reward:
                        np.testing.assert_array_equal(reward, raw_scalar)
                    else:
                        self.assertTrue(np.all(np.abs(reward) <= config.normalization.clip_reward))
                    if norm_obs:
                        self.assertTrue(np.all(np.abs(obs) <= config.normalization.clip_obs))
                    for item in info:
                        np.testing.assert_array_equal(item["vector_reward"], item["vector_reward_raw"])
                        np.testing.assert_array_equal(item["vector_reward_raw"], item["vector_reward_scaled"])
                        self.assertEqual(item["tracking_reward"], float(item["vector_reward_raw"][0]))
                        self.assertEqual(item["search_reward"], float(item["vector_reward_raw"][1]))
                        if "terminal_observation" in item and norm_obs:
                            self.assertTrue(np.all(np.abs(item["terminal_observation"]) <= config.normalization.clip_obs))
                self.assertTrue(done.all())

    # Params: self, created; returns: factory.
    def track_raw(self, created):
        original = mo_envs._make_dummy_vec_env
        # Params: factories; returns: raw.
        def factory(factories):
            raw = original(factories)
            raw.close = Mock(wraps=raw.close)
            created.append(raw)
            return raw
        return factory

    # Params: self; returns: None.
    def test_partially_initialized_normalizer_failure_closes_registered_raw(self):
        from stable_baselines3.common.vec_env import VecNormalize
        config = self.config()
        created = []
        with patch.object(mo_envs, "_make_dummy_vec_env", side_effect=self.track_raw(created)), \
             patch.object(VecNormalize, "_sanity_checks", side_effect=RuntimeError("partial normalizer failure")), \
             self.assertRaisesRegex(RuntimeError, "partial normalizer failure"):
            mo_envs.make_train_vec_env(config, [.85, .15], make_seed_plan(config, 0))
        self.assertEqual(len(created), 1)
        created[0].close.assert_called_once_with()

    # Params: self; returns: None.
    def test_failure_after_normalizer_registration_closes_both_layers_once(self):
        from stable_baselines3.common.vec_env import VecNormalize
        config = self.config()
        raw, normalized = [], []
        # Params: *args, **kwargs; returns: normalizer.
        def create(*args, **kwargs):
            normalizer = VecNormalize(*args, **kwargs)
            normalizer.close = Mock(wraps=normalizer.close)
            normalized.append(normalizer)
            return normalizer
        with patch.object(mo_envs, "_make_dummy_vec_env", side_effect=self.track_raw(raw)), \
             patch.object(mo_envs, "VecNormalize", side_effect=create), \
             patch.object(VecNormalize, "seed", side_effect=RuntimeError("seed failure")), \
             self.assertRaisesRegex(RuntimeError, "seed failure"):
            mo_envs.make_train_vec_env(config, [.85, .15], make_seed_plan(config, 0))
        normalized[0].close.assert_called_once_with()
        raw[0].close.assert_called_once_with()

    # Params: self; returns: None.
    def test_normal_close_is_idempotent_through_normalization_chain(self):
        config = self.config()
        handle = mo_envs.make_train_vec_env(config, [.85, .15], make_seed_plan(config, 0))
        raw, normalizer = handle.raw_vec_env, handle.vec_normalize
        raw.close = Mock(wraps=raw.close)
        normalizer.close = Mock(wraps=normalizer.close)
        with handle:
            handle.vec_env.reset()
        handle.close()
        self.assertTrue(handle.closed)
        normalizer.close.assert_called_once_with()
        raw.close.assert_called_once_with()

    # Params: self; returns: None.
    def test_eight_normalized_spawn_workers_reset_and_close_and_failure_leaves_none(self):
        import multiprocessing as mp
        config = self.config(parallel=True, n_envs=8)
        plan = make_seed_plan(config, 0)
        handle = mo_envs.make_train_vec_env(config, [.85, .15], plan)
        workers = tuple(handle.workers)
        try:
            self.assertEqual(len(workers), 8)
            self.assertTrue(all(worker.is_alive() for worker in workers))
            obs = handle.vec_env.reset()
            self.assertTrue(np.isfinite(obs).all())
            for action in ([5, 5], [5, 6], [0, 0]):
                obs, reward, done, info = handle.vec_env.step(np.tile(action, (8, 1)))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward).all())
            self.assertTrue(done.all())
        finally:
            handle.close()
            handle.close()
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        # Inject a construction failure after another real spawn, then prove
        # the already-registered worker environment has been closed.
        original = mo_envs.TrainingSubprocVecEnv
        created = []
        # Params: *args, **kwargs; returns: raw.
        def create(*args, **kwargs):
            raw = original(*args, **kwargs)
            created.append(raw)
            return raw
        with patch.object(mo_envs, "TrainingSubprocVecEnv", side_effect=create), \
             patch.object(mo_envs, "VecNormalize", side_effect=RuntimeError("normalizer failed after spawn")), \
             self.assertRaisesRegex(RuntimeError, "normalizer failed after spawn"):
            mo_envs.make_train_vec_env(config, [.85, .15], plan)
        self.assertTrue(all(not worker.is_alive() for worker in created[0].processes))
        self.assertTrue(all(remote.closed for remote in created[0].remotes))
        owned_pids = {worker.pid for worker in workers + tuple(created[0].processes)}
        self.assertFalse(owned_pids & {child.pid for child in mp.active_children()})


if __name__ == "__main__":
    unittest.main()
