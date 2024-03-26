import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/rasp/most_freq/vocab8maxlen8dvar100/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/most_freq_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 5, 6}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 87
        elif q_position in {2, 4}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {81, 7}:
            return k_position == 6
        elif q_position in {8, 77}:
            return k_position == 77
        elif q_position in {40, 9, 71}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 62
        elif q_position in {11}:
            return k_position == 78
        elif q_position in {12}:
            return k_position == 43
        elif q_position in {13}:
            return k_position == 56
        elif q_position in {14}:
            return k_position == 40
        elif q_position in {62, 15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 14
        elif q_position in {17, 68}:
            return k_position == 48
        elif q_position in {18}:
            return k_position == 30
        elif q_position in {19}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 86
        elif q_position in {21}:
            return k_position == 58
        elif q_position in {22}:
            return k_position == 70
        elif q_position in {75, 29, 23}:
            return k_position == 76
        elif q_position in {24}:
            return k_position == 11
        elif q_position in {25}:
            return k_position == 46
        elif q_position in {26, 55}:
            return k_position == 13
        elif q_position in {65, 27}:
            return k_position == 23
        elif q_position in {28}:
            return k_position == 93
        elif q_position in {30}:
            return k_position == 88
        elif q_position in {64, 31}:
            return k_position == 9
        elif q_position in {32, 43, 92}:
            return k_position == 81
        elif q_position in {33}:
            return k_position == 99
        elif q_position in {34, 95}:
            return k_position == 47
        elif q_position in {35}:
            return k_position == 41
        elif q_position in {36}:
            return k_position == 28
        elif q_position in {37}:
            return k_position == 90
        elif q_position in {74, 38}:
            return k_position == 44
        elif q_position in {39}:
            return k_position == 42
        elif q_position in {41, 52}:
            return k_position == 94
        elif q_position in {42, 53}:
            return k_position == 19
        elif q_position in {44, 63}:
            return k_position == 82
        elif q_position in {45}:
            return k_position == 38
        elif q_position in {46}:
            return k_position == 92
        elif q_position in {47}:
            return k_position == 68
        elif q_position in {48}:
            return k_position == 15
        elif q_position in {49}:
            return k_position == 45
        elif q_position in {89, 50}:
            return k_position == 33
        elif q_position in {51}:
            return k_position == 80
        elif q_position in {97, 54}:
            return k_position == 36
        elif q_position in {56}:
            return k_position == 75
        elif q_position in {57, 82}:
            return k_position == 66
        elif q_position in {58, 78}:
            return k_position == 22
        elif q_position in {59}:
            return k_position == 74
        elif q_position in {96, 66, 60}:
            return k_position == 24
        elif q_position in {88, 61}:
            return k_position == 57
        elif q_position in {67}:
            return k_position == 53
        elif q_position in {69}:
            return k_position == 29
        elif q_position in {93, 70}:
            return k_position == 31
        elif q_position in {72}:
            return k_position == 34
        elif q_position in {73}:
            return k_position == 50
        elif q_position in {76}:
            return k_position == 95
        elif q_position in {87, 79}:
            return k_position == 61
        elif q_position in {80}:
            return k_position == 91
        elif q_position in {83}:
            return k_position == 37
        elif q_position in {84}:
            return k_position == 60
        elif q_position in {85}:
            return k_position == 12
        elif q_position in {86}:
            return k_position == 96
        elif q_position in {90}:
            return k_position == 72
        elif q_position in {91}:
            return k_position == 73
        elif q_position in {94}:
            return k_position == 64
        elif q_position in {98}:
            return k_position == 84
        elif q_position in {99}:
            return k_position == 59

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2, 3, 6}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 96
        elif q_position in {9}:
            return k_position == 54
        elif q_position in {10, 14}:
            return k_position == 85
        elif q_position in {11}:
            return k_position == 84
        elif q_position in {12, 62}:
            return k_position == 94
        elif q_position in {13}:
            return k_position == 55
        elif q_position in {36, 15}:
            return k_position == 80
        elif q_position in {16, 38}:
            return k_position == 71
        elif q_position in {17, 93}:
            return k_position == 23
        elif q_position in {18, 47}:
            return k_position == 46
        elif q_position in {91, 19, 79}:
            return k_position == 68
        elif q_position in {20, 71}:
            return k_position == 15
        elif q_position in {48, 21}:
            return k_position == 33
        elif q_position in {57, 22}:
            return k_position == 17
        elif q_position in {23}:
            return k_position == 58
        elif q_position in {24, 30}:
            return k_position == 77
        elif q_position in {25}:
            return k_position == 20
        elif q_position in {97, 26}:
            return k_position == 10
        elif q_position in {27}:
            return k_position == 72
        elif q_position in {28}:
            return k_position == 79
        elif q_position in {68, 29}:
            return k_position == 50
        elif q_position in {88, 31}:
            return k_position == 36
        elif q_position in {32}:
            return k_position == 65
        elif q_position in {40, 33, 96}:
            return k_position == 69
        elif q_position in {34, 39}:
            return k_position == 83
        elif q_position in {35}:
            return k_position == 18
        elif q_position in {37, 78}:
            return k_position == 48
        elif q_position in {41}:
            return k_position == 14
        elif q_position in {42, 90}:
            return k_position == 59
        elif q_position in {43}:
            return k_position == 32
        elif q_position in {44}:
            return k_position == 47
        elif q_position in {45}:
            return k_position == 53
        elif q_position in {46}:
            return k_position == 64
        elif q_position in {49}:
            return k_position == 27
        elif q_position in {50}:
            return k_position == 28
        elif q_position in {51}:
            return k_position == 61
        elif q_position in {52}:
            return k_position == 60
        elif q_position in {53}:
            return k_position == 51
        elif q_position in {54}:
            return k_position == 38
        elif q_position in {55}:
            return k_position == 89
        elif q_position in {56}:
            return k_position == 44
        elif q_position in {58}:
            return k_position == 42
        elif q_position in {59}:
            return k_position == 30
        elif q_position in {60}:
            return k_position == 82
        elif q_position in {84, 61}:
            return k_position == 92
        elif q_position in {63}:
            return k_position == 56
        elif q_position in {64}:
            return k_position == 63
        elif q_position in {65, 92}:
            return k_position == 78
        elif q_position in {80, 66, 98}:
            return k_position == 16
        elif q_position in {67, 94}:
            return k_position == 40
        elif q_position in {69}:
            return k_position == 29
        elif q_position in {81, 70}:
            return k_position == 81
        elif q_position in {72}:
            return k_position == 11
        elif q_position in {73}:
            return k_position == 75
        elif q_position in {74, 86, 87}:
            return k_position == 86
        elif q_position in {75}:
            return k_position == 66
        elif q_position in {76}:
            return k_position == 87
        elif q_position in {77}:
            return k_position == 37
        elif q_position in {82}:
            return k_position == 57
        elif q_position in {83}:
            return k_position == 62
        elif q_position in {85}:
            return k_position == 73
        elif q_position in {89}:
            return k_position == 70
        elif q_position in {95}:
            return k_position == 97
        elif q_position in {99}:
            return k_position == 21

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2, 4}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 70
        elif q_position in {56, 9}:
            return k_position == 21
        elif q_position in {10, 58, 82}:
            return k_position == 96
        elif q_position in {18, 11, 14, 95}:
            return k_position == 79
        elif q_position in {12, 63}:
            return k_position == 19
        elif q_position in {49, 13}:
            return k_position == 22
        elif q_position in {35, 60, 15}:
            return k_position == 78
        elif q_position in {16, 94}:
            return k_position == 85
        elif q_position in {17}:
            return k_position == 38
        elif q_position in {64, 72, 19, 86}:
            return k_position == 48
        elif q_position in {50, 20}:
            return k_position == 53
        elif q_position in {42, 21, 38}:
            return k_position == 95
        elif q_position in {27, 22}:
            return k_position == 36
        elif q_position in {34, 23}:
            return k_position == 13
        elif q_position in {24, 68}:
            return k_position == 83
        elif q_position in {25}:
            return k_position == 43
        elif q_position in {73, 26}:
            return k_position == 74
        elif q_position in {97, 28}:
            return k_position == 63
        elif q_position in {88, 29}:
            return k_position == 58
        elif q_position in {74, 52, 30}:
            return k_position == 66
        elif q_position in {31}:
            return k_position == 15
        elif q_position in {32}:
            return k_position == 32
        elif q_position in {33, 39}:
            return k_position == 28
        elif q_position in {36}:
            return k_position == 89
        elif q_position in {37}:
            return k_position == 86
        elif q_position in {40}:
            return k_position == 76
        elif q_position in {41, 76}:
            return k_position == 10
        elif q_position in {43}:
            return k_position == 9
        elif q_position in {44, 46}:
            return k_position == 62
        elif q_position in {45}:
            return k_position == 82
        elif q_position in {47}:
            return k_position == 67
        elif q_position in {48}:
            return k_position == 45
        elif q_position in {51}:
            return k_position == 52
        elif q_position in {53}:
            return k_position == 30
        elif q_position in {80, 54}:
            return k_position == 88
        elif q_position in {89, 55}:
            return k_position == 37
        elif q_position in {96, 57}:
            return k_position == 97
        elif q_position in {59, 77}:
            return k_position == 47
        elif q_position in {61, 70}:
            return k_position == 57
        elif q_position in {75, 62}:
            return k_position == 94
        elif q_position in {65}:
            return k_position == 65
        elif q_position in {66}:
            return k_position == 91
        elif q_position in {67}:
            return k_position == 14
        elif q_position in {69}:
            return k_position == 23
        elif q_position in {71}:
            return k_position == 68
        elif q_position in {78}:
            return k_position == 61
        elif q_position in {79}:
            return k_position == 77
        elif q_position in {81}:
            return k_position == 72
        elif q_position in {98, 83}:
            return k_position == 44
        elif q_position in {84}:
            return k_position == 40
        elif q_position in {85}:
            return k_position == 60
        elif q_position in {87}:
            return k_position == 11
        elif q_position in {90, 91}:
            return k_position == 34
        elif q_position in {92}:
            return k_position == 24
        elif q_position in {93}:
            return k_position == 31
        elif q_position in {99}:
            return k_position == 73

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0}:
            return token == "2"
        elif position in {
            1,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            24,
            25,
            26,
            28,
            29,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
        }:
            return token == "5"
        elif position in {2}:
            return token == "0"
        elif position in {3, 7}:
            return token == "3"
        elif position in {4, 5, 6}:
            return token == "1"
        elif position in {30, 23}:
            return token == ""
        elif position in {27, 77}:
            return token == "4"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 32, 36, 8, 40, 43, 76, 79, 17, 85, 87, 27}:
            return token == "3"
        elif position in {
            1,
            9,
            10,
            11,
            13,
            14,
            15,
            18,
            23,
            25,
            26,
            33,
            37,
            38,
            41,
            47,
            48,
            49,
            50,
            51,
            53,
            58,
            64,
            66,
            67,
            69,
            70,
            72,
            77,
            83,
            92,
            95,
            96,
            98,
        }:
            return token == "2"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            7,
            12,
            16,
            22,
            24,
            28,
            30,
            34,
            35,
            39,
            44,
            46,
            52,
            54,
            55,
            59,
            60,
            62,
            63,
            73,
            75,
            80,
            81,
            82,
            86,
            90,
            91,
            99,
        }:
            return token == ""
        elif position in {
            19,
            20,
            29,
            31,
            42,
            45,
            56,
            57,
            61,
            65,
            68,
            71,
            74,
            78,
            84,
            88,
            89,
            93,
            94,
            97,
        }:
            return token == "4"
        elif position in {21}:
            return token == "<s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {65, 1, 90}:
            return k_position == 81
        elif q_position in {2}:
            return k_position == 64
        elif q_position in {3}:
            return k_position == 17
        elif q_position in {4}:
            return k_position == 35
        elif q_position in {5, 6, 7}:
            return k_position == 4
        elif q_position in {8, 75}:
            return k_position == 56
        elif q_position in {9, 63, 39}:
            return k_position == 12
        elif q_position in {72, 10, 50}:
            return k_position == 24
        elif q_position in {88, 11}:
            return k_position == 32
        elif q_position in {80, 43, 12}:
            return k_position == 16
        elif q_position in {97, 13, 47}:
            return k_position == 8
        elif q_position in {91, 14}:
            return k_position == 42
        elif q_position in {32, 15}:
            return k_position == 92
        elif q_position in {16}:
            return k_position == 80
        elif q_position in {17, 58, 92, 46}:
            return k_position == 31
        elif q_position in {64, 33, 18}:
            return k_position == 67
        elif q_position in {19, 45, 31}:
            return k_position == 71
        elif q_position in {73, 20}:
            return k_position == 9
        elif q_position in {21}:
            return k_position == 72
        elif q_position in {36, 40, 77, 22, 61}:
            return k_position == 76
        elif q_position in {23}:
            return k_position == 82
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25, 99}:
            return k_position == 88
        elif q_position in {26}:
            return k_position == 83
        elif q_position in {35, 27, 78}:
            return k_position == 74
        elif q_position in {28}:
            return k_position == 51
        elif q_position in {66, 44, 29}:
            return k_position == 27
        elif q_position in {30}:
            return k_position == 63
        elif q_position in {34}:
            return k_position == 86
        elif q_position in {81, 37}:
            return k_position == 62
        elif q_position in {38}:
            return k_position == 89
        elif q_position in {41}:
            return k_position == 28
        elif q_position in {42, 83}:
            return k_position == 60
        elif q_position in {48}:
            return k_position == 41
        elif q_position in {49, 94, 87}:
            return k_position == 65
        elif q_position in {51}:
            return k_position == 29
        elif q_position in {52}:
            return k_position == 78
        elif q_position in {53}:
            return k_position == 18
        elif q_position in {85, 54}:
            return k_position == 37
        elif q_position in {55}:
            return k_position == 26
        elif q_position in {56}:
            return k_position == 52
        elif q_position in {57}:
            return k_position == 47
        elif q_position in {59}:
            return k_position == 49
        elif q_position in {60}:
            return k_position == 11
        elif q_position in {62}:
            return k_position == 39
        elif q_position in {67}:
            return k_position == 30
        elif q_position in {68}:
            return k_position == 90
        elif q_position in {69}:
            return k_position == 38
        elif q_position in {70, 79}:
            return k_position == 61
        elif q_position in {71}:
            return k_position == 94
        elif q_position in {74}:
            return k_position == 19
        elif q_position in {76}:
            return k_position == 34
        elif q_position in {82}:
            return k_position == 45
        elif q_position in {96, 84}:
            return k_position == 46
        elif q_position in {86}:
            return k_position == 66
        elif q_position in {89}:
            return k_position == 48
        elif q_position in {93}:
            return k_position == 22
        elif q_position in {95}:
            return k_position == 93
        elif q_position in {98}:
            return k_position == 40

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {
            0,
            11,
            14,
            15,
            18,
            22,
            25,
            28,
            30,
            32,
            35,
            39,
            40,
            48,
            50,
            54,
            57,
            58,
            60,
            64,
            66,
            68,
            71,
            73,
            75,
            79,
            80,
            83,
            89,
            90,
        }:
            return token == "3"
        elif position in {1, 99, 36, 77, 49, 23}:
            return token == "1"
        elif position in {
            2,
            3,
            10,
            13,
            17,
            20,
            31,
            33,
            34,
            42,
            43,
            52,
            55,
            56,
            63,
            67,
            72,
            82,
            84,
            87,
            88,
            91,
            93,
            96,
        }:
            return token == "<s>"
        elif position in {4, 5, 6, 7, 37, 38, 70, 44, 53, 29}:
            return token == ""
        elif position in {8, 74, 76, 46, 26, 27, 92, 61, 94}:
            return token == "4"
        elif position in {65, 69, 9, 12, 78, 47, 16, 81, 21, 24, 59}:
            return token == "0"
        elif position in {19, 45, 86, 95}:
            return token == "2"
        elif position in {97, 98, 41, 51, 85, 62}:
            return token == "5"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 67, 76, 22, 87, 57, 26, 60, 95}:
            return token == "3"
        elif position in {
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
            25,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            58,
            59,
            61,
            62,
            63,
            64,
            65,
            66,
            68,
            69,
            70,
            71,
            72,
            73,
            75,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            96,
            97,
            98,
            99,
        }:
            return token == ""
        elif position in {6}:
            return token == "<s>"
        elif position in {74, 45}:
            return token == "2"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_3_output):
        key = (position, attn_0_3_output)
        if key in {
            (0, "0"),
            (0, "2"),
            (0, "4"),
            (0, "<s>"),
            (1, "0"),
            (1, "2"),
            (1, "4"),
            (2, "4"),
            (8, "0"),
            (8, "2"),
            (8, "4"),
            (9, "0"),
            (9, "2"),
            (9, "4"),
            (10, "0"),
            (10, "2"),
            (10, "4"),
            (11, "0"),
            (11, "4"),
            (12, "0"),
            (12, "2"),
            (12, "4"),
            (13, "0"),
            (13, "2"),
            (13, "4"),
            (14, "0"),
            (14, "2"),
            (14, "4"),
            (15, "0"),
            (15, "2"),
            (15, "4"),
            (16, "0"),
            (16, "2"),
            (16, "4"),
            (17, "0"),
            (17, "2"),
            (17, "4"),
            (18, "0"),
            (18, "2"),
            (18, "4"),
            (19, "0"),
            (19, "2"),
            (19, "4"),
            (20, "0"),
            (20, "2"),
            (20, "4"),
            (21, "0"),
            (21, "2"),
            (21, "4"),
            (22, "0"),
            (22, "2"),
            (22, "4"),
            (23, "0"),
            (23, "2"),
            (23, "4"),
            (24, "0"),
            (24, "2"),
            (24, "4"),
            (25, "0"),
            (25, "2"),
            (25, "4"),
            (26, "0"),
            (26, "2"),
            (26, "4"),
            (27, "0"),
            (27, "2"),
            (27, "4"),
            (28, "0"),
            (28, "2"),
            (28, "4"),
            (29, "0"),
            (29, "2"),
            (29, "4"),
            (30, "0"),
            (30, "2"),
            (30, "4"),
            (31, "0"),
            (31, "2"),
            (31, "4"),
            (32, "0"),
            (32, "2"),
            (32, "4"),
            (33, "0"),
            (33, "2"),
            (33, "4"),
            (34, "0"),
            (34, "2"),
            (34, "4"),
            (35, "0"),
            (35, "2"),
            (35, "4"),
            (36, "0"),
            (36, "2"),
            (36, "4"),
            (37, "0"),
            (37, "2"),
            (37, "4"),
            (38, "0"),
            (38, "2"),
            (38, "4"),
            (39, "0"),
            (39, "2"),
            (39, "4"),
            (40, "0"),
            (40, "2"),
            (40, "4"),
            (41, "0"),
            (41, "4"),
            (42, "0"),
            (42, "2"),
            (42, "4"),
            (43, "0"),
            (43, "2"),
            (43, "4"),
            (44, "0"),
            (44, "2"),
            (44, "4"),
            (45, "0"),
            (45, "2"),
            (45, "4"),
            (46, "0"),
            (46, "4"),
            (47, "0"),
            (47, "2"),
            (47, "4"),
            (48, "0"),
            (48, "2"),
            (48, "4"),
            (49, "0"),
            (49, "4"),
            (50, "0"),
            (50, "2"),
            (50, "4"),
            (51, "0"),
            (51, "2"),
            (51, "4"),
            (52, "0"),
            (52, "2"),
            (52, "4"),
            (53, "0"),
            (53, "2"),
            (53, "4"),
            (54, "0"),
            (54, "2"),
            (54, "4"),
            (55, "0"),
            (55, "2"),
            (55, "4"),
            (56, "0"),
            (56, "2"),
            (56, "4"),
            (57, "0"),
            (57, "2"),
            (57, "4"),
            (58, "0"),
            (58, "2"),
            (58, "4"),
            (59, "0"),
            (59, "2"),
            (59, "4"),
            (60, "0"),
            (60, "2"),
            (60, "4"),
            (61, "0"),
            (61, "2"),
            (61, "4"),
            (62, "0"),
            (62, "2"),
            (62, "4"),
            (63, "0"),
            (63, "2"),
            (63, "4"),
            (64, "0"),
            (64, "2"),
            (64, "4"),
            (65, "0"),
            (65, "2"),
            (65, "4"),
            (66, "0"),
            (66, "2"),
            (66, "4"),
            (67, "0"),
            (67, "2"),
            (67, "4"),
            (68, "0"),
            (68, "2"),
            (68, "4"),
            (69, "0"),
            (69, "2"),
            (69, "4"),
            (70, "0"),
            (70, "2"),
            (70, "4"),
            (71, "0"),
            (71, "2"),
            (71, "4"),
            (72, "0"),
            (72, "2"),
            (72, "4"),
            (73, "0"),
            (73, "2"),
            (73, "4"),
            (74, "0"),
            (74, "2"),
            (74, "4"),
            (75, "0"),
            (75, "2"),
            (75, "4"),
            (76, "0"),
            (76, "2"),
            (76, "4"),
            (77, "0"),
            (77, "2"),
            (77, "4"),
            (78, "0"),
            (78, "2"),
            (78, "4"),
            (79, "0"),
            (79, "4"),
            (80, "0"),
            (80, "2"),
            (80, "4"),
            (81, "0"),
            (81, "2"),
            (81, "4"),
            (82, "0"),
            (82, "2"),
            (82, "4"),
            (83, "0"),
            (83, "2"),
            (83, "4"),
            (84, "0"),
            (84, "2"),
            (84, "4"),
            (85, "0"),
            (85, "4"),
            (86, "0"),
            (86, "2"),
            (86, "4"),
            (87, "0"),
            (87, "2"),
            (87, "4"),
            (88, "0"),
            (88, "2"),
            (88, "4"),
            (89, "0"),
            (89, "2"),
            (89, "4"),
            (90, "0"),
            (90, "2"),
            (90, "4"),
            (91, "0"),
            (91, "2"),
            (91, "4"),
            (92, "0"),
            (92, "2"),
            (92, "4"),
            (93, "0"),
            (93, "2"),
            (93, "4"),
            (94, "0"),
            (94, "2"),
            (94, "4"),
            (95, "0"),
            (95, "2"),
            (95, "4"),
            (96, "0"),
            (96, "4"),
            (97, "0"),
            (97, "2"),
            (97, "4"),
            (98, "0"),
            (98, "2"),
            (98, "4"),
            (99, "0"),
            (99, "2"),
            (99, "4"),
        }:
            return 65
        elif key in {
            (0, "1"),
            (0, "5"),
            (1, "5"),
            (2, "1"),
            (2, "5"),
            (5, "5"),
            (6, "5"),
            (7, "5"),
            (8, "5"),
            (9, "5"),
            (10, "5"),
            (11, "5"),
            (12, "5"),
            (13, "5"),
            (14, "5"),
            (15, "5"),
            (16, "5"),
            (17, "5"),
            (18, "5"),
            (19, "5"),
            (20, "5"),
            (21, "5"),
            (22, "5"),
            (23, "5"),
            (24, "5"),
            (25, "5"),
            (26, "5"),
            (27, "5"),
            (28, "5"),
            (29, "5"),
            (30, "5"),
            (31, "5"),
            (32, "5"),
            (33, "5"),
            (34, "5"),
            (35, "5"),
            (36, "5"),
            (37, "5"),
            (38, "5"),
            (39, "5"),
            (40, "5"),
            (41, "5"),
            (42, "5"),
            (43, "5"),
            (44, "5"),
            (45, "5"),
            (46, "5"),
            (47, "5"),
            (48, "5"),
            (49, "5"),
            (50, "5"),
            (51, "5"),
            (52, "5"),
            (53, "5"),
            (54, "5"),
            (55, "5"),
            (56, "5"),
            (57, "5"),
            (58, "5"),
            (59, "5"),
            (60, "5"),
            (61, "5"),
            (62, "5"),
            (63, "5"),
            (64, "5"),
            (65, "5"),
            (66, "5"),
            (67, "5"),
            (68, "5"),
            (69, "5"),
            (70, "5"),
            (71, "5"),
            (72, "5"),
            (73, "5"),
            (74, "5"),
            (75, "5"),
            (76, "5"),
            (77, "5"),
            (78, "5"),
            (79, "5"),
            (80, "5"),
            (81, "5"),
            (82, "5"),
            (83, "5"),
            (84, "5"),
            (85, "5"),
            (86, "5"),
            (87, "5"),
            (88, "5"),
            (89, "5"),
            (90, "5"),
            (91, "5"),
            (92, "5"),
            (93, "5"),
            (94, "5"),
            (95, "5"),
            (96, "5"),
            (97, "5"),
            (98, "5"),
            (99, "5"),
        }:
            return 58
        elif key in {
            (4, "0"),
            (4, "2"),
            (4, "4"),
            (4, "<s>"),
            (5, "0"),
            (5, "2"),
            (5, "4"),
            (5, "<s>"),
            (6, "0"),
        }:
            return 70
        elif key in {
            (1, "<s>"),
            (8, "<s>"),
            (9, "<s>"),
            (10, "<s>"),
            (11, "<s>"),
            (12, "<s>"),
            (13, "<s>"),
            (14, "<s>"),
            (15, "<s>"),
            (16, "<s>"),
            (17, "<s>"),
            (18, "<s>"),
            (19, "<s>"),
            (20, "<s>"),
            (21, "<s>"),
            (22, "<s>"),
            (23, "<s>"),
            (24, "<s>"),
            (25, "<s>"),
            (26, "<s>"),
            (27, "<s>"),
            (28, "<s>"),
            (29, "<s>"),
            (30, "<s>"),
            (31, "<s>"),
            (32, "<s>"),
            (33, "<s>"),
            (34, "<s>"),
            (35, "<s>"),
            (36, "<s>"),
            (37, "<s>"),
            (38, "<s>"),
            (39, "<s>"),
            (40, "<s>"),
            (41, "<s>"),
            (42, "<s>"),
            (43, "<s>"),
            (44, "<s>"),
            (45, "<s>"),
            (47, "<s>"),
            (48, "<s>"),
            (49, "<s>"),
            (50, "<s>"),
            (51, "<s>"),
            (52, "<s>"),
            (53, "<s>"),
            (54, "<s>"),
            (55, "<s>"),
            (56, "<s>"),
            (57, "<s>"),
            (58, "<s>"),
            (59, "<s>"),
            (60, "<s>"),
            (61, "<s>"),
            (62, "<s>"),
            (63, "<s>"),
            (64, "<s>"),
            (65, "<s>"),
            (66, "<s>"),
            (67, "<s>"),
            (68, "<s>"),
            (69, "<s>"),
            (70, "<s>"),
            (71, "<s>"),
            (72, "<s>"),
            (73, "<s>"),
            (74, "<s>"),
            (75, "<s>"),
            (76, "<s>"),
            (77, "<s>"),
            (78, "<s>"),
            (79, "<s>"),
            (80, "<s>"),
            (81, "<s>"),
            (82, "<s>"),
            (83, "<s>"),
            (84, "<s>"),
            (85, "<s>"),
            (86, "<s>"),
            (87, "<s>"),
            (88, "<s>"),
            (89, "<s>"),
            (90, "<s>"),
            (91, "<s>"),
            (92, "<s>"),
            (93, "<s>"),
            (94, "<s>"),
            (95, "<s>"),
            (96, "<s>"),
            (97, "<s>"),
            (98, "<s>"),
            (99, "<s>"),
        }:
            return 87
        elif key in {(2, "2"), (2, "<s>")}:
            return 37
        elif key in {(3, "<s>"), (6, "4"), (6, "<s>"), (7, "<s>"), (46, "<s>")}:
            return 15
        elif key in {(4, "5")}:
            return 40
        elif key in {(2, "0")}:
            return 98
        return 78

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_3_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("0", 0),
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("0", 16),
            ("0", 17),
            ("0", 18),
            ("0", 19),
            ("0", 20),
            ("0", 21),
            ("0", 22),
            ("0", 23),
            ("0", 24),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("0", 28),
            ("0", 29),
            ("0", 30),
            ("0", 31),
            ("0", 32),
            ("0", 33),
            ("0", 34),
            ("0", 35),
            ("0", 36),
            ("0", 37),
            ("0", 38),
            ("0", 39),
            ("0", 40),
            ("0", 41),
            ("0", 42),
            ("0", 43),
            ("0", 44),
            ("0", 45),
            ("0", 46),
            ("0", 47),
            ("0", 48),
            ("0", 49),
            ("0", 50),
            ("0", 51),
            ("0", 52),
            ("0", 53),
            ("0", 54),
            ("0", 55),
            ("0", 56),
            ("0", 57),
            ("0", 58),
            ("0", 59),
            ("0", 60),
            ("0", 61),
            ("0", 62),
            ("0", 63),
            ("0", 64),
            ("0", 65),
            ("0", 66),
            ("0", 67),
            ("0", 68),
            ("0", 69),
            ("0", 70),
            ("0", 71),
            ("0", 72),
            ("0", 73),
            ("0", 74),
            ("0", 75),
            ("0", 76),
            ("0", 77),
            ("0", 78),
            ("0", 79),
            ("0", 80),
            ("0", 81),
            ("0", 82),
            ("0", 83),
            ("0", 84),
            ("0", 85),
            ("0", 86),
            ("0", 87),
            ("0", 88),
            ("0", 89),
            ("0", 90),
            ("0", 91),
            ("0", 92),
            ("0", 93),
            ("0", 94),
            ("0", 95),
            ("0", 96),
            ("0", 97),
            ("0", 98),
            ("0", 99),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 14),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 14),
            ("5", 0),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("5", 13),
            ("5", 14),
            ("5", 15),
            ("5", 16),
            ("5", 17),
            ("5", 18),
            ("5", 19),
            ("5", 20),
            ("5", 21),
            ("5", 22),
            ("5", 23),
            ("5", 24),
            ("5", 25),
            ("5", 26),
            ("5", 27),
            ("5", 28),
            ("5", 29),
            ("5", 30),
            ("5", 31),
            ("5", 32),
            ("5", 33),
            ("5", 34),
            ("5", 35),
            ("5", 36),
            ("5", 37),
            ("5", 38),
            ("5", 39),
            ("5", 40),
            ("5", 41),
            ("5", 42),
            ("5", 43),
            ("5", 44),
            ("5", 45),
            ("5", 46),
            ("5", 47),
            ("5", 48),
            ("5", 49),
            ("5", 50),
            ("5", 51),
            ("5", 52),
            ("5", 53),
            ("5", 54),
            ("5", 55),
            ("5", 56),
            ("5", 57),
            ("5", 58),
            ("5", 59),
            ("5", 60),
            ("5", 61),
            ("5", 62),
            ("5", 63),
            ("5", 64),
            ("5", 65),
            ("5", 66),
            ("5", 67),
            ("5", 68),
            ("5", 69),
            ("5", 70),
            ("5", 71),
            ("5", 72),
            ("5", 73),
            ("5", 74),
            ("5", 75),
            ("5", 76),
            ("5", 77),
            ("5", 78),
            ("5", 79),
            ("5", 80),
            ("5", 81),
            ("5", 82),
            ("5", 83),
            ("5", 84),
            ("5", 85),
            ("5", 86),
            ("5", 87),
            ("5", 88),
            ("5", 89),
            ("5", 90),
            ("5", 91),
            ("5", 92),
            ("5", 93),
            ("5", 94),
            ("5", 95),
            ("5", 96),
            ("5", 97),
            ("5", 98),
            ("5", 99),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
        }:
            return 54
        elif key in {
            ("1", 0),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 11),
            ("1", 12),
            ("1", 13),
            ("1", 15),
            ("1", 16),
            ("1", 17),
            ("1", 18),
            ("1", 20),
            ("1", 21),
            ("1", 22),
            ("1", 24),
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("1", 30),
            ("1", 31),
            ("1", 32),
            ("1", 33),
            ("1", 34),
            ("1", 35),
            ("1", 36),
            ("1", 38),
            ("1", 39),
            ("1", 40),
            ("1", 41),
            ("1", 42),
            ("1", 43),
            ("1", 44),
            ("1", 45),
            ("1", 46),
            ("1", 48),
            ("1", 49),
            ("1", 52),
            ("1", 54),
            ("1", 55),
            ("1", 58),
            ("1", 59),
            ("1", 60),
            ("1", 62),
            ("1", 63),
            ("1", 64),
            ("1", 66),
            ("1", 67),
            ("1", 69),
            ("1", 70),
            ("1", 72),
            ("1", 75),
            ("1", 77),
            ("1", 78),
            ("1", 79),
            ("1", 80),
            ("1", 82),
            ("1", 83),
            ("1", 84),
            ("1", 85),
            ("1", 88),
            ("1", 91),
            ("1", 92),
            ("1", 93),
            ("1", 94),
            ("1", 97),
            ("1", 99),
            ("2", 0),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 8),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 12),
            ("2", 13),
            ("2", 14),
            ("2", 15),
            ("2", 16),
            ("2", 17),
            ("2", 18),
            ("2", 19),
            ("2", 20),
            ("2", 21),
            ("2", 22),
            ("2", 23),
            ("2", 24),
            ("2", 25),
            ("2", 26),
            ("2", 27),
            ("2", 28),
            ("2", 29),
            ("2", 30),
            ("2", 31),
            ("2", 32),
            ("2", 33),
            ("2", 34),
            ("2", 35),
            ("2", 36),
            ("2", 37),
            ("2", 38),
            ("2", 39),
            ("2", 40),
            ("2", 41),
            ("2", 42),
            ("2", 43),
            ("2", 44),
            ("2", 45),
            ("2", 46),
            ("2", 47),
            ("2", 48),
            ("2", 49),
            ("2", 50),
            ("2", 51),
            ("2", 52),
            ("2", 53),
            ("2", 54),
            ("2", 55),
            ("2", 56),
            ("2", 57),
            ("2", 58),
            ("2", 59),
            ("2", 60),
            ("2", 61),
            ("2", 62),
            ("2", 63),
            ("2", 64),
            ("2", 65),
            ("2", 66),
            ("2", 67),
            ("2", 68),
            ("2", 69),
            ("2", 70),
            ("2", 71),
            ("2", 72),
            ("2", 73),
            ("2", 74),
            ("2", 75),
            ("2", 76),
            ("2", 77),
            ("2", 78),
            ("2", 79),
            ("2", 80),
            ("2", 81),
            ("2", 82),
            ("2", 83),
            ("2", 84),
            ("2", 85),
            ("2", 86),
            ("2", 87),
            ("2", 88),
            ("2", 89),
            ("2", 90),
            ("2", 91),
            ("2", 92),
            ("2", 93),
            ("2", 94),
            ("2", 95),
            ("2", 96),
            ("2", 97),
            ("2", 98),
            ("2", 99),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("4", 2),
            ("4", 4),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
        }:
            return 13
        elif key in {
            ("2", 6),
            ("4", 0),
            ("4", 3),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 15),
            ("4", 16),
            ("4", 17),
            ("4", 18),
            ("4", 19),
            ("4", 20),
            ("4", 21),
            ("4", 22),
            ("4", 23),
            ("4", 24),
            ("4", 25),
            ("4", 26),
            ("4", 27),
            ("4", 28),
            ("4", 29),
            ("4", 30),
            ("4", 31),
            ("4", 32),
            ("4", 33),
            ("4", 34),
            ("4", 35),
            ("4", 36),
            ("4", 37),
            ("4", 38),
            ("4", 39),
            ("4", 40),
            ("4", 41),
            ("4", 42),
            ("4", 43),
            ("4", 44),
            ("4", 45),
            ("4", 46),
            ("4", 47),
            ("4", 48),
            ("4", 49),
            ("4", 50),
            ("4", 51),
            ("4", 52),
            ("4", 53),
            ("4", 54),
            ("4", 55),
            ("4", 56),
            ("4", 57),
            ("4", 58),
            ("4", 59),
            ("4", 60),
            ("4", 61),
            ("4", 62),
            ("4", 63),
            ("4", 64),
            ("4", 65),
            ("4", 66),
            ("4", 67),
            ("4", 68),
            ("4", 69),
            ("4", 70),
            ("4", 71),
            ("4", 72),
            ("4", 73),
            ("4", 74),
            ("4", 75),
            ("4", 76),
            ("4", 77),
            ("4", 78),
            ("4", 79),
            ("4", 80),
            ("4", 81),
            ("4", 82),
            ("4", 83),
            ("4", 84),
            ("4", 85),
            ("4", 86),
            ("4", 87),
            ("4", 88),
            ("4", 89),
            ("4", 90),
            ("4", 91),
            ("4", 92),
            ("4", 93),
            ("4", 94),
            ("4", 95),
            ("4", 96),
            ("4", 97),
            ("4", 98),
            ("4", 99),
        }:
            return 89
        elif key in {
            ("1", 9),
            ("1", 14),
            ("1", 23),
            ("1", 61),
            ("1", 65),
            ("1", 73),
            ("1", 74),
            ("1", 76),
            ("1", 81),
            ("1", 96),
            ("1", 98),
            ("3", 8),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 15),
            ("3", 16),
            ("3", 17),
            ("3", 18),
            ("3", 19),
            ("3", 20),
            ("3", 21),
            ("3", 22),
            ("3", 23),
            ("3", 24),
            ("3", 25),
            ("3", 26),
            ("3", 27),
            ("3", 28),
            ("3", 29),
            ("3", 30),
            ("3", 31),
            ("3", 32),
            ("3", 33),
            ("3", 34),
            ("3", 35),
            ("3", 36),
            ("3", 37),
            ("3", 38),
            ("3", 39),
            ("3", 40),
            ("3", 41),
            ("3", 42),
            ("3", 43),
            ("3", 44),
            ("3", 45),
            ("3", 46),
            ("3", 47),
            ("3", 48),
            ("3", 49),
            ("3", 50),
            ("3", 51),
            ("3", 52),
            ("3", 53),
            ("3", 54),
            ("3", 55),
            ("3", 56),
            ("3", 57),
            ("3", 58),
            ("3", 59),
            ("3", 60),
            ("3", 61),
            ("3", 62),
            ("3", 63),
            ("3", 64),
            ("3", 65),
            ("3", 66),
            ("3", 67),
            ("3", 68),
            ("3", 69),
            ("3", 70),
            ("3", 71),
            ("3", 72),
            ("3", 73),
            ("3", 74),
            ("3", 75),
            ("3", 76),
            ("3", 77),
            ("3", 78),
            ("3", 79),
            ("3", 80),
            ("3", 81),
            ("3", 82),
            ("3", 83),
            ("3", 84),
            ("3", 85),
            ("3", 86),
            ("3", 87),
            ("3", 88),
            ("3", 89),
            ("3", 90),
            ("3", 91),
            ("3", 92),
            ("3", 93),
            ("3", 94),
            ("3", 95),
            ("3", 96),
            ("3", 97),
            ("3", 98),
            ("3", 99),
        }:
            return 53
        elif key in {("0", 2), ("0", 3), ("0", 4)}:
            return 97
        elif key in {("2", 7), ("5", 2), ("5", 3), ("5", 4)}:
            return 64
        elif key in {("3", 0)}:
            return 74
        return 20

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 86

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 88

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3, 7}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5, 6, 74, 45, 14, 81}:
            return k_position == 7
        elif q_position in {
            8,
            10,
            11,
            13,
            16,
            17,
            18,
            21,
            22,
            25,
            26,
            27,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            44,
            47,
            49,
            50,
            53,
            56,
            58,
            59,
            62,
            64,
            65,
            66,
            70,
            71,
            75,
            76,
            82,
            83,
            84,
            86,
            87,
            88,
            90,
            91,
            92,
            94,
            95,
            96,
        }:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 29
        elif q_position in {
            33,
            98,
            67,
            68,
            69,
            43,
            12,
            77,
            79,
            19,
            20,
            52,
            54,
            23,
            55,
            61,
            60,
            29,
        }:
            return k_position == 45
        elif q_position in {15}:
            return k_position == 95
        elif q_position in {24}:
            return k_position == 72
        elif q_position in {73, 28}:
            return k_position == 66
        elif q_position in {46}:
            return k_position == 14
        elif q_position in {48}:
            return k_position == 69
        elif q_position in {51}:
            return k_position == 46
        elif q_position in {57}:
            return k_position == 99
        elif q_position in {63}:
            return k_position == 18
        elif q_position in {72}:
            return k_position == 80
        elif q_position in {78}:
            return k_position == 36
        elif q_position in {80}:
            return k_position == 86
        elif q_position in {85}:
            return k_position == 28
        elif q_position in {89}:
            return k_position == 43
        elif q_position in {93}:
            return k_position == 35
        elif q_position in {97}:
            return k_position == 82
        elif q_position in {99}:
            return k_position == 75

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 2, 98, 20, 24}:
            return k_position == 1
        elif q_position in {1, 7}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4, 5, 6}:
            return k_position == 3
        elif q_position in {
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            21,
            22,
            23,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            99,
        }:
            return k_position == 45
        elif q_position in {32}:
            return k_position == 20
        elif q_position in {45}:
            return k_position == 89
        elif q_position in {89}:
            return k_position == 14
        elif q_position in {97}:
            return k_position == 92

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 3}:
            return k_position == 0
        elif q_position in {
            4,
            5,
            8,
            10,
            11,
            12,
            13,
            15,
            16,
            17,
            18,
            19,
            21,
            22,
            25,
            26,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            37,
            39,
            41,
            42,
            43,
            44,
            47,
            48,
            49,
            50,
            52,
            53,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            99,
        }:
            return k_position == 6
        elif q_position in {89, 45, 6}:
            return k_position == 7
        elif q_position in {14, 7}:
            return k_position == 84
        elif q_position in {67, 36, 38, 40, 9, 46, 78, 51, 20, 23, 24, 28}:
            return k_position == 45
        elif q_position in {54}:
            return k_position == 28
        elif q_position in {98}:
            return k_position == 20

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, mlp_0_1_output):
        if attn_0_1_output in {"0", "<s>"}:
            return mlp_0_1_output == 45
        elif attn_0_1_output in {"1"}:
            return mlp_0_1_output == 39
        elif attn_0_1_output in {"2"}:
            return mlp_0_1_output == 14
        elif attn_0_1_output in {"3"}:
            return mlp_0_1_output == 85
        elif attn_0_1_output in {"4"}:
            return mlp_0_1_output == 97
        elif attn_0_1_output in {"5"}:
            return mlp_0_1_output == 72

    attn_1_3_pattern = select_closest(mlp_0_1_outputs, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {
            0,
            65,
            96,
            97,
            4,
            5,
            6,
            36,
            43,
            44,
            48,
            84,
            55,
            23,
            24,
            59,
            92,
            30,
        }:
            return token == "3"
        elif position in {32, 1, 68, 9, 46, 15, 20, 58}:
            return token == "5"
        elif position in {
            2,
            3,
            7,
            8,
            10,
            11,
            12,
            14,
            16,
            17,
            18,
            21,
            22,
            25,
            26,
            27,
            28,
            29,
            31,
            34,
            35,
            38,
            39,
            41,
            42,
            47,
            49,
            50,
            51,
            52,
            53,
            56,
            57,
            60,
            61,
            62,
            63,
            64,
            66,
            67,
            69,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            82,
            83,
            85,
            86,
            88,
            89,
            90,
            91,
            93,
            94,
            95,
            98,
            99,
        }:
            return token == ""
        elif position in {40, 72, 13, 70}:
            return token == "<s>"
        elif position in {81, 19, 71}:
            return token == "0"
        elif position in {80, 33, 87}:
            return token == "4"
        elif position in {37}:
            return token == "<pad>"
        elif position in {45, 54}:
            return token == "2"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            7,
            12,
            13,
            17,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            34,
            40,
            46,
            47,
            62,
            70,
            75,
            78,
            82,
            83,
            95,
            98,
        }:
            return token == ""
        elif mlp_0_0_output in {73, 1, 57}:
            return token == "4"
        elif mlp_0_0_output in {
            2,
            5,
            6,
            8,
            9,
            14,
            16,
            18,
            20,
            49,
            56,
            58,
            64,
            65,
            68,
            71,
            72,
            76,
            85,
            87,
            89,
            90,
            93,
        }:
            return token == "0"
        elif mlp_0_0_output in {3, 39, 45, 77, 15, 79, 80, 81, 92, 52, 54, 86, 60}:
            return token == "3"
        elif mlp_0_0_output in {59, 4, 44}:
            return token == "2"
        elif mlp_0_0_output in {
            10,
            11,
            19,
            28,
            29,
            30,
            31,
            35,
            36,
            38,
            41,
            42,
            48,
            50,
            51,
            53,
            61,
            63,
            66,
            67,
            69,
            74,
            88,
            91,
            94,
            97,
            99,
        }:
            return token == "5"
        elif mlp_0_0_output in {32, 33, 96, 43, 84, 55}:
            return token == "1"
        elif mlp_0_0_output in {37}:
            return token == "<s>"

    num_attn_1_1_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            2,
            3,
            10,
            11,
            17,
            18,
            20,
            22,
            24,
            27,
            31,
            34,
            35,
            37,
            42,
            57,
            58,
            62,
            63,
            65,
            66,
            67,
            73,
            75,
            76,
            78,
            82,
            83,
            85,
            87,
            90,
            91,
            95,
            97,
            98,
        }:
            return token == ""
        elif mlp_0_0_output in {1, 12, 13, 47, 19}:
            return token == "4"
        elif mlp_0_0_output in {
            4,
            5,
            6,
            29,
            30,
            36,
            39,
            40,
            41,
            43,
            44,
            48,
            55,
            59,
            61,
            70,
            77,
            84,
            92,
            96,
        }:
            return token == "0"
        elif mlp_0_0_output in {69, 7, 45, 60, 15, 80, 81, 52, 54, 28}:
            return token == "<s>"
        elif mlp_0_0_output in {
            8,
            14,
            16,
            21,
            23,
            25,
            26,
            38,
            49,
            50,
            51,
            53,
            56,
            64,
            72,
            74,
            79,
            86,
            88,
            89,
            94,
        }:
            return token == "2"
        elif mlp_0_0_output in {9, 46, 71}:
            return token == "5"
        elif mlp_0_0_output in {32}:
            return token == "1"
        elif mlp_0_0_output in {33, 68, 93}:
            return token == "3"
        elif mlp_0_0_output in {99}:
            return token == "<pad>"

    num_attn_1_2_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 67, 73, 50, 21, 55, 91, 29}:
            return token == "3"
        elif mlp_0_0_output in {32, 1, 98, 5, 6, 71, 9, 20, 87, 58}:
            return token == "2"
        elif mlp_0_0_output in {
            2,
            7,
            8,
            12,
            13,
            14,
            16,
            19,
            23,
            24,
            26,
            27,
            31,
            33,
            34,
            36,
            37,
            43,
            44,
            47,
            49,
            54,
            56,
            57,
            59,
            62,
            63,
            64,
            65,
            76,
            79,
            80,
            82,
            83,
            85,
            86,
            88,
            89,
            92,
            93,
            96,
            99,
        }:
            return token == ""
        elif mlp_0_0_output in {3, 39, 42, 81, 30}:
            return token == "1"
        elif mlp_0_0_output in {35, 4, 75, 18, 22}:
            return token == "4"
        elif mlp_0_0_output in {
            10,
            11,
            15,
            17,
            25,
            28,
            41,
            46,
            48,
            51,
            52,
            53,
            60,
            61,
            66,
            69,
            70,
            72,
            74,
            77,
            78,
            84,
            90,
            97,
        }:
            return token == "5"
        elif mlp_0_0_output in {94, 68, 38, 95}:
            return token == "0"
        elif mlp_0_0_output in {40, 45}:
            return token == "<s>"

    num_attn_1_3_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_0_3_output):
        key = (attn_1_1_output, attn_0_3_output)
        return 80

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_2_output, mlp_0_1_output):
        key = (attn_0_2_output, mlp_0_1_output)
        if key in {
            ("1", 1),
            ("1", 4),
            ("1", 7),
            ("1", 8),
            ("1", 12),
            ("1", 29),
            ("1", 53),
            ("1", 60),
            ("1", 81),
            ("2", 1),
            ("2", 4),
            ("2", 5),
            ("2", 7),
            ("2", 8),
            ("2", 53),
            ("2", 60),
            ("3", 1),
            ("3", 4),
            ("3", 5),
            ("3", 7),
            ("3", 8),
            ("3", 53),
            ("3", 60),
            ("4", 13),
            ("4", 20),
            ("4", 32),
            ("5", 1),
            ("5", 4),
            ("5", 5),
            ("5", 7),
            ("5", 8),
            ("5", 12),
            ("5", 29),
            ("5", 53),
            ("5", 60),
            ("5", 81),
            ("<s>", 1),
            ("<s>", 4),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 12),
            ("<s>", 29),
            ("<s>", 53),
            ("<s>", 60),
            ("<s>", 76),
            ("<s>", 81),
            ("<s>", 92),
        }:
            return 12
        elif key in {
            ("0", 13),
            ("0", 20),
            ("0", 31),
            ("0", 32),
            ("0", 87),
            ("1", 0),
            ("1", 2),
            ("1", 3),
            ("1", 6),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 13),
            ("1", 14),
            ("1", 15),
            ("1", 16),
            ("1", 17),
            ("1", 18),
            ("1", 19),
            ("1", 20),
            ("1", 21),
            ("1", 22),
            ("1", 23),
            ("1", 24),
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("1", 28),
            ("1", 30),
            ("1", 31),
            ("1", 32),
            ("1", 33),
            ("1", 34),
            ("1", 35),
            ("1", 36),
            ("1", 37),
            ("1", 38),
            ("1", 39),
            ("1", 40),
            ("1", 41),
            ("1", 42),
            ("1", 43),
            ("1", 44),
            ("1", 46),
            ("1", 47),
            ("1", 48),
            ("1", 49),
            ("1", 50),
            ("1", 51),
            ("1", 52),
            ("1", 54),
            ("1", 55),
            ("1", 56),
            ("1", 57),
            ("1", 58),
            ("1", 59),
            ("1", 61),
            ("1", 62),
            ("1", 63),
            ("1", 64),
            ("1", 65),
            ("1", 66),
            ("1", 67),
            ("1", 68),
            ("1", 69),
            ("1", 70),
            ("1", 71),
            ("1", 72),
            ("1", 73),
            ("1", 74),
            ("1", 75),
            ("1", 76),
            ("1", 77),
            ("1", 78),
            ("1", 79),
            ("1", 80),
            ("1", 82),
            ("1", 83),
            ("1", 84),
            ("1", 85),
            ("1", 86),
            ("1", 87),
            ("1", 88),
            ("1", 89),
            ("1", 90),
            ("1", 91),
            ("1", 92),
            ("1", 93),
            ("1", 94),
            ("1", 95),
            ("1", 96),
            ("1", 97),
            ("1", 98),
            ("1", 99),
            ("2", 0),
            ("2", 2),
            ("2", 3),
            ("2", 6),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 12),
            ("2", 13),
            ("2", 14),
            ("2", 15),
            ("2", 16),
            ("2", 17),
            ("2", 18),
            ("2", 19),
            ("2", 20),
            ("2", 21),
            ("2", 22),
            ("2", 23),
            ("2", 24),
            ("2", 25),
            ("2", 26),
            ("2", 27),
            ("2", 28),
            ("2", 29),
            ("2", 30),
            ("2", 31),
            ("2", 32),
            ("2", 33),
            ("2", 34),
            ("2", 35),
            ("2", 36),
            ("2", 37),
            ("2", 38),
            ("2", 39),
            ("2", 40),
            ("2", 41),
            ("2", 42),
            ("2", 43),
            ("2", 44),
            ("2", 46),
            ("2", 47),
            ("2", 48),
            ("2", 49),
            ("2", 50),
            ("2", 51),
            ("2", 52),
            ("2", 54),
            ("2", 55),
            ("2", 56),
            ("2", 57),
            ("2", 58),
            ("2", 59),
            ("2", 61),
            ("2", 62),
            ("2", 63),
            ("2", 64),
            ("2", 65),
            ("2", 66),
            ("2", 67),
            ("2", 68),
            ("2", 69),
            ("2", 70),
            ("2", 71),
            ("2", 72),
            ("2", 73),
            ("2", 74),
            ("2", 75),
            ("2", 76),
            ("2", 77),
            ("2", 78),
            ("2", 79),
            ("2", 80),
            ("2", 81),
            ("2", 82),
            ("2", 83),
            ("2", 84),
            ("2", 85),
            ("2", 86),
            ("2", 87),
            ("2", 88),
            ("2", 89),
            ("2", 90),
            ("2", 91),
            ("2", 92),
            ("2", 93),
            ("2", 94),
            ("2", 95),
            ("2", 96),
            ("2", 97),
            ("2", 98),
            ("2", 99),
            ("3", 0),
            ("3", 2),
            ("3", 3),
            ("3", 6),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 14),
            ("3", 15),
            ("3", 16),
            ("3", 17),
            ("3", 18),
            ("3", 19),
            ("3", 20),
            ("3", 21),
            ("3", 22),
            ("3", 23),
            ("3", 24),
            ("3", 25),
            ("3", 26),
            ("3", 27),
            ("3", 28),
            ("3", 29),
            ("3", 30),
            ("3", 31),
            ("3", 32),
            ("3", 33),
            ("3", 34),
            ("3", 35),
            ("3", 36),
            ("3", 37),
            ("3", 38),
            ("3", 39),
            ("3", 40),
            ("3", 41),
            ("3", 42),
            ("3", 43),
            ("3", 44),
            ("3", 46),
            ("3", 47),
            ("3", 48),
            ("3", 49),
            ("3", 50),
            ("3", 51),
            ("3", 52),
            ("3", 54),
            ("3", 55),
            ("3", 56),
            ("3", 57),
            ("3", 58),
            ("3", 59),
            ("3", 61),
            ("3", 62),
            ("3", 63),
            ("3", 64),
            ("3", 65),
            ("3", 66),
            ("3", 67),
            ("3", 68),
            ("3", 69),
            ("3", 70),
            ("3", 71),
            ("3", 72),
            ("3", 73),
            ("3", 74),
            ("3", 75),
            ("3", 76),
            ("3", 77),
            ("3", 78),
            ("3", 79),
            ("3", 80),
            ("3", 81),
            ("3", 82),
            ("3", 83),
            ("3", 84),
            ("3", 85),
            ("3", 86),
            ("3", 87),
            ("3", 88),
            ("3", 89),
            ("3", 90),
            ("3", 91),
            ("3", 92),
            ("3", 93),
            ("3", 94),
            ("3", 95),
            ("3", 96),
            ("3", 97),
            ("3", 98),
            ("3", 99),
            ("5", 0),
            ("5", 2),
            ("5", 3),
            ("5", 6),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 13),
            ("5", 14),
            ("5", 15),
            ("5", 16),
            ("5", 17),
            ("5", 18),
            ("5", 19),
            ("5", 20),
            ("5", 21),
            ("5", 22),
            ("5", 23),
            ("5", 24),
            ("5", 25),
            ("5", 26),
            ("5", 27),
            ("5", 28),
            ("5", 30),
            ("5", 31),
            ("5", 32),
            ("5", 33),
            ("5", 34),
            ("5", 35),
            ("5", 36),
            ("5", 37),
            ("5", 38),
            ("5", 39),
            ("5", 40),
            ("5", 41),
            ("5", 42),
            ("5", 43),
            ("5", 44),
            ("5", 46),
            ("5", 47),
            ("5", 48),
            ("5", 49),
            ("5", 50),
            ("5", 51),
            ("5", 52),
            ("5", 54),
            ("5", 55),
            ("5", 56),
            ("5", 57),
            ("5", 58),
            ("5", 59),
            ("5", 61),
            ("5", 62),
            ("5", 63),
            ("5", 64),
            ("5", 65),
            ("5", 66),
            ("5", 67),
            ("5", 68),
            ("5", 69),
            ("5", 70),
            ("5", 71),
            ("5", 72),
            ("5", 73),
            ("5", 74),
            ("5", 75),
            ("5", 76),
            ("5", 77),
            ("5", 78),
            ("5", 79),
            ("5", 80),
            ("5", 82),
            ("5", 83),
            ("5", 84),
            ("5", 85),
            ("5", 86),
            ("5", 87),
            ("5", 88),
            ("5", 89),
            ("5", 90),
            ("5", 91),
            ("5", 92),
            ("5", 93),
            ("5", 94),
            ("5", 95),
            ("5", 96),
            ("5", 97),
            ("5", 98),
            ("5", 99),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 6),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
            ("<s>", 16),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 19),
            ("<s>", 20),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 24),
            ("<s>", 25),
            ("<s>", 26),
            ("<s>", 27),
            ("<s>", 28),
            ("<s>", 30),
            ("<s>", 31),
            ("<s>", 32),
            ("<s>", 33),
            ("<s>", 34),
            ("<s>", 35),
            ("<s>", 36),
            ("<s>", 37),
            ("<s>", 38),
            ("<s>", 39),
            ("<s>", 40),
            ("<s>", 41),
            ("<s>", 42),
            ("<s>", 43),
            ("<s>", 44),
            ("<s>", 46),
            ("<s>", 47),
            ("<s>", 48),
            ("<s>", 49),
            ("<s>", 50),
            ("<s>", 51),
            ("<s>", 52),
            ("<s>", 54),
            ("<s>", 55),
            ("<s>", 56),
            ("<s>", 57),
            ("<s>", 58),
            ("<s>", 59),
            ("<s>", 61),
            ("<s>", 62),
            ("<s>", 63),
            ("<s>", 64),
            ("<s>", 65),
            ("<s>", 66),
            ("<s>", 67),
            ("<s>", 68),
            ("<s>", 69),
            ("<s>", 70),
            ("<s>", 71),
            ("<s>", 72),
            ("<s>", 73),
            ("<s>", 74),
            ("<s>", 75),
            ("<s>", 77),
            ("<s>", 78),
            ("<s>", 79),
            ("<s>", 80),
            ("<s>", 82),
            ("<s>", 83),
            ("<s>", 84),
            ("<s>", 85),
            ("<s>", 86),
            ("<s>", 87),
            ("<s>", 88),
            ("<s>", 89),
            ("<s>", 90),
            ("<s>", 91),
            ("<s>", 93),
            ("<s>", 94),
            ("<s>", 95),
            ("<s>", 96),
            ("<s>", 97),
            ("<s>", 98),
            ("<s>", 99),
        }:
            return 95
        elif key in {("1", 5), ("4", 31), ("4", 68), ("4", 87), ("<s>", 5)}:
            return 44
        elif key in {
            ("0", 1),
            ("0", 4),
            ("0", 5),
            ("0", 7),
            ("0", 8),
            ("0", 45),
            ("0", 53),
            ("0", 60),
        }:
            return 34
        elif key in {
            ("0", 0),
            ("0", 2),
            ("0", 3),
            ("0", 6),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 14),
            ("0", 15),
            ("0", 16),
            ("0", 17),
            ("0", 18),
            ("0", 19),
            ("0", 21),
            ("0", 22),
            ("0", 23),
            ("0", 24),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("0", 28),
            ("0", 29),
            ("0", 30),
            ("0", 33),
            ("0", 34),
            ("0", 35),
            ("0", 36),
            ("0", 37),
            ("0", 38),
            ("0", 39),
            ("0", 40),
            ("0", 41),
            ("0", 42),
            ("0", 43),
            ("0", 44),
            ("0", 46),
            ("0", 47),
            ("0", 48),
            ("0", 49),
            ("0", 50),
            ("0", 51),
            ("0", 52),
            ("0", 54),
            ("0", 55),
            ("0", 56),
            ("0", 57),
            ("0", 58),
            ("0", 59),
            ("0", 61),
            ("0", 62),
            ("0", 63),
            ("0", 64),
            ("0", 65),
            ("0", 66),
            ("0", 67),
            ("0", 68),
            ("0", 69),
            ("0", 70),
            ("0", 71),
            ("0", 72),
            ("0", 73),
            ("0", 74),
            ("0", 75),
            ("0", 76),
            ("0", 77),
            ("0", 78),
            ("0", 79),
            ("0", 80),
            ("0", 81),
            ("0", 82),
            ("0", 83),
            ("0", 84),
            ("0", 85),
            ("0", 86),
            ("0", 88),
            ("0", 89),
            ("0", 90),
            ("0", 91),
            ("0", 92),
            ("0", 93),
            ("0", 94),
            ("0", 95),
            ("0", 96),
            ("0", 97),
            ("0", 98),
            ("0", 99),
        }:
            return 76
        elif key in {
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 14),
            ("4", 15),
            ("4", 16),
            ("4", 17),
            ("4", 18),
            ("4", 19),
            ("4", 21),
            ("4", 22),
            ("4", 23),
            ("4", 24),
            ("4", 25),
            ("4", 26),
            ("4", 27),
            ("4", 28),
            ("4", 29),
            ("4", 30),
            ("4", 33),
            ("4", 34),
            ("4", 35),
            ("4", 36),
            ("4", 37),
            ("4", 38),
            ("4", 39),
            ("4", 40),
            ("4", 41),
            ("4", 42),
            ("4", 43),
            ("4", 44),
            ("4", 46),
            ("4", 47),
            ("4", 48),
            ("4", 49),
            ("4", 50),
            ("4", 51),
            ("4", 52),
            ("4", 53),
            ("4", 54),
            ("4", 55),
            ("4", 56),
            ("4", 57),
            ("4", 58),
            ("4", 59),
            ("4", 60),
            ("4", 61),
            ("4", 62),
            ("4", 63),
            ("4", 64),
            ("4", 65),
            ("4", 66),
            ("4", 67),
            ("4", 69),
            ("4", 70),
            ("4", 71),
            ("4", 72),
            ("4", 73),
            ("4", 74),
            ("4", 75),
            ("4", 76),
            ("4", 77),
            ("4", 78),
            ("4", 79),
            ("4", 80),
            ("4", 81),
            ("4", 82),
            ("4", 83),
            ("4", 84),
            ("4", 85),
            ("4", 86),
            ("4", 88),
            ("4", 89),
            ("4", 90),
            ("4", 91),
            ("4", 92),
            ("4", 93),
            ("4", 94),
            ("4", 95),
            ("4", 96),
            ("4", 97),
            ("4", 98),
            ("4", 99),
        }:
            return 0
        elif key in {("4", 45)}:
            return 85
        return 11

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, mlp_0_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        return 59

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_0_output, num_attn_0_1_output):
        key = (num_attn_0_0_output, num_attn_0_1_output)
        return 67

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 94, 86, 97}:
            return position == 89
        elif mlp_0_0_output in {1}:
            return position == 4
        elif mlp_0_0_output in {2}:
            return position == 0
        elif mlp_0_0_output in {98, 3, 37, 20, 29}:
            return position == 6
        elif mlp_0_0_output in {4}:
            return position == 14
        elif mlp_0_0_output in {5, 55}:
            return position == 51
        elif mlp_0_0_output in {18, 6}:
            return position == 20
        elif mlp_0_0_output in {7, 72, 12, 77, 53, 63}:
            return position == 37
        elif mlp_0_0_output in {8, 41, 66}:
            return position == 32
        elif mlp_0_0_output in {9, 74, 19}:
            return position == 19
        elif mlp_0_0_output in {10, 59}:
            return position == 79
        elif mlp_0_0_output in {11}:
            return position == 86
        elif mlp_0_0_output in {48, 13}:
            return position == 24
        elif mlp_0_0_output in {14}:
            return position == 76
        elif mlp_0_0_output in {78, 70, 15}:
            return position == 1
        elif mlp_0_0_output in {16, 58, 82}:
            return position == 74
        elif mlp_0_0_output in {17}:
            return position == 94
        elif mlp_0_0_output in {91, 21}:
            return position == 22
        elif mlp_0_0_output in {22}:
            return position == 35
        elif mlp_0_0_output in {23}:
            return position == 31
        elif mlp_0_0_output in {24, 43}:
            return position == 49
        elif mlp_0_0_output in {25, 83, 35}:
            return position == 9
        elif mlp_0_0_output in {81, 26}:
            return position == 88
        elif mlp_0_0_output in {27, 47}:
            return position == 42
        elif mlp_0_0_output in {62, 28, 46}:
            return position == 68
        elif mlp_0_0_output in {49, 30}:
            return position == 95
        elif mlp_0_0_output in {31}:
            return position == 58
        elif mlp_0_0_output in {32, 52}:
            return position == 27
        elif mlp_0_0_output in {33}:
            return position == 12
        elif mlp_0_0_output in {34}:
            return position == 72
        elif mlp_0_0_output in {65, 36}:
            return position == 56
        elif mlp_0_0_output in {54, 38}:
            return position == 91
        elif mlp_0_0_output in {95, 39}:
            return position == 82
        elif mlp_0_0_output in {40, 45}:
            return position == 2
        elif mlp_0_0_output in {42}:
            return position == 17
        elif mlp_0_0_output in {92, 44, 69, 87}:
            return position == 7
        elif mlp_0_0_output in {89, 50, 51, 90}:
            return position == 90
        elif mlp_0_0_output in {56}:
            return position == 25
        elif mlp_0_0_output in {57}:
            return position == 84
        elif mlp_0_0_output in {60}:
            return position == 44
        elif mlp_0_0_output in {61}:
            return position == 62
        elif mlp_0_0_output in {64}:
            return position == 73
        elif mlp_0_0_output in {88, 67}:
            return position == 8
        elif mlp_0_0_output in {68, 76}:
            return position == 26
        elif mlp_0_0_output in {71}:
            return position == 66
        elif mlp_0_0_output in {80, 73}:
            return position == 34
        elif mlp_0_0_output in {75}:
            return position == 13
        elif mlp_0_0_output in {79}:
            return position == 83
        elif mlp_0_0_output in {84}:
            return position == 29
        elif mlp_0_0_output in {85}:
            return position == 30
        elif mlp_0_0_output in {93}:
            return position == 18
        elif mlp_0_0_output in {96}:
            return position == 67
        elif mlp_0_0_output in {99}:
            return position == 38

    attn_2_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {
            0,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
        }:
            return k_position == 45
        elif q_position in {1, 4}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {45, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 2

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {
            0,
            8,
            9,
            11,
            16,
            19,
            20,
            22,
            23,
            25,
            27,
            29,
            30,
            32,
            34,
            35,
            36,
            37,
            41,
            45,
            48,
            49,
            50,
            51,
            52,
            53,
            56,
            59,
            61,
            62,
            65,
            68,
            71,
            72,
            73,
            74,
            75,
            81,
            82,
            84,
            85,
            87,
            88,
            92,
            96,
            98,
        }:
            return k_position == 45
        elif q_position in {1, 66, 70, 57}:
            return k_position == 6
        elif q_position in {2, 3}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 23
        elif q_position in {12}:
            return k_position == 96
        elif q_position in {64, 67, 58, 99, 40, 43, 13, 77, 15, 47, 83, 54, 26}:
            return k_position == 7
        elif q_position in {14}:
            return k_position == 20
        elif q_position in {17}:
            return k_position == 99
        elif q_position in {18}:
            return k_position == 16
        elif q_position in {90, 60, 21}:
            return k_position == 50
        elif q_position in {24, 80, 94}:
            return k_position == 89
        elif q_position in {28}:
            return k_position == 63
        elif q_position in {78, 31}:
            return k_position == 37
        elif q_position in {33}:
            return k_position == 9
        elif q_position in {38}:
            return k_position == 39
        elif q_position in {55, 39}:
            return k_position == 5
        elif q_position in {42}:
            return k_position == 65
        elif q_position in {44}:
            return k_position == 57
        elif q_position in {46}:
            return k_position == 68
        elif q_position in {63}:
            return k_position == 94
        elif q_position in {69}:
            return k_position == 14
        elif q_position in {76}:
            return k_position == 61
        elif q_position in {79}:
            return k_position == 21
        elif q_position in {86}:
            return k_position == 38
        elif q_position in {89}:
            return k_position == 97
        elif q_position in {91}:
            return k_position == 28
        elif q_position in {93}:
            return k_position == 79
        elif q_position in {95}:
            return k_position == 52
        elif q_position in {97}:
            return k_position == 82

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, position):
        if token in {"1", "4", "5", "0", "3"}:
            return position == 0
        elif token in {"2"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 74

    attn_2_3_pattern = select_closest(positions, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(token, attn_1_3_output):
        if token in {"0"}:
            return attn_1_3_output == "4"
        elif token in {"1", "4", "5", "3", "2"}:
            return attn_1_3_output == "0"
        elif token in {"<s>"}:
            return attn_1_3_output == ""

    num_attn_2_0_pattern = select(attn_1_3_outputs, tokens, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "5"
        elif attn_0_1_output in {"1", "4", "5", "3", "<s>", "2"}:
            return token == "0"

    num_attn_2_1_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_1_3_output):
        if position in {
            0,
            3,
            4,
            5,
            6,
            7,
            10,
            31,
            38,
            40,
            41,
            47,
            53,
            61,
            64,
            66,
            72,
            90,
            95,
        }:
            return attn_1_3_output == ""
        elif position in {
            1,
            2,
            11,
            15,
            16,
            19,
            20,
            22,
            26,
            32,
            35,
            37,
            39,
            49,
            54,
            57,
            58,
            60,
            63,
            68,
            71,
            73,
            74,
            76,
            80,
            81,
            82,
            86,
            93,
            98,
        }:
            return attn_1_3_output == "3"
        elif position in {65, 34, 8, 44, 46, 78, 48, 18, 52, 87, 55, 62}:
            return attn_1_3_output == "5"
        elif position in {9, 42, 59, 36}:
            return attn_1_3_output == "2"
        elif position in {12, 45, 13, 14, 51, 85}:
            return attn_1_3_output == "0"
        elif position in {17, 67}:
            return attn_1_3_output == "<s>"
        elif position in {
            21,
            24,
            25,
            28,
            30,
            43,
            56,
            69,
            70,
            75,
            77,
            79,
            84,
            88,
            89,
            91,
            94,
            96,
            97,
            99,
        }:
            return attn_1_3_output == "4"
        elif position in {33, 50, 83, 23, 27, 92, 29}:
            return attn_1_3_output == "1"

    num_attn_2_2_pattern = select(attn_1_3_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_attn_1_3_output, k_attn_1_3_output):
        if q_attn_1_3_output in {"1", "4", "5", "0", "2"}:
            return k_attn_1_3_output == ""
        elif q_attn_1_3_output in {"3", "<s>"}:
            return k_attn_1_3_output == "3"

    num_attn_2_3_pattern = select(attn_1_3_outputs, attn_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_0_output, num_mlp_1_1_output):
        key = (attn_1_0_output, num_mlp_1_1_output)
        if key in {("5", 28)}:
            return 5
        return 89

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_3_output, attn_0_3_output):
        key = (attn_2_3_output, attn_0_3_output)
        return 79

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_0_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 55

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_0_1_output):
        key = (num_attn_2_3_output, num_attn_0_1_output)
        return 38

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "1", "5", "1", "2", "0", "3"]))
