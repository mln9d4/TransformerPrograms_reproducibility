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
        "output/rasp/sort/vocab8maxlen8dvar100/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/sort_weights.csv",
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
        if q_position in {0, 1}:
            return k_position == 4
        elif q_position in {2, 3, 85, 7}:
            return k_position == 1
        elif q_position in {4, 5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 8
        elif q_position in {9, 99}:
            return k_position == 56
        elif q_position in {10}:
            return k_position == 6
        elif q_position in {40, 82, 11}:
            return k_position == 22
        elif q_position in {12}:
            return k_position == 88
        elif q_position in {83, 19, 13}:
            return k_position == 69
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {80, 94, 15}:
            return k_position == 42
        elif q_position in {16}:
            return k_position == 50
        elif q_position in {17}:
            return k_position == 59
        elif q_position in {18}:
            return k_position == 83
        elif q_position in {20, 21}:
            return k_position == 27
        elif q_position in {22, 31}:
            return k_position == 39
        elif q_position in {96, 47, 23}:
            return k_position == 68
        elif q_position in {24}:
            return k_position == 75
        elif q_position in {25, 62, 70}:
            return k_position == 97
        elif q_position in {26}:
            return k_position == 13
        elif q_position in {27, 93}:
            return k_position == 76
        elif q_position in {28}:
            return k_position == 19
        elif q_position in {29}:
            return k_position == 73
        elif q_position in {90, 52, 30}:
            return k_position == 96
        elif q_position in {32}:
            return k_position == 48
        elif q_position in {81, 33, 84, 69}:
            return k_position == 63
        elif q_position in {34}:
            return k_position == 18
        elif q_position in {57, 35, 37}:
            return k_position == 53
        elif q_position in {36}:
            return k_position == 54
        elif q_position in {38}:
            return k_position == 84
        elif q_position in {98, 39}:
            return k_position == 67
        elif q_position in {41}:
            return k_position == 82
        elif q_position in {42, 51}:
            return k_position == 80
        elif q_position in {73, 43}:
            return k_position == 52
        elif q_position in {44}:
            return k_position == 98
        elif q_position in {45}:
            return k_position == 33
        elif q_position in {46}:
            return k_position == 25
        elif q_position in {48, 88, 60}:
            return k_position == 58
        elif q_position in {65, 49}:
            return k_position == 23
        elif q_position in {50}:
            return k_position == 61
        elif q_position in {53}:
            return k_position == 14
        elif q_position in {54, 63}:
            return k_position == 86
        elif q_position in {79, 55}:
            return k_position == 16
        elif q_position in {56}:
            return k_position == 32
        elif q_position in {58}:
            return k_position == 94
        elif q_position in {67, 59}:
            return k_position == 66
        elif q_position in {61}:
            return k_position == 12
        elif q_position in {64}:
            return k_position == 64
        elif q_position in {66, 74}:
            return k_position == 92
        elif q_position in {68, 86}:
            return k_position == 43
        elif q_position in {71}:
            return k_position == 21
        elif q_position in {72}:
            return k_position == 91
        elif q_position in {75}:
            return k_position == 46
        elif q_position in {76}:
            return k_position == 44
        elif q_position in {77}:
            return k_position == 89
        elif q_position in {78}:
            return k_position == 45
        elif q_position in {87}:
            return k_position == 31
        elif q_position in {89}:
            return k_position == 74
        elif q_position in {91}:
            return k_position == 28
        elif q_position in {92}:
            return k_position == 77
        elif q_position in {95}:
            return k_position == 62
        elif q_position in {97}:
            return k_position == 71

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 4
        elif q_position in {
            1,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
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
            32,
            33,
            36,
            37,
            38,
            39,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            49,
            50,
            51,
            52,
            53,
            54,
            57,
            59,
            60,
            62,
            63,
            64,
            65,
            67,
            68,
            69,
            70,
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
            return k_position == 5
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {66, 4, 5, 48, 85, 56}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 2
        elif q_position in {34, 61, 7}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 76
        elif q_position in {24}:
            return k_position == 65
        elif q_position in {35}:
            return k_position == 69
        elif q_position in {40}:
            return k_position == 59
        elif q_position in {55}:
            return k_position == 38
        elif q_position in {58}:
            return k_position == 81
        elif q_position in {71}:
            return k_position == 96

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 3, 4, 71, 85, 94}:
            return k_position == 5
        elif q_position in {1, 5}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {
            6,
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
            38,
            39,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            54,
            55,
            56,
            57,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            69,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            80,
            81,
            82,
            83,
            84,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            95,
            96,
            97,
            98,
            99,
        }:
            return k_position == 1
        elif q_position in {37}:
            return k_position == 86
        elif q_position in {43}:
            return k_position == 9
        elif q_position in {53}:
            return k_position == 6
        elif q_position in {58}:
            return k_position == 96
        elif q_position in {68}:
            return k_position == 62
        elif q_position in {79}:
            return k_position == 77
        elif q_position in {93}:
            return k_position == 19

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 5, 9, 41, 82}:
            return k_position == 4
        elif q_position in {32, 1, 2, 89}:
            return k_position == 6
        elif q_position in {
            3,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            18,
            19,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            33,
            35,
            36,
            37,
            38,
            39,
            40,
            42,
            45,
            46,
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
            67,
            68,
            69,
            70,
            71,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            84,
            85,
            86,
            88,
            90,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
        }:
            return k_position == 5
        elif q_position in {4, 72, 44, 47, 83}:
            return k_position == 2
        elif q_position in {34, 66, 6, 7, 73, 43, 17, 20, 87, 25, 91, 60, 57}:
            return k_position == 1

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 97, 66, 68, 7, 71, 73, 44, 49, 91, 29, 30}:
            return token == "1"
        elif position in {1, 92}:
            return token == "<s>"
        elif position in {
            2,
            3,
            4,
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
            45,
            46,
            47,
            48,
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
            67,
            69,
            70,
            72,
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
            93,
            94,
            95,
            96,
            98,
            99,
        }:
            return token == "2"
        elif position in {74, 5, 6}:
            return token == ""

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0}:
            return token == "2"
        elif position in {
            1,
            2,
            3,
            4,
            11,
            13,
            15,
            18,
            20,
            23,
            25,
            28,
            29,
            30,
            32,
            37,
            38,
            40,
            42,
            43,
            44,
            45,
            47,
            48,
            51,
            52,
            57,
            58,
            60,
            62,
            64,
            67,
            71,
            72,
            73,
            74,
            76,
            77,
            79,
            84,
            90,
            91,
            96,
            97,
        }:
            return token == ""
        elif position in {
            5,
            6,
            8,
            9,
            10,
            12,
            14,
            16,
            17,
            19,
            21,
            22,
            26,
            27,
            31,
            33,
            34,
            35,
            36,
            39,
            41,
            46,
            49,
            50,
            53,
            54,
            55,
            56,
            61,
            63,
            65,
            66,
            68,
            69,
            70,
            75,
            78,
            80,
            81,
            82,
            83,
            85,
            86,
            87,
            88,
            89,
            92,
            93,
            94,
            95,
            98,
            99,
        }:
            return token == "4"
        elif position in {24, 59, 7}:
            return token == "1"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 7}:
            return token == "1"
        elif position in {
            1,
            2,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
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
            return token == "0"
        elif position in {3, 4, 5, 6}:
            return token == ""
        elif position in {16, 12}:
            return token == "<s>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {
            0,
            5,
            6,
            8,
            11,
            13,
            14,
            15,
            17,
            18,
            22,
            23,
            24,
            26,
            27,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            39,
            41,
            42,
            43,
            44,
            45,
            47,
            48,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            61,
            62,
            63,
            65,
            67,
            68,
            70,
            71,
            72,
            73,
            74,
            76,
            77,
            80,
            81,
            82,
            83,
            88,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
        }:
            return k_position == 7
        elif q_position in {1, 2}:
            return k_position == 5
        elif q_position in {3, 4, 7, 40, 84, 85, 86, 60}:
            return k_position == 6
        elif q_position in {
            66,
            98,
            99,
            69,
            9,
            10,
            75,
            46,
            78,
            49,
            50,
            19,
            51,
            21,
            59,
            28,
            29,
        }:
            return k_position == 2
        elif q_position in {64, 89, 38, 12, 79, 16, 20, 87, 25}:
            return k_position == 1

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {
            5,
            6,
            8,
            11,
            20,
            36,
            38,
            39,
            43,
            44,
            45,
            56,
            62,
            64,
            65,
            68,
            70,
            71,
            76,
            80,
            85,
            88,
            89,
            94,
            99,
        }:
            return 28
        elif key in {2, 3}:
            return 37
        return 27

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {1, 12, 14, 30, 38, 39, 49, 51, 53, 55, 56, 62, 69, 75, 83, 96, 99}:
            return 43
        elif key in {4, 5}:
            return 14
        elif key in {0, 6}:
            return 88
        return 35

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        return 51

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 18

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 4, 37, 39, 51}:
            return token == "4"
        elif position in {1}:
            return token == "1"
        elif position in {2, 26}:
            return token == "0"
        elif position in {
            3,
            5,
            8,
            10,
            11,
            12,
            13,
            14,
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
            29,
            30,
            31,
            32,
            33,
            34,
            38,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            50,
            52,
            53,
            55,
            56,
            57,
            58,
            59,
            60,
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
            75,
            76,
            77,
            79,
            80,
            82,
            83,
            84,
            86,
            87,
            89,
            90,
            91,
            92,
            94,
            95,
            96,
            98,
        }:
            return token == "</s>"
        elif position in {93, 6}:
            return token == "3"
        elif position in {35, 99, 7, 40, 74, 43, 78, 15, 81, 85, 54, 88, 27, 28, 61}:
            return token == ""
        elif position in {9, 36, 97, 49}:
            return token == "<s>"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 5, 74, 85, 56}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 78
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 76}:
            return k_position == 6
        elif q_position in {4, 7}:
            return k_position == 2
        elif q_position in {60, 6}:
            return k_position == 5
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
            57,
            58,
            59,
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
            75,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
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
            return k_position == 28
        elif q_position in {86, 30}:
            return k_position == 3
        elif q_position in {61}:
            return k_position == 84

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 3
        elif q_position in {1, 3, 6}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {74, 99, 85, 5}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 6
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
        }:
            return k_position == 28

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 2
        elif q_position in {1, 6}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {99, 4, 7, 74, 85}:
            return k_position == 1
        elif q_position in {
            5,
            8,
            9,
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
            55,
            56,
            57,
            58,
            59,
            60,
            61,
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
            86,
            87,
            89,
            90,
            91,
            92,
            94,
            95,
            96,
            97,
            98,
        }:
            return k_position == 6
        elif q_position in {39, 40, 10, 45, 53, 54, 88, 93, 62}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 88

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, token):
        if mlp_0_1_output in {
            0,
            1,
            2,
            4,
            5,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            18,
            19,
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
            39,
            41,
            42,
            43,
            44,
            46,
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
            66,
            67,
            69,
            71,
            73,
            75,
            76,
            77,
            79,
            80,
            81,
            82,
            84,
            86,
            87,
            89,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
        }:
            return token == "2"
        elif mlp_0_1_output in {65, 3, 6, 70, 72, 45, 78, 47, 48, 83, 88, 91}:
            return token == "1"
        elif mlp_0_1_output in {99, 68, 38, 40, 12, 20, 85}:
            return token == "3"
        elif mlp_0_1_output in {17}:
            return token == "0"
        elif mlp_0_1_output in {74}:
            return token == "4"
        elif mlp_0_1_output in {90}:
            return token == ""

    num_attn_1_0_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_1_output, token):
        if mlp_0_1_output in {
            0,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            14,
            17,
            21,
            25,
            26,
            27,
            28,
            30,
            31,
            32,
            34,
            35,
            37,
            42,
            45,
            46,
            47,
            49,
            50,
            51,
            57,
            58,
            61,
            62,
            63,
            64,
            65,
            66,
            70,
            71,
            75,
            77,
            81,
            83,
            84,
            86,
            89,
            90,
            91,
            92,
            93,
            98,
        }:
            return token == "0"
        elif mlp_0_1_output in {
            1,
            8,
            11,
            12,
            13,
            15,
            16,
            18,
            19,
            20,
            22,
            23,
            24,
            36,
            38,
            39,
            41,
            43,
            44,
            52,
            53,
            54,
            55,
            59,
            60,
            67,
            68,
            69,
            72,
            73,
            76,
            78,
            79,
            80,
            82,
            87,
            95,
            96,
            97,
            99,
        }:
            return token == "1"
        elif mlp_0_1_output in {33, 2, 40, 85, 56, 29, 94}:
            return token == "2"
        elif mlp_0_1_output in {48, 88}:
            return token == "</s>"
        elif mlp_0_1_output in {74}:
            return token == "3"

    num_attn_1_1_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 52, 61}:
            return token == "</s>"
        elif mlp_0_0_output in {
            1,
            2,
            3,
            6,
            7,
            9,
            11,
            12,
            16,
            18,
            19,
            20,
            23,
            24,
            25,
            26,
            29,
            30,
            31,
            32,
            36,
            38,
            40,
            42,
            43,
            44,
            46,
            49,
            55,
            56,
            59,
            60,
            63,
            64,
            67,
            68,
            71,
            72,
            73,
            74,
            76,
            77,
            78,
            80,
            82,
            83,
            84,
            85,
            86,
            89,
            91,
            92,
            93,
            94,
            96,
            97,
            99,
        }:
            return token == "0"
        elif mlp_0_0_output in {
            4,
            5,
            10,
            13,
            14,
            15,
            17,
            21,
            22,
            28,
            33,
            34,
            41,
            45,
            48,
            50,
            53,
            57,
            58,
            62,
            65,
            66,
            69,
            70,
            79,
            81,
            90,
            95,
            98,
        }:
            return token == "2"
        elif mlp_0_0_output in {8}:
            return token == "3"
        elif mlp_0_0_output in {35, 37, 39, 75, 51, 54, 87, 88, 27}:
            return token == ""
        elif mlp_0_0_output in {47}:
            return token == "4"

    num_attn_1_2_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 10, 45, 14, 90, 28}:
            return token == "0"
        elif position in {
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            11,
            12,
            13,
            15,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
            24,
            25,
            26,
            27,
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
            89,
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
            return token == "1"
        elif position in {88, 6}:
            return token == ""
        elif position in {74, 43, 21, 56, 29}:
            return token == "2"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, position):
        key = (attn_1_3_output, position)
        if key in {
            ("0", 6),
            ("0", 11),
            ("0", 23),
            ("0", 28),
            ("0", 35),
            ("0", 38),
            ("0", 41),
            ("0", 45),
            ("0", 46),
            ("0", 51),
            ("0", 58),
            ("0", 59),
            ("0", 64),
            ("0", 65),
            ("0", 67),
            ("0", 69),
            ("0", 75),
            ("0", 77),
            ("0", 82),
            ("0", 84),
            ("0", 88),
            ("0", 95),
            ("0", 96),
            ("0", 97),
            ("0", 98),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 12),
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
            ("1", 25),
            ("1", 26),
            ("1", 28),
            ("1", 29),
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
            ("1", 44),
            ("1", 45),
            ("1", 46),
            ("1", 47),
            ("1", 49),
            ("1", 50),
            ("1", 51),
            ("1", 52),
            ("1", 53),
            ("1", 54),
            ("1", 56),
            ("1", 57),
            ("1", 58),
            ("1", 59),
            ("1", 60),
            ("1", 61),
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
            ("1", 81),
            ("1", 82),
            ("1", 83),
            ("1", 84),
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
            ("2", 6),
            ("2", 11),
            ("2", 23),
            ("2", 28),
            ("2", 35),
            ("2", 45),
            ("2", 46),
            ("2", 65),
            ("2", 75),
            ("2", 88),
            ("<s>", 6),
            ("<s>", 11),
            ("<s>", 23),
            ("<s>", 25),
            ("<s>", 31),
            ("<s>", 35),
            ("<s>", 38),
            ("<s>", 41),
            ("<s>", 44),
            ("<s>", 45),
            ("<s>", 46),
            ("<s>", 47),
            ("<s>", 51),
            ("<s>", 54),
            ("<s>", 58),
            ("<s>", 59),
            ("<s>", 61),
            ("<s>", 63),
            ("<s>", 64),
            ("<s>", 65),
            ("<s>", 67),
            ("<s>", 69),
            ("<s>", 77),
            ("<s>", 81),
            ("<s>", 82),
            ("<s>", 84),
            ("<s>", 88),
            ("<s>", 95),
            ("<s>", 96),
            ("<s>", 97),
            ("<s>", 99),
        }:
            return 61
        elif key in {
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 7),
            ("2", 8),
            ("2", 9),
            ("2", 12),
            ("2", 13),
            ("2", 14),
            ("2", 15),
            ("2", 16),
            ("2", 18),
            ("2", 19),
            ("2", 20),
            ("2", 21),
            ("2", 22),
            ("2", 24),
            ("2", 25),
            ("2", 26),
            ("2", 27),
            ("2", 29),
            ("2", 30),
            ("2", 31),
            ("2", 32),
            ("2", 33),
            ("2", 34),
            ("2", 36),
            ("2", 37),
            ("2", 38),
            ("2", 39),
            ("2", 40),
            ("2", 41),
            ("2", 42),
            ("2", 43),
            ("2", 44),
            ("2", 47),
            ("2", 48),
            ("2", 49),
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
            ("2", 63),
            ("2", 64),
            ("2", 66),
            ("2", 67),
            ("2", 68),
            ("2", 69),
            ("2", 71),
            ("2", 72),
            ("2", 73),
            ("2", 74),
            ("2", 76),
            ("2", 77),
            ("2", 78),
            ("2", 79),
            ("2", 80),
            ("2", 81),
            ("2", 82),
            ("2", 83),
            ("2", 84),
            ("2", 86),
            ("2", 87),
            ("2", 89),
            ("2", 91),
            ("2", 92),
            ("2", 93),
            ("2", 94),
            ("2", 95),
            ("2", 96),
            ("2", 97),
            ("2", 98),
            ("2", 99),
            ("<s>", 4),
        }:
            return 32
        elif key in {
            ("0", 1),
            ("0", 3),
            ("0", 24),
            ("0", 43),
            ("0", 54),
            ("0", 55),
            ("0", 74),
            ("0", 85),
            ("1", 85),
            ("2", 85),
            ("3", 85),
            ("<s>", 85),
        }:
            return 82
        elif key in {
            ("2", 0),
            ("2", 10),
            ("2", 17),
            ("2", 50),
            ("2", 70),
            ("2", 90),
            ("3", 0),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 8),
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
            return 77
        return 64

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_1_1_output):
        key = (position, attn_1_1_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (31, "3"),
            (54, "3"),
            (74, "3"),
            (85, "0"),
            (85, "2"),
            (85, "3"),
            (85, "4"),
        }:
            return 71
        return 88

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_1_1_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 1

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_0_output, num_attn_0_1_output):
        key = (num_attn_0_0_output, num_attn_0_1_output)
        return 99

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {
            0,
            3,
            4,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
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
            98,
        }:
            return k_position == 6
        elif q_position in {89, 1, 2, 6}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {40, 18, 11, 61}:
            return k_position == 28
        elif q_position in {26, 99, 85, 74}:
            return k_position == 1

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {
            0,
            2,
            3,
            4,
            11,
            22,
            23,
            25,
            29,
            33,
            36,
            45,
            48,
            49,
            51,
            53,
            56,
            64,
            65,
            69,
            71,
            72,
            76,
            78,
            80,
            82,
            86,
            88,
            90,
            91,
            92,
        }:
            return token == ""
        elif position in {
            1,
            7,
            10,
            21,
            24,
            27,
            31,
            32,
            34,
            35,
            39,
            43,
            47,
            50,
            52,
            54,
            58,
            59,
            60,
            62,
            66,
            73,
            79,
            81,
            87,
            94,
            95,
            97,
        }:
            return token == "2"
        elif position in {98, 67, 5, 38, 8, 40, 75, 13, 18, 26, 61, 63}:
            return token == "</s>"
        elif position in {70, 6}:
            return token == "<s>"
        elif position in {9, 74, 84}:
            return token == "0"
        elif position in {
            89,
            96,
            99,
            68,
            37,
            12,
            44,
            15,
            16,
            17,
            83,
            20,
            85,
            55,
            57,
            30,
        }:
            return token == "1"
        elif position in {93, 28, 77, 14}:
            return token == "4"
        elif position in {41, 42, 19, 46}:
            return token == "3"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {
            0,
            1,
            8,
            10,
            12,
            13,
            14,
            15,
            17,
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
            30,
            32,
            34,
            35,
            37,
            38,
            39,
            40,
            41,
            42,
            44,
            45,
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
            70,
            72,
            73,
            76,
            77,
            78,
            79,
            81,
            82,
            83,
            84,
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
            97,
            98,
        }:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 22
        elif q_position in {33, 3, 36, 68, 99, 71, 74, 43, 85}:
            return k_position == 1
        elif q_position in {75, 4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 0
        elif q_position in {11, 46, 7}:
            return k_position == 28
        elif q_position in {9}:
            return k_position == 93
        elif q_position in {16}:
            return k_position == 75
        elif q_position in {18}:
            return k_position == 58
        elif q_position in {29}:
            return k_position == 95
        elif q_position in {31}:
            return k_position == 34
        elif q_position in {67}:
            return k_position == 77
        elif q_position in {69}:
            return k_position == 14
        elif q_position in {80}:
            return k_position == 11
        elif q_position in {96}:
            return k_position == 85

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            64,
            65,
            66,
            89,
            98,
            70,
            72,
            46,
            78,
            19,
            20,
            52,
            86,
            95,
            25,
            94,
            63,
        }:
            return token == "2"
        elif mlp_0_0_output in {
            1,
            4,
            6,
            7,
            8,
            11,
            13,
            16,
            26,
            29,
            32,
            34,
            38,
            41,
            43,
            50,
            53,
            54,
            56,
            58,
            60,
            62,
            67,
            73,
            75,
            77,
            80,
            81,
            88,
            96,
        }:
            return token == ""
        elif mlp_0_0_output in {
            2,
            3,
            9,
            10,
            12,
            14,
            15,
            17,
            18,
            23,
            27,
            30,
            31,
            36,
            37,
            39,
            42,
            44,
            45,
            47,
            49,
            51,
            55,
            59,
            68,
            69,
            76,
            82,
            83,
            84,
            90,
            91,
            92,
            93,
            97,
            99,
        }:
            return token == "</s>"
        elif mlp_0_0_output in {5, 79, 21, 57, 61}:
            return token == "3"
        elif mlp_0_0_output in {40, 48, 85, 22, 24}:
            return token == "1"
        elif mlp_0_0_output in {28, 87}:
            return token == "<s>"
        elif mlp_0_0_output in {33, 74}:
            return token == "0"
        elif mlp_0_0_output in {35, 71}:
            return token == "<pad>"

    attn_2_3_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_attn_1_0_output, k_attn_1_0_output):
        if q_attn_1_0_output in {"0", "4", "</s>", "3", "<s>", "2", "1"}:
            return k_attn_1_0_output == "0"

    num_attn_2_0_pattern = select(attn_1_0_outputs, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_0_output, attn_1_2_output):
        if mlp_0_0_output in {
            0,
            2,
            4,
            7,
            9,
            11,
            14,
            16,
            17,
            18,
            23,
            25,
            26,
            27,
            30,
            31,
            32,
            34,
            42,
            43,
            45,
            46,
            47,
            49,
            50,
            57,
            58,
            59,
            60,
            61,
            62,
            64,
            66,
            67,
            72,
            74,
            78,
            80,
            81,
            82,
            83,
            89,
            91,
            92,
            94,
            97,
            98,
        }:
            return attn_1_2_output == "0"
        elif mlp_0_0_output in {
            1,
            8,
            12,
            15,
            20,
            22,
            24,
            29,
            33,
            35,
            36,
            37,
            38,
            39,
            40,
            44,
            53,
            54,
            55,
            56,
            63,
            65,
            68,
            69,
            71,
            73,
            75,
            76,
            79,
            85,
            86,
            93,
            95,
            96,
            99,
        }:
            return attn_1_2_output == "1"
        elif mlp_0_0_output in {3, 84, 77}:
            return attn_1_2_output == "2"
        elif mlp_0_0_output in {88, 28, 5, 6}:
            return attn_1_2_output == ""
        elif mlp_0_0_output in {70, 10, 13, 48, 19, 52, 21}:
            return attn_1_2_output == "3"
        elif mlp_0_0_output in {41, 51}:
            return attn_1_2_output == "</s>"
        elif mlp_0_0_output in {90, 87}:
            return attn_1_2_output == "4"

    num_attn_2_1_pattern = select(attn_1_2_outputs, mlp_0_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, token):
        if attn_1_0_output in {"0", "4", "</s>", "<s>", "2", "1"}:
            return token == "3"
        elif attn_1_0_output in {"3"}:
            return token == "</s>"

    num_attn_2_2_pattern = select(tokens, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            1,
            2,
            3,
            7,
            8,
            9,
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
            49,
            50,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            62,
            63,
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
            89,
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
            return token == "0"
        elif mlp_0_0_output in {4, 5, 6, 10, 51, 87, 88, 90, 27, 28}:
            return token == ""
        elif mlp_0_0_output in {64, 45, 47}:
            return token == "2"
        elif mlp_0_0_output in {48, 61}:
            return token == "1"

    num_attn_2_3_pattern = select(tokens, mlp_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, position):
        key = (attn_2_2_output, position)
        if key in {
            ("0", 5),
            ("1", 5),
            ("2", 0),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 7),
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
            ("3", 5),
            ("3", 6),
            ("3", 27),
            ("3", 28),
            ("3", 51),
            ("3", 88),
            ("4", 5),
            ("4", 6),
            ("4", 27),
            ("4", 28),
            ("4", 37),
            ("4", 50),
            ("4", 57),
            ("4", 58),
            ("4", 62),
            ("4", 88),
            ("</s>", 5),
            ("</s>", 28),
            ("<s>", 5),
        }:
            return 87
        elif key in {
            ("0", 6),
            ("1", 0),
            ("1", 4),
            ("1", 6),
            ("1", 7),
            ("1", 9),
            ("1", 12),
            ("1", 16),
            ("1", 17),
            ("1", 18),
            ("1", 21),
            ("1", 22),
            ("1", 27),
            ("1", 28),
            ("1", 29),
            ("1", 33),
            ("1", 34),
            ("1", 38),
            ("1", 39),
            ("1", 41),
            ("1", 43),
            ("1", 44),
            ("1", 48),
            ("1", 51),
            ("1", 53),
            ("1", 55),
            ("1", 56),
            ("1", 57),
            ("1", 58),
            ("1", 60),
            ("1", 62),
            ("1", 65),
            ("1", 66),
            ("1", 67),
            ("1", 68),
            ("1", 70),
            ("1", 71),
            ("1", 72),
            ("1", 73),
            ("1", 78),
            ("1", 86),
            ("1", 87),
            ("1", 88),
            ("1", 94),
            ("1", 98),
            ("4", 0),
            ("4", 3),
            ("4", 4),
            ("4", 7),
            ("4", 21),
            ("4", 43),
            ("4", 51),
            ("4", 68),
            ("4", 71),
            ("</s>", 0),
            ("</s>", 1),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 10),
            ("</s>", 11),
            ("</s>", 12),
            ("</s>", 13),
            ("</s>", 14),
            ("</s>", 15),
            ("</s>", 16),
            ("</s>", 17),
            ("</s>", 18),
            ("</s>", 19),
            ("</s>", 20),
            ("</s>", 21),
            ("</s>", 22),
            ("</s>", 23),
            ("</s>", 24),
            ("</s>", 25),
            ("</s>", 26),
            ("</s>", 27),
            ("</s>", 29),
            ("</s>", 30),
            ("</s>", 31),
            ("</s>", 32),
            ("</s>", 33),
            ("</s>", 34),
            ("</s>", 35),
            ("</s>", 36),
            ("</s>", 37),
            ("</s>", 38),
            ("</s>", 39),
            ("</s>", 40),
            ("</s>", 41),
            ("</s>", 42),
            ("</s>", 43),
            ("</s>", 44),
            ("</s>", 45),
            ("</s>", 46),
            ("</s>", 47),
            ("</s>", 48),
            ("</s>", 49),
            ("</s>", 50),
            ("</s>", 51),
            ("</s>", 52),
            ("</s>", 53),
            ("</s>", 54),
            ("</s>", 55),
            ("</s>", 56),
            ("</s>", 57),
            ("</s>", 58),
            ("</s>", 59),
            ("</s>", 60),
            ("</s>", 61),
            ("</s>", 62),
            ("</s>", 63),
            ("</s>", 64),
            ("</s>", 65),
            ("</s>", 66),
            ("</s>", 67),
            ("</s>", 68),
            ("</s>", 69),
            ("</s>", 70),
            ("</s>", 71),
            ("</s>", 72),
            ("</s>", 73),
            ("</s>", 74),
            ("</s>", 75),
            ("</s>", 76),
            ("</s>", 77),
            ("</s>", 78),
            ("</s>", 79),
            ("</s>", 80),
            ("</s>", 81),
            ("</s>", 82),
            ("</s>", 83),
            ("</s>", 84),
            ("</s>", 85),
            ("</s>", 86),
            ("</s>", 87),
            ("</s>", 88),
            ("</s>", 89),
            ("</s>", 90),
            ("</s>", 91),
            ("</s>", 92),
            ("</s>", 93),
            ("</s>", 94),
            ("</s>", 95),
            ("</s>", 96),
            ("</s>", 97),
            ("</s>", 98),
            ("</s>", 99),
            ("<s>", 0),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 21),
            ("<s>", 27),
            ("<s>", 28),
            ("<s>", 43),
        }:
            return 97
        elif key in {
            ("0", 0),
            ("0", 3),
            ("0", 4),
            ("0", 7),
            ("0", 21),
            ("0", 27),
            ("0", 28),
            ("0", 33),
            ("0", 43),
            ("0", 51),
            ("0", 55),
            ("0", 57),
            ("0", 68),
            ("0", 71),
            ("0", 88),
            ("1", 3),
            ("1", 10),
            ("1", 11),
            ("1", 13),
            ("1", 20),
            ("1", 25),
            ("1", 26),
            ("1", 45),
            ("1", 82),
            ("1", 83),
            ("1", 92),
            ("1", 93),
            ("2", 3),
            ("3", 0),
            ("3", 3),
            ("3", 4),
            ("3", 7),
            ("3", 17),
            ("3", 21),
            ("3", 29),
            ("3", 33),
            ("3", 43),
            ("3", 44),
            ("3", 55),
            ("3", 57),
            ("3", 67),
            ("3", 68),
            ("3", 70),
            ("3", 72),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
            ("4", 16),
            ("4", 17),
            ("4", 18),
            ("4", 19),
            ("4", 20),
            ("4", 22),
            ("4", 23),
            ("4", 24),
            ("4", 26),
            ("4", 29),
            ("4", 30),
            ("4", 31),
            ("4", 32),
            ("4", 33),
            ("4", 34),
            ("4", 35),
            ("4", 36),
            ("4", 38),
            ("4", 39),
            ("4", 40),
            ("4", 41),
            ("4", 42),
            ("4", 44),
            ("4", 45),
            ("4", 46),
            ("4", 47),
            ("4", 48),
            ("4", 49),
            ("4", 52),
            ("4", 53),
            ("4", 55),
            ("4", 56),
            ("4", 59),
            ("4", 60),
            ("4", 61),
            ("4", 63),
            ("4", 65),
            ("4", 66),
            ("4", 67),
            ("4", 69),
            ("4", 70),
            ("4", 72),
            ("4", 73),
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
            ("4", 86),
            ("4", 87),
            ("4", 89),
            ("4", 90),
            ("4", 91),
            ("4", 92),
            ("4", 93),
            ("4", 94),
            ("4", 96),
            ("4", 97),
            ("4", 98),
            ("4", 99),
        }:
            return 18
        elif key in {
            ("0", 1),
            ("0", 2),
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
            ("0", 22),
            ("0", 23),
            ("0", 24),
            ("0", 25),
            ("0", 26),
            ("0", 29),
            ("0", 30),
            ("0", 31),
            ("0", 32),
            ("0", 34),
            ("0", 35),
            ("0", 36),
            ("0", 37),
            ("0", 38),
            ("0", 39),
            ("0", 40),
            ("0", 41),
            ("0", 42),
            ("0", 44),
            ("0", 45),
            ("0", 46),
            ("0", 47),
            ("0", 48),
            ("0", 49),
            ("0", 50),
            ("0", 52),
            ("0", 53),
            ("0", 54),
            ("0", 56),
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
            ("0", 69),
            ("0", 70),
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
            ("1", 1),
            ("1", 74),
            ("1", 85),
            ("2", 1),
            ("2", 85),
            ("3", 1),
            ("3", 15),
            ("3", 25),
            ("3", 54),
            ("3", 74),
            ("3", 85),
            ("4", 1),
            ("4", 2),
            ("4", 15),
            ("4", 25),
            ("4", 54),
            ("4", 64),
            ("4", 74),
            ("4", 85),
            ("4", 95),
        }:
            return 46
        elif key in {
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 13),
            ("3", 18),
            ("3", 19),
            ("3", 22),
            ("3", 26),
            ("3", 30),
            ("3", 31),
            ("3", 42),
            ("3", 45),
            ("3", 48),
            ("3", 50),
            ("3", 53),
            ("3", 56),
            ("3", 58),
            ("3", 60),
            ("3", 62),
            ("3", 63),
            ("3", 73),
            ("3", 78),
            ("3", 80),
            ("3", 86),
            ("3", 87),
            ("3", 90),
            ("3", 91),
            ("3", 92),
            ("3", 93),
            ("3", 98),
        }:
            return 67
        elif key in {("2", 54), ("2", 74)}:
            return 25
        return 32

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, position):
        key = (attn_2_1_output, position)
        if key in {
            ("1", 0),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 12),
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
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("1", 28),
            ("1", 29),
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
            ("1", 45),
            ("1", 46),
            ("1", 47),
            ("1", 48),
            ("1", 49),
            ("1", 50),
            ("1", 51),
            ("1", 52),
            ("1", 53),
            ("1", 54),
            ("1", 55),
            ("1", 56),
            ("1", 57),
            ("1", 58),
            ("1", 59),
            ("1", 60),
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
            ("1", 81),
            ("1", 82),
            ("1", 83),
            ("1", 84),
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
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 7),
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
            ("3", 4),
            ("3", 5),
            ("3", 7),
            ("3", 8),
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
            ("4", 0),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 7),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
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
            ("</s>", 4),
            ("</s>", 5),
            ("</s>", 37),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 37),
        }:
            return 10
        elif key in {
            ("0", 1),
            ("1", 1),
            ("1", 24),
            ("1", 85),
            ("2", 1),
            ("2", 24),
            ("2", 85),
            ("3", 1),
            ("4", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 96
        elif key in {("2", 2)}:
            return 93
        elif key in {("0", 2), ("0", 24), ("0", 85)}:
            return 39
        return 84

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_0_0_output):
        key = (num_attn_2_3_output, num_attn_0_0_output)
        return 26

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_2_0_output):
        key = (num_attn_1_1_output, num_attn_2_0_output)
        return 93

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_0_outputs)
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


print(run(["<s>", "0", "4", "1", "1", "4", "2", "</s>"]))
