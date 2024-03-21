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
        "output/rasp/sort/vocab8maxlen8dvar50/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/sort_weights.csv",
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
        if q_position in {0}:
            return k_position == 4
        elif q_position in {
            1,
            33,
            35,
            37,
            38,
            39,
            12,
            13,
            47,
            48,
            49,
            18,
            22,
            23,
            24,
            26,
            29,
            31,
        }:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {32, 34, 4, 5, 6, 8, 40, 41, 46, 16, 20, 21, 27}:
            return k_position == 1
        elif q_position in {36, 7, 10, 44, 45, 14, 15, 19}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 47
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 35
        elif q_position in {25}:
            return k_position == 17
        elif q_position in {28}:
            return k_position == 40
        elif q_position in {30}:
            return k_position == 23
        elif q_position in {42}:
            return k_position == 36
        elif q_position in {43}:
            return k_position == 46

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 10, 43, 15, 21}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 40, 12, 13, 47, 48, 49, 22, 23, 26, 27, 31}:
            return k_position == 5
        elif q_position in {
            5,
            7,
            9,
            14,
            18,
            19,
            20,
            25,
            28,
            29,
            30,
            35,
            36,
            38,
            39,
            41,
            42,
            44,
            45,
            46,
        }:
            return k_position == 6
        elif q_position in {8, 16}:
            return k_position == 49
        elif q_position in {11}:
            return k_position == 38
        elif q_position in {17}:
            return k_position == 39
        elif q_position in {24}:
            return k_position == 32
        elif q_position in {32}:
            return k_position == 24
        elif q_position in {33}:
            return k_position == 44
        elif q_position in {34}:
            return k_position == 20
        elif q_position in {37}:
            return k_position == 8

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 33, 34, 6, 38, 8, 41, 10, 14, 48, 20, 29}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {35, 4, 11, 43, 44, 46, 16, 49, 18, 21, 23, 27}:
            return k_position == 5
        elif q_position in {5, 7}:
            return k_position == 6
        elif q_position in {24, 9, 17}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 43
        elif q_position in {13}:
            return k_position == 28
        elif q_position in {22, 15}:
            return k_position == 44
        elif q_position in {19}:
            return k_position == 14
        elif q_position in {25}:
            return k_position == 25
        elif q_position in {26}:
            return k_position == 18
        elif q_position in {28}:
            return k_position == 9
        elif q_position in {30}:
            return k_position == 23
        elif q_position in {31}:
            return k_position == 49
        elif q_position in {32, 42}:
            return k_position == 12
        elif q_position in {36}:
            return k_position == 48
        elif q_position in {37}:
            return k_position == 27
        elif q_position in {39}:
            return k_position == 39
        elif q_position in {40}:
            return k_position == 34
        elif q_position in {45}:
            return k_position == 38
        elif q_position in {47}:
            return k_position == 33

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {34, 3, 37, 13, 16}:
            return k_position == 5
        elif q_position in {32, 4, 5, 9, 45, 14, 15, 47, 18, 28, 29, 30}:
            return k_position == 6
        elif q_position in {
            7,
            8,
            10,
            12,
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
            31,
            33,
            35,
            36,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            48,
            49,
        }:
            return k_position == 1
        elif q_position in {11}:
            return k_position == 24

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 43, 28}:
            return token == "2"
        elif position in {
            1,
            2,
            3,
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
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == "0"
        elif position in {4, 5, 6}:
            return token == ""
        elif position in {7}:
            return token == "1"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 5, 6}:
            return token == "1"
        elif position in {1, 9}:
            return token == "<s>"
        elif position in {2}:
            return token == "</s>"
        elif position in {
            3,
            4,
            8,
            12,
            13,
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
        }:
            return token == "2"
        elif position in {7}:
            return token == "3"
        elif position in {10, 11, 14, 49, 28, 29, 30}:
            return token == ""

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 7}:
            return token == "2"
        elif position in {
            1,
            2,
            3,
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
        }:
            return token == "1"
        elif position in {4}:
            return token == "0"
        elif position in {5, 6, 39}:
            return token == ""

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 4, 34, 12, 15, 49, 20, 27}:
            return token == ""
        elif position in {
            5,
            6,
            8,
            9,
            10,
            11,
            13,
            14,
            16,
            17,
            18,
            19,
            21,
            22,
            23,
            24,
            25,
            26,
            28,
            29,
            30,
            31,
            32,
            33,
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
        }:
            return token == "4"
        elif position in {7}:
            return token == "3"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {2, 20, 23, 27, 42}:
            return 49
        elif key in {3, 43}:
            return 37
        elif key in {1}:
            return 4
        return 45

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {1, 2}:
            return 48
        elif key in {5}:
            return 13
        return 19

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        return 10

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(one, num_attn_0_3_output):
        key = (one, num_attn_0_3_output)
        return 38

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1) for k0, k1 in zip(ones, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 44, 45, 47, 49}:
            return position == 19
        elif mlp_0_1_output in {1, 7, 13, 18, 24, 26, 28, 29}:
            return position == 2
        elif mlp_0_1_output in {2}:
            return position == 13
        elif mlp_0_1_output in {19, 3}:
            return position == 1
        elif mlp_0_1_output in {34, 4, 42, 43, 12, 31}:
            return position == 15
        elif mlp_0_1_output in {5}:
            return position == 31
        elif mlp_0_1_output in {6}:
            return position == 37
        elif mlp_0_1_output in {8}:
            return position == 42
        elif mlp_0_1_output in {9, 14}:
            return position == 4
        elif mlp_0_1_output in {10}:
            return position == 35
        elif mlp_0_1_output in {32, 11}:
            return position == 40
        elif mlp_0_1_output in {15}:
            return position == 20
        elif mlp_0_1_output in {16}:
            return position == 22
        elif mlp_0_1_output in {17}:
            return position == 34
        elif mlp_0_1_output in {20}:
            return position == 6
        elif mlp_0_1_output in {21}:
            return position == 0
        elif mlp_0_1_output in {38, 22}:
            return position == 11
        elif mlp_0_1_output in {23}:
            return position == 26
        elif mlp_0_1_output in {25}:
            return position == 36
        elif mlp_0_1_output in {27}:
            return position == 30
        elif mlp_0_1_output in {36, 30}:
            return position == 3
        elif mlp_0_1_output in {33}:
            return position == 25
        elif mlp_0_1_output in {35, 37, 46}:
            return position == 21
        elif mlp_0_1_output in {39}:
            return position == 33
        elif mlp_0_1_output in {40}:
            return position == 29
        elif mlp_0_1_output in {41}:
            return position == 8
        elif mlp_0_1_output in {48}:
            return position == 5

    attn_1_0_pattern = select_closest(positions, mlp_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 1, 5, 6}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3, 7}:
            return k_position == 2
        elif q_position in {
            4,
            10,
            11,
            14,
            16,
            17,
            20,
            23,
            26,
            28,
            29,
            30,
            31,
            33,
            34,
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
        }:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 28
        elif q_position in {35, 9, 12, 13, 45, 47, 18, 24}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 26
        elif q_position in {19}:
            return k_position == 42
        elif q_position in {27, 36, 21}:
            return k_position == 1
        elif q_position in {32, 25, 22}:
            return k_position == 15
        elif q_position in {48}:
            return k_position == 8

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 27, 36, 29}:
            return k_position == 1
        elif q_position in {1, 5}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {
            4,
            8,
            10,
            11,
            12,
            14,
            17,
            18,
            22,
            23,
            25,
            26,
            28,
            31,
            33,
            34,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
        }:
            return k_position == 6
        elif q_position in {6, 7}:
            return k_position == 2
        elif q_position in {9}:
            return k_position == 42
        elif q_position in {13}:
            return k_position == 23
        elif q_position in {32, 35, 38, 45, 15, 16, 49, 20, 21, 30}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 16
        elif q_position in {24}:
            return k_position == 35
        elif q_position in {37}:
            return k_position == 17

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {19, 3}:
            return k_position == 5
        elif q_position in {4, 5, 22, 27, 29}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {32, 33, 34, 7, 39, 40, 41, 43, 18, 24, 30}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 28
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 20
        elif q_position in {37, 11, 45, 14, 15, 49, 21}:
            return k_position == 19
        elif q_position in {35, 12, 31}:
            return k_position == 15
        elif q_position in {13}:
            return k_position == 48
        elif q_position in {16, 42}:
            return k_position == 34
        elif q_position in {17}:
            return k_position == 13
        elif q_position in {20}:
            return k_position == 11
        elif q_position in {48, 25, 46, 23}:
            return k_position == 42
        elif q_position in {26}:
            return k_position == 44
        elif q_position in {28}:
            return k_position == 21
        elif q_position in {36}:
            return k_position == 12
        elif q_position in {38}:
            return k_position == 31
        elif q_position in {44}:
            return k_position == 39
        elif q_position in {47}:
            return k_position == 35

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, attn_0_2_output):
        if mlp_0_1_output in {0, 4, 36, 38, 7, 8, 40, 14, 46, 16, 17, 22, 27, 28, 29}:
            return attn_0_2_output == "0"
        elif mlp_0_1_output in {
            1,
            2,
            3,
            10,
            11,
            12,
            18,
            20,
            21,
            23,
            24,
            25,
            26,
            30,
            31,
            32,
            33,
            34,
            37,
            39,
            41,
            44,
            45,
            49,
        }:
            return attn_0_2_output == "3"
        elif mlp_0_1_output in {35, 5, 6, 9, 42, 13}:
            return attn_0_2_output == "2"
        elif mlp_0_1_output in {15}:
            return attn_0_2_output == "<s>"
        elif mlp_0_1_output in {19}:
            return attn_0_2_output == "</s>"
        elif mlp_0_1_output in {48, 43, 47}:
            return attn_0_2_output == "4"

    num_attn_1_0_pattern = select(attn_0_2_outputs, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
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
            7,
            10,
            11,
            14,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            28,
            31,
            32,
            33,
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
            46,
        }:
            return token == "3"
        elif mlp_0_1_output in {1, 2, 8, 9, 19, 26, 27, 29}:
            return token == ""
        elif mlp_0_1_output in {6}:
            return token == "</s>"
        elif mlp_0_1_output in {36, 43, 12, 47, 49, 30}:
            return token == "4"
        elif mlp_0_1_output in {13}:
            return token == "2"
        elif mlp_0_1_output in {48}:
            return token == "<pad>"

    num_attn_1_1_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_2_output):
        if position in {
            0,
            2,
            3,
            8,
            9,
            10,
            12,
            14,
            15,
            17,
            18,
            20,
            22,
            28,
            33,
            34,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
        }:
            return attn_0_2_output == "2"
        elif position in {48, 1, 36, 29}:
            return attn_0_2_output == "4"
        elif position in {32, 35, 4, 5, 7, 11, 45, 47, 16, 49, 21, 23, 24, 25, 26, 31}:
            return attn_0_2_output == "1"
        elif position in {27, 19, 6}:
            return attn_0_2_output == ""
        elif position in {13}:
            return attn_0_2_output == "0"
        elif position in {37, 30}:
            return attn_0_2_output == "3"

    num_attn_1_2_pattern = select(attn_0_2_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1, 36, 27, 29}:
            return token == "0"
        elif mlp_0_1_output in {2, 34, 9, 47, 48, 49}:
            return token == ""
        elif mlp_0_1_output in {
            3,
            4,
            6,
            10,
            11,
            13,
            14,
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
            28,
            31,
            32,
            33,
            35,
            37,
            38,
            39,
            40,
            41,
            42,
            44,
            45,
            46,
        }:
            return token == "3"
        elif mlp_0_1_output in {5, 15}:
            return token == "</s>"
        elif mlp_0_1_output in {7, 8, 43, 12, 30}:
            return token == "4"
        elif mlp_0_1_output in {21}:
            return token == "2"

    num_attn_1_3_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position):
        key = position
        if key in {3, 8, 20, 28, 29, 36, 42}:
            return 31
        elif key in {2, 13, 21, 45}:
            return 0
        elif key in {1, 27, 47, 48}:
            return 6
        return 8

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in positions]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_1_3_output):
        key = (position, attn_1_3_output)
        if key in {
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (3, "0"),
            (3, "3"),
            (8, "3"),
            (12, "3"),
            (13, "0"),
            (13, "3"),
            (14, "3"),
            (15, "0"),
            (15, "3"),
            (16, "3"),
            (17, "3"),
            (18, "3"),
            (19, "0"),
            (19, "3"),
            (20, "0"),
            (20, "3"),
            (21, "3"),
            (22, "3"),
            (25, "3"),
            (26, "3"),
            (28, "0"),
            (28, "3"),
            (29, "3"),
            (31, "3"),
            (33, "3"),
            (35, "3"),
            (37, "0"),
            (37, "3"),
            (38, "3"),
            (39, "3"),
            (40, "3"),
            (41, "3"),
            (42, "3"),
            (43, "0"),
            (43, "3"),
            (44, "3"),
            (46, "3"),
            (49, "3"),
        }:
            return 13
        elif key in {(6, "0"), (6, "1"), (6, "3"), (6, "4"), (6, "</s>"), (6, "<s>")}:
            return 17
        elif key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
        }:
            return 30
        elif key in {(2, "</s>")}:
            return 42
        return 16

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_1_3_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 13

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_0_output, num_attn_0_1_output):
        key = (num_attn_0_0_output, num_attn_0_1_output)
        return 23

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 22
        elif q_position in {1, 5, 6}:
            return k_position == 2
        elif q_position in {2, 4}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {7}:
            return k_position == 42
        elif q_position in {32, 8, 13, 15, 19, 21, 31}:
            return k_position == 19
        elif q_position in {24, 9}:
            return k_position == 30
        elif q_position in {34, 10, 11, 43, 14, 46, 49, 20, 28}:
            return k_position == 6
        elif q_position in {12}:
            return k_position == 20
        elif q_position in {16}:
            return k_position == 26
        elif q_position in {36, 39, 42, 44, 17, 18, 23, 27, 29}:
            return k_position == 1
        elif q_position in {22}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 49
        elif q_position in {26}:
            return k_position == 28
        elif q_position in {30}:
            return k_position == 39
        elif q_position in {33}:
            return k_position == 25
        elif q_position in {35, 38, 40, 41, 45}:
            return k_position == 15
        elif q_position in {37}:
            return k_position == 43
        elif q_position in {47}:
            return k_position == 14
        elif q_position in {48}:
            return k_position == 44

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 25, 13}:
            return k_position == 13
        elif q_position in {32, 1, 7, 39, 49, 23, 27, 30}:
            return k_position == 6
        elif q_position in {16, 2, 47}:
            return k_position == 0
        elif q_position in {3, 10, 42, 43, 44, 17, 19, 21, 29}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 35
        elif q_position in {26, 5}:
            return k_position == 9
        elif q_position in {38, 6}:
            return k_position == 22
        elif q_position in {8}:
            return k_position == 46
        elif q_position in {9}:
            return k_position == 1
        elif q_position in {11}:
            return k_position == 34
        elif q_position in {12}:
            return k_position == 3
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {35, 15}:
            return k_position == 4
        elif q_position in {18, 34}:
            return k_position == 5
        elif q_position in {20}:
            return k_position == 49
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {24, 48}:
            return k_position == 7
        elif q_position in {28}:
            return k_position == 40
        elif q_position in {31}:
            return k_position == 47
        elif q_position in {33}:
            return k_position == 18
        elif q_position in {36}:
            return k_position == 27
        elif q_position in {37}:
            return k_position == 26
        elif q_position in {40}:
            return k_position == 14
        elif q_position in {41}:
            return k_position == 24
        elif q_position in {45}:
            return k_position == 16
        elif q_position in {46}:
            return k_position == 39

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {
            0,
            7,
            8,
            9,
            10,
            11,
            12,
            14,
            15,
            19,
            20,
            21,
            22,
            23,
            25,
            26,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            37,
            38,
            40,
            41,
            42,
            43,
            46,
            47,
            48,
            49,
        }:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2, 6}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 5, 36, 39, 44, 16, 17, 27}:
            return k_position == 1
        elif q_position in {18, 45, 13}:
            return k_position == 19
        elif q_position in {24}:
            return k_position == 41

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {
            0,
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
            26,
            28,
            30,
            31,
            32,
            33,
            34,
            35,
            37,
            38,
            39,
            40,
            41,
            43,
            44,
            46,
            47,
            48,
            49,
        }:
            return k_position == 6
        elif q_position in {1, 2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4, 5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 0
        elif q_position in {9, 15, 45, 7}:
            return k_position == 19
        elif q_position in {42, 27, 36, 29}:
            return k_position == 1

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_0_1_output):
        if attn_1_2_output in {"0"}:
            return attn_0_1_output == "0"
        elif attn_1_2_output in {"<s>", "</s>", "3", "4", "1"}:
            return attn_0_1_output == "2"
        elif attn_1_2_output in {"2"}:
            return attn_0_1_output == "<s>"

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_3_output, position):
        if attn_1_3_output in {"</s>", "0"}:
            return position == 2
        elif attn_1_3_output in {"<s>", "1", "4", "3"}:
            return position == 1
        elif attn_1_3_output in {"2"}:
            return position == 3

    num_attn_2_1_pattern = select(positions, attn_1_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_1_output, attn_0_0_output):
        if attn_1_1_output in {"</s>", "3", "4", "0", "2"}:
            return attn_0_0_output == "2"
        elif attn_1_1_output in {"1"}:
            return attn_0_0_output == "1"
        elif attn_1_1_output in {"<s>"}:
            return attn_0_0_output == "0"

    num_attn_2_2_pattern = select(attn_0_0_outputs, attn_1_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, token):
        if position in {
            0,
            1,
            3,
            4,
            8,
            9,
            10,
            11,
            13,
            15,
            17,
            20,
            21,
            22,
            23,
            24,
            26,
            28,
            33,
            34,
            35,
            37,
            39,
            41,
            42,
            43,
            46,
            48,
            49,
        }:
            return token == "1"
        elif position in {2}:
            return token == "<s>"
        elif position in {5, 31}:
            return token == "0"
        elif position in {27, 19, 6}:
            return token == ""
        elif position in {40, 18, 7}:
            return token == "2"
        elif position in {32, 36, 38, 12, 44, 14, 45, 16, 47, 25, 29, 30}:
            return token == "3"

    num_attn_2_3_pattern = select(tokens, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, position):
        key = (attn_2_0_output, position)
        if key in {
            ("0", 1),
            ("0", 19),
            ("1", 1),
            ("3", 0),
            ("3", 1),
            ("3", 2),
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
            ("4", 1),
            ("</s>", 1),
        }:
            return 33
        elif key in {
            ("0", 3),
            ("0", 4),
            ("0", 8),
            ("1", 3),
            ("1", 4),
            ("2", 3),
            ("2", 4),
            ("2", 7),
            ("2", 8),
            ("2", 11),
            ("2", 15),
            ("2", 19),
            ("3", 3),
            ("3", 4),
            ("4", 3),
            ("4", 4),
            ("</s>", 3),
            ("</s>", 4),
        }:
            return 8
        elif key in {
            ("2", 0),
            ("2", 2),
            ("2", 9),
            ("2", 10),
            ("2", 12),
            ("2", 13),
            ("2", 14),
            ("2", 16),
            ("2", 17),
            ("2", 18),
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
        }:
            return 40
        elif key in {("2", 5), ("2", 6), ("<s>", 3)}:
            return 28
        elif key in {("2", 1)}:
            return 38
        elif key in {("3", 6)}:
            return 48
        return 30

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, attn_2_2_output):
        key = (position, attn_2_2_output)
        if key in {
            (0, "1"),
            (1, "0"),
            (1, "1"),
            (1, "3"),
            (1, "<s>"),
            (2, "1"),
            (2, "3"),
            (3, "1"),
            (3, "3"),
            (4, "1"),
            (7, "1"),
            (8, "1"),
            (8, "3"),
            (9, "1"),
            (10, "1"),
            (11, "1"),
            (12, "1"),
            (12, "3"),
            (13, "1"),
            (13, "3"),
            (14, "1"),
            (15, "1"),
            (16, "1"),
            (17, "1"),
            (18, "1"),
            (19, "1"),
            (20, "1"),
            (21, "1"),
            (21, "3"),
            (22, "1"),
            (23, "1"),
            (24, "1"),
            (24, "3"),
            (25, "1"),
            (26, "1"),
            (26, "3"),
            (27, "1"),
            (28, "1"),
            (29, "1"),
            (30, "1"),
            (31, "1"),
            (32, "1"),
            (33, "1"),
            (33, "3"),
            (34, "1"),
            (35, "1"),
            (35, "3"),
            (36, "1"),
            (37, "1"),
            (37, "3"),
            (38, "1"),
            (38, "3"),
            (39, "1"),
            (40, "1"),
            (41, "1"),
            (41, "3"),
            (42, "1"),
            (42, "3"),
            (43, "1"),
            (44, "1"),
            (44, "3"),
            (45, "1"),
            (46, "1"),
            (46, "3"),
            (47, "1"),
            (48, "1"),
            (48, "3"),
            (49, "1"),
        }:
            return 40
        elif key in {(6, "1")}:
            return 43
        return 21

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(positions, attn_2_2_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0, 1}:
            return 36
        return 33

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_0_3_output, num_attn_2_1_output):
        key = (num_attn_0_3_output, num_attn_2_1_output)
        return 39

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_2_1_outputs)
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
