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
        "output/rasp/most_freq/vocab8maxlen8dvar50/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/most_freq_weights.csv",
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
    def predicate_0_0(position, token):
        if position in {0}:
            return token == "2"
        elif position in {
            1,
            5,
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
        }:
            return token == "5"
        elif position in {2, 7}:
            return token == "4"
        elif position in {3, 4, 6}:
            return token == "0"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 4, 5, 23}:
            return token == "4"
        elif position in {1, 2}:
            return token == "0"
        elif position in {3}:
            return token == "1"
        elif position in {6, 7}:
            return token == "3"
        elif position in {
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
            20,
            21,
            22,
            24,
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
            36,
            37,
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
        }:
            return token == ""
        elif position in {27, 19, 38}:
            return token == "5"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {8, 16}:
            return k_position == 35
        elif q_position in {9}:
            return k_position == 37
        elif q_position in {10, 39}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 43
        elif q_position in {43, 12}:
            return k_position == 26
        elif q_position in {13}:
            return k_position == 25
        elif q_position in {29, 14, 22}:
            return k_position == 40
        elif q_position in {24, 15}:
            return k_position == 16
        elif q_position in {17, 44}:
            return k_position == 34
        elif q_position in {48, 18}:
            return k_position == 44
        elif q_position in {19, 20, 31}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 31
        elif q_position in {49, 42, 23}:
            return k_position == 10
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {26, 28, 46}:
            return k_position == 9
        elif q_position in {27}:
            return k_position == 32
        elif q_position in {30}:
            return k_position == 17
        elif q_position in {32, 37}:
            return k_position == 20
        elif q_position in {33}:
            return k_position == 42
        elif q_position in {34}:
            return k_position == 22
        elif q_position in {40, 35}:
            return k_position == 23
        elif q_position in {36}:
            return k_position == 48
        elif q_position in {38}:
            return k_position == 29
        elif q_position in {41}:
            return k_position == 41
        elif q_position in {45}:
            return k_position == 38
        elif q_position in {47}:
            return k_position == 12

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 4}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {32, 6, 8, 43, 14, 48, 24, 27, 28, 30}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 0
        elif q_position in {9, 25}:
            return k_position == 11
        elif q_position in {17, 10, 46}:
            return k_position == 42
        elif q_position in {11}:
            return k_position == 41
        elif q_position in {12}:
            return k_position == 45
        elif q_position in {13, 23}:
            return k_position == 46
        elif q_position in {15}:
            return k_position == 20
        elif q_position in {16, 39}:
            return k_position == 37
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19, 36}:
            return k_position == 10
        elif q_position in {20}:
            return k_position == 12
        elif q_position in {21}:
            return k_position == 23
        elif q_position in {35, 22}:
            return k_position == 31
        elif q_position in {49, 26}:
            return k_position == 9
        elif q_position in {42, 29}:
            return k_position == 24
        elif q_position in {31}:
            return k_position == 22
        elif q_position in {33}:
            return k_position == 13
        elif q_position in {34, 44}:
            return k_position == 36
        elif q_position in {37}:
            return k_position == 26
        elif q_position in {38}:
            return k_position == 28
        elif q_position in {40}:
            return k_position == 47
        elif q_position in {41}:
            return k_position == 21
        elif q_position in {45}:
            return k_position == 33
        elif q_position in {47}:
            return k_position == 34

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 3, 4, 5}:
            return token == "4"
        elif position in {1}:
            return token == "1"
        elif position in {2, 35, 6, 9, 18, 26}:
            return token == "<s>"
        elif position in {
            7,
            8,
            10,
            11,
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
            36,
            37,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            47,
            48,
            49,
        }:
            return token == "5"
        elif position in {12, 13, 46}:
            return token == "2"
        elif position in {38}:
            return token == ""

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 34}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {
            2,
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
            20,
            21,
            22,
            23,
            25,
            26,
            27,
            28,
            30,
            32,
            33,
            35,
            36,
            37,
            39,
            41,
            42,
            44,
            46,
            47,
            49,
        }:
            return token == ""
        elif position in {3, 4, 5, 38, 19, 29}:
            return token == "3"
        elif position in {48, 45, 6}:
            return token == "<s>"
        elif position in {24, 43, 31}:
            return token == "5"
        elif position in {40}:
            return token == "4"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1}:
            return token == "5"
        elif position in {
            2,
            3,
            4,
            5,
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
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {6}:
            return token == "<s>"
        elif position in {7}:
            return token == "2"
        elif position in {37}:
            return token == "0"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 4, 5, 6}:
            return token == "<s>"
        elif position in {1}:
            return token == "0"
        elif position in {
            2,
            3,
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
        }:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, position):
        key = (attn_0_0_output, position)
        if key in {
            ("5", 1),
            ("5", 2),
            ("5", 3),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 13),
            ("5", 14),
            ("5", 15),
            ("5", 16),
            ("5", 17),
            ("5", 23),
            ("5", 24),
            ("5", 25),
            ("5", 27),
            ("5", 28),
            ("5", 29),
            ("5", 30),
            ("5", 31),
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
        }:
            return 4
        elif key in {
            ("4", 0),
            ("5", 12),
            ("5", 18),
            ("5", 19),
            ("5", 20),
            ("5", 21),
            ("5", 22),
            ("5", 26),
            ("5", 32),
            ("5", 45),
            ("5", 49),
        }:
            return 0
        elif key in {("0", 6), ("2", 6), ("3", 6), ("4", 6), ("5", 6), ("5", 7)}:
            return 38
        elif key in {
            ("1", 6),
            ("1", 9),
            ("1", 10),
            ("1", 12),
            ("1", 13),
            ("1", 16),
            ("1", 17),
            ("1", 18),
            ("1", 19),
            ("1", 20),
            ("1", 21),
            ("1", 22),
            ("1", 24),
            ("1", 26),
            ("1", 29),
            ("1", 32),
            ("1", 35),
            ("1", 36),
            ("1", 38),
            ("1", 39),
            ("1", 40),
            ("1", 45),
            ("1", 48),
            ("1", 49),
            ("<s>", 4),
            ("<s>", 6),
            ("<s>", 12),
            ("<s>", 16),
            ("<s>", 19),
            ("<s>", 20),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 26),
            ("<s>", 32),
            ("<s>", 45),
            ("<s>", 49),
        }:
            return 27
        elif key in {("3", 4), ("4", 4), ("5", 4)}:
            return 36
        elif key in {("0", 0), ("1", 0), ("3", 0), ("5", 0), ("<s>", 0)}:
            return 21
        elif key in {("1", 4), ("2", 4)}:
            return 37
        elif key in {("5", 5)}:
            return 15
        elif key in {("2", 0)}:
            return 11
        return 9

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_1_output):
        key = (position, attn_0_1_output)
        if key in {
            (8, "<s>"),
            (9, "<s>"),
            (10, "2"),
            (10, "<s>"),
            (11, "<s>"),
            (12, "<s>"),
            (13, "<s>"),
            (14, "2"),
            (14, "<s>"),
            (15, "<s>"),
            (16, "<s>"),
            (17, "<s>"),
            (18, "2"),
            (18, "<s>"),
            (19, "<s>"),
            (20, "<s>"),
            (21, "<s>"),
            (22, "<s>"),
            (23, "<s>"),
            (24, "0"),
            (24, "2"),
            (24, "5"),
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
            (35, "<s>"),
            (36, "2"),
            (36, "<s>"),
            (37, "<s>"),
            (38, "2"),
            (38, "<s>"),
            (39, "<s>"),
            (40, "<s>"),
            (41, "<s>"),
            (42, "0"),
            (42, "2"),
            (42, "5"),
            (42, "<s>"),
            (43, "<s>"),
            (44, "<s>"),
            (45, "<s>"),
            (46, "<s>"),
            (47, "2"),
            (47, "<s>"),
            (48, "<s>"),
            (49, "<s>"),
        }:
            return 46
        elif key in {
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
            (7, "0"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
            (9, "3"),
            (9, "5"),
            (16, "5"),
            (34, "5"),
            (43, "3"),
        }:
            return 20
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
            (3, "1"),
            (3, "2"),
            (3, "4"),
            (7, "1"),
            (8, "1"),
            (9, "1"),
            (10, "1"),
            (11, "1"),
            (12, "1"),
            (13, "1"),
            (14, "1"),
            (15, "1"),
            (16, "1"),
            (17, "1"),
            (18, "1"),
            (19, "1"),
            (20, "1"),
            (21, "1"),
            (22, "1"),
            (23, "1"),
            (24, "1"),
            (25, "1"),
            (26, "1"),
            (27, "1"),
            (28, "1"),
            (29, "1"),
            (30, "1"),
            (31, "1"),
            (32, "1"),
            (33, "1"),
            (34, "1"),
            (35, "1"),
            (36, "1"),
            (37, "1"),
            (38, "1"),
            (39, "1"),
            (40, "1"),
            (41, "1"),
            (42, "1"),
            (43, "1"),
            (44, "1"),
            (45, "1"),
            (46, "1"),
            (47, "1"),
            (48, "1"),
            (49, "1"),
        }:
            return 14
        elif key in {(1, "0"), (1, "1"), (1, "2"), (1, "4"), (1, "5"), (1, "<s>")}:
            return 42
        return 30

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_1_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 3

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 40

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 2, 4, 5, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 5
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
        }:
            return k_position == 27

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"2", "5", "0", "1"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"<s>", "4"}:
            return k_token == "4"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {38, 3, 36, 6}:
            return k_position == 3
        elif q_position in {4, 5}:
            return k_position == 1
        elif q_position in {37, 7, 8, 18, 27}:
            return k_position == 7
        elif q_position in {
            9,
            10,
            11,
            12,
            13,
            14,
            15,
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
            32,
            33,
            34,
            35,
            39,
            41,
            42,
            43,
            46,
            47,
            48,
            49,
        }:
            return k_position == 27
        elif q_position in {16, 28}:
            return k_position == 8
        elif q_position in {17}:
            return k_position == 26
        elif q_position in {31}:
            return k_position == 47
        elif q_position in {40}:
            return k_position == 25
        elif q_position in {44}:
            return k_position == 41
        elif q_position in {45}:
            return k_position == 15

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {
            0,
            4,
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
            28,
            29,
            30,
            31,
            32,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            43,
            45,
            46,
            47,
            48,
            49,
        }:
            return k_position == 27
        elif q_position in {1, 34, 33, 6, 10}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {42}:
            return k_position == 46
        elif q_position in {44}:
            return k_position == 37

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 9, 32, 38}:
            return token == "5"
        elif position in {1}:
            return token == "3"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            12,
            13,
            16,
            19,
            23,
            25,
            34,
            36,
            39,
            42,
            44,
            49,
        }:
            return token == ""
        elif position in {33, 37, 10, 43, 47, 17, 18, 20, 22, 24, 30}:
            return token == "2"
        elif position in {48, 11, 15}:
            return token == "4"
        elif position in {14, 31}:
            return token == "<s>"
        elif position in {35, 40, 41, 45, 46, 21, 26, 28, 29}:
            return token == "1"
        elif position in {27}:
            return token == "0"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"0"}:
            return attn_0_1_output == "0"
        elif attn_0_3_output in {"<s>", "5", "1", "3", "2"}:
            return attn_0_1_output == ""
        elif attn_0_3_output in {"4"}:
            return attn_0_1_output == "4"

    num_attn_1_1_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, token):
        if attn_0_3_output in {"4", "<s>", "5", "1", "2", "0"}:
            return token == ""
        elif attn_0_3_output in {"3"}:
            return token == "3"

    num_attn_1_2_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 27}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {
            2,
            6,
            8,
            10,
            12,
            13,
            14,
            15,
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
            49,
        }:
            return token == ""
        elif position in {3, 4, 5}:
            return token == "1"
        elif position in {7, 9, 11, 16, 48, 20, 30}:
            return token == "5"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, position):
        key = (attn_1_2_output, position)
        if key in {
            ("4", 25),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
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
            ("<s>", 29),
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
            ("<s>", 45),
            ("<s>", 46),
            ("<s>", 47),
            ("<s>", 48),
            ("<s>", 49),
        }:
            return 15
        return 41

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_0_output, mlp_0_0_output):
        key = (num_mlp_0_0_output, mlp_0_0_output)
        return 33

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 15

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output, num_attn_1_2_output):
        key = (num_attn_0_3_output, num_attn_1_2_output)
        return 5

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_3_output, position):
        if attn_0_3_output in {"2", "0", "1"}:
            return position == 1
        elif attn_0_3_output in {"3"}:
            return position == 5
        elif attn_0_3_output in {"4"}:
            return position == 31
        elif attn_0_3_output in {"5"}:
            return position == 0
        elif attn_0_3_output in {"<s>"}:
            return position == 27

    attn_2_0_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"3", "5", "0", "1"}:
            return k_token == "<s>"
        elif q_token in {"2"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, mlp_0_0_output):
        if token in {"0"}:
            return mlp_0_0_output == 49
        elif token in {"1"}:
            return mlp_0_0_output == 36
        elif token in {"2"}:
            return mlp_0_0_output == 37
        elif token in {"3", "4"}:
            return mlp_0_0_output == 4
        elif token in {"5"}:
            return mlp_0_0_output == 38
        elif token in {"<s>"}:
            return mlp_0_0_output == 28

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(position, token):
        if position in {
            0,
            8,
            9,
            10,
            11,
            12,
            13,
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
            26,
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
            40,
            41,
            42,
            43,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {1, 5, 6}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {3, 4}:
            return token == "5"
        elif position in {7}:
            return token == "4"
        elif position in {38, 39, 44, 14, 22, 27}:
            return token == "<s>"

    attn_2_3_pattern = select_closest(tokens, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"5", "1", "3", "2", "0"}:
            return attn_0_1_output == "4"
        elif attn_0_3_output in {"4"}:
            return attn_0_1_output == "1"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_1_output == ""

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_0_output, attn_0_1_output):
        if attn_0_0_output in {"<s>", "5", "1", "2", "0"}:
            return attn_0_1_output == "3"
        elif attn_0_0_output in {"3"}:
            return attn_0_1_output == "5"
        elif attn_0_0_output in {"4"}:
            return attn_0_1_output == ""

    num_attn_2_1_pattern = select(attn_0_1_outputs, attn_0_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_3_output, token):
        if attn_0_3_output in {"4", "5", "3", "2", "0"}:
            return token == "1"
        elif attn_0_3_output in {"1"}:
            return token == "0"
        elif attn_0_3_output in {"<s>"}:
            return token == "4"

    num_attn_2_2_pattern = select(tokens, attn_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, attn_1_1_output):
        if position in {
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            10,
            12,
            13,
            14,
            18,
            19,
            21,
            24,
            26,
            28,
            29,
            30,
            32,
            34,
            35,
            36,
            37,
            39,
            42,
            46,
            47,
        }:
            return attn_1_1_output == ""
        elif position in {1}:
            return attn_1_1_output == "4"
        elif position in {33, 38, 8, 11, 43, 44, 45, 16, 22, 25}:
            return attn_1_1_output == "5"
        elif position in {9, 41, 48, 20, 23, 31}:
            return attn_1_1_output == "2"
        elif position in {17, 49, 15}:
            return attn_1_1_output == "3"
        elif position in {27}:
            return attn_1_1_output == "0"
        elif position in {40}:
            return attn_1_1_output == "<s>"

    num_attn_2_3_pattern = select(attn_1_1_outputs, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_1_0_output, attn_2_1_output):
        key = (mlp_1_0_output, attn_2_1_output)
        return 7

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_1_0_outputs, attn_2_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_2_output, num_mlp_0_1_output):
        key = (attn_1_2_output, num_mlp_0_1_output)
        return 7

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, num_mlp_0_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_2_output, num_attn_1_0_output):
        key = (num_attn_0_2_output, num_attn_1_0_output)
        return 1

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output):
        key = num_attn_2_1_output
        return 1

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_1_outputs]
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
