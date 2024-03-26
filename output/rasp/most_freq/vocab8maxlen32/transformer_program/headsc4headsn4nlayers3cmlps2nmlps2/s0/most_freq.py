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
        "output/rasp/most_freq/vocab8maxlen32/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/most_freq_weights.csv",
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
        if position in {
            0,
            1,
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
            21,
            23,
            26,
        }:
            return token == "4"
        elif position in {2}:
            return token == "0"
        elif position in {3, 4, 20, 22, 24, 25, 27, 28, 29, 30, 31}:
            return token == "2"
        elif position in {5, 6}:
            return token == "5"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 7, 11, 12, 22}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3, 20}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {25, 19, 5, 15}:
            return k_position == 8
        elif q_position in {6, 14}:
            return k_position == 10
        elif q_position in {8, 16}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 3
        elif q_position in {10}:
            return k_position == 17
        elif q_position in {13}:
            return k_position == 9
        elif q_position in {17}:
            return k_position == 25
        elif q_position in {18}:
            return k_position == 24
        elif q_position in {21, 30}:
            return k_position == 7
        elif q_position in {28, 23}:
            return k_position == 15
        elif q_position in {24, 29}:
            return k_position == 13
        elif q_position in {26}:
            return k_position == 14
        elif q_position in {27}:
            return k_position == 19
        elif q_position in {31}:
            return k_position == 30

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {
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
            19,
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
        }:
            return k_position == 23
        elif q_position in {4}:
            return k_position == 22
        elif q_position in {5}:
            return k_position == 17
        elif q_position in {6}:
            return k_position == 20
        elif q_position in {10}:
            return k_position == 10
        elif q_position in {18, 20}:
            return k_position == 3
        elif q_position in {21}:
            return k_position == 4

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 7, 9, 11, 14, 15, 17, 18}:
            return token == "4"
        elif position in {1, 2, 3, 20, 22, 23, 24, 27, 28, 29, 30, 31}:
            return token == "2"
        elif position in {4}:
            return token == "5"
        elif position in {5, 6}:
            return token == "3"
        elif position in {8, 10}:
            return token == "1"
        elif position in {16, 12}:
            return token == "<pad>"
        elif position in {13}:
            return token == ""
        elif position in {25, 26, 19}:
            return token == "<s>"
        elif position in {21}:
            return token == "0"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 1}:
            return token == "4"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "2"
        elif position in {
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
        }:
            return token == ""

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 5, 6, 9, 12, 15, 17}:
            return token == "<s>"
        elif position in {1}:
            return token == "3"
        elif position in {
            2,
            3,
            11,
            14,
            16,
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
        }:
            return token == ""
        elif position in {4, 13}:
            return token == "<pad>"
        elif position in {8, 10, 7}:
            return token == "1"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 4, 5}:
            return token == "<pad>"
        elif position in {1, 7, 9, 13, 15, 18, 21, 25, 31}:
            return token == "1"
        elif position in {2, 3, 6}:
            return token == ""
        elif position in {8, 10, 12, 14, 22, 26, 28, 29}:
            return token == "5"
        elif position in {17, 11}:
            return token == "3"
        elif position in {16, 19, 23, 24, 30}:
            return token == "2"
        elif position in {27, 20}:
            return token == "4"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 2}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {3, 4, 5, 6}:
            return token == "0"
        elif position in {
            7,
            8,
            9,
            10,
            11,
            12,
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
            26,
            27,
            28,
            29,
            30,
            31,
        }:
            return token == ""
        elif position in {16, 24, 25}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, position):
        key = (attn_0_0_output, position)
        if key in {
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
            ("<s>", 17),
            ("<s>", 31),
        }:
            return 12
        elif key in {
            ("0", 28),
            ("2", 28),
            ("3", 28),
            ("4", 9),
            ("4", 16),
            ("4", 21),
            ("4", 23),
            ("4", 28),
            ("4", 29),
            ("4", 30),
        }:
            return 8
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output):
        key = attn_0_0_output
        if key in {""}:
            return 17
        return 3

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in attn_0_0_outputs]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0}:
            return 14
        return 28

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 1

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"3", "1", "4", "0", "2"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"0"}:
            return position == 1
        elif token in {"1", "2"}:
            return position == 5
        elif token in {"4", "3"}:
            return position == 6
        elif token in {"5"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 30

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, positions)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {0, 1}:
            return token == "4"
        elif position in {2, 15}:
            return token == "2"
        elif position in {3, 4, 5, 6, 30}:
            return token == "3"
        elif position in {8, 7}:
            return token == "1"
        elif position in {9, 13, 14}:
            return token == "5"
        elif position in {
            10,
            11,
            16,
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
            29,
            31,
        }:
            return token == "0"
        elif position in {18, 12}:
            return token == ""

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0, 2, 3, 4, 5}:
            return token == "1"
        elif position in {1, 11, 12, 14, 16, 18, 20, 25, 26, 28, 29, 31}:
            return token == "2"
        elif position in {6}:
            return token == "<s>"
        elif position in {7, 8, 9, 10, 13, 17, 19, 22, 23, 24}:
            return token == "5"
        elif position in {27, 21, 30, 15}:
            return token == "0"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 5, 9, 10, 11}:
            return token == "<s>"
        elif position in {8, 1}:
            return token == "4"
        elif position in {
            2,
            3,
            4,
            12,
            13,
            14,
            15,
            16,
            21,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
            31,
        }:
            return token == ""
        elif position in {6}:
            return token == "5"
        elif position in {7, 17, 18, 19, 20, 22, 29}:
            return token == "<pad>"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_0_output):
        if position in {
            0,
            3,
            4,
            5,
            6,
            7,
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
            26,
            27,
            29,
            31,
        }:
            return attn_0_0_output == ""
        elif position in {1}:
            return attn_0_0_output == "2"
        elif position in {2, 19, 28, 30}:
            return attn_0_0_output == "<pad>"
        elif position in {8, 13, 14}:
            return attn_0_0_output == "4"
        elif position in {9}:
            return attn_0_0_output == "1"
        elif position in {10, 11, 12}:
            return attn_0_0_output == "5"

    num_attn_1_1_pattern = select(attn_0_0_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 7, 8, 9, 14}:
            return token == "<s>"
        elif position in {1}:
            return token == "0"
        elif position in {2, 10, 29}:
            return token == "<pad>"
        elif position in {
            3,
            4,
            5,
            6,
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
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
            31,
        }:
            return token == ""

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 9, 6}:
            return token == "5"
        elif position in {1}:
            return token == "1"
        elif position in {2, 3, 4, 5, 7, 8}:
            return token == "<s>"
        elif position in {
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
        }:
            return token == ""

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position):
        key = position
        if key in {0, 2, 3}:
            return 3
        elif key in {4}:
            return 7
        elif key in {1}:
            return 26
        return 27

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in positions]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_1_output, mlp_0_0_output):
        key = (num_mlp_0_1_output, mlp_0_0_output)
        return 31

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_1_0_output):
        key = (num_attn_1_2_output, num_attn_1_0_output)
        return 23

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (4, 40),
            (4, 41),
            (4, 42),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 23),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (5, 32),
            (5, 33),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 37),
            (5, 38),
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (7, 32),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (7, 40),
            (7, 41),
            (7, 42),
            (7, 43),
            (7, 44),
            (7, 45),
            (7, 46),
            (7, 47),
            (7, 48),
            (7, 49),
            (7, 50),
            (7, 51),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (8, 32),
            (8, 33),
            (8, 34),
            (8, 35),
            (8, 36),
            (8, 37),
            (8, 38),
            (8, 39),
            (8, 40),
            (8, 41),
            (8, 42),
            (8, 43),
            (8, 44),
            (8, 45),
            (8, 46),
            (8, 47),
            (8, 48),
            (8, 49),
            (8, 50),
            (8, 51),
            (8, 52),
            (8, 53),
            (8, 54),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 38),
            (9, 39),
            (9, 40),
            (9, 41),
            (9, 42),
            (9, 43),
            (9, 44),
            (9, 45),
            (9, 46),
            (9, 47),
            (9, 48),
            (9, 49),
            (9, 50),
            (9, 51),
            (9, 52),
            (9, 53),
            (9, 54),
            (9, 55),
            (9, 56),
            (9, 57),
            (10, 31),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 36),
            (10, 37),
            (10, 38),
            (10, 39),
            (10, 40),
            (10, 41),
            (10, 42),
            (10, 43),
            (10, 44),
            (10, 45),
            (10, 46),
            (10, 47),
            (10, 48),
            (10, 49),
            (10, 50),
            (10, 51),
            (10, 52),
            (10, 53),
            (10, 54),
            (10, 55),
            (10, 56),
            (10, 57),
            (10, 58),
            (10, 59),
            (10, 60),
            (11, 34),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 38),
            (11, 39),
            (11, 40),
            (11, 41),
            (11, 42),
            (11, 43),
            (11, 44),
            (11, 45),
            (11, 46),
            (11, 47),
            (11, 48),
            (11, 49),
            (11, 50),
            (11, 51),
            (11, 52),
            (11, 53),
            (11, 54),
            (11, 55),
            (11, 56),
            (11, 57),
            (11, 58),
            (11, 59),
            (11, 60),
            (11, 61),
            (11, 62),
            (11, 63),
            (12, 37),
            (12, 38),
            (12, 39),
            (12, 40),
            (12, 41),
            (12, 42),
            (12, 43),
            (12, 44),
            (12, 45),
            (12, 46),
            (12, 47),
            (12, 48),
            (12, 49),
            (12, 50),
            (12, 51),
            (12, 52),
            (12, 53),
            (12, 54),
            (12, 55),
            (12, 56),
            (12, 57),
            (12, 58),
            (12, 59),
            (12, 60),
            (12, 61),
            (12, 62),
            (12, 63),
            (13, 40),
            (13, 41),
            (13, 42),
            (13, 43),
            (13, 44),
            (13, 45),
            (13, 46),
            (13, 47),
            (13, 48),
            (13, 49),
            (13, 50),
            (13, 51),
            (13, 52),
            (13, 53),
            (13, 54),
            (13, 55),
            (13, 56),
            (13, 57),
            (13, 58),
            (13, 59),
            (13, 60),
            (13, 61),
            (13, 62),
            (13, 63),
            (14, 43),
            (14, 44),
            (14, 45),
            (14, 46),
            (14, 47),
            (14, 48),
            (14, 49),
            (14, 50),
            (14, 51),
            (14, 52),
            (14, 53),
            (14, 54),
            (14, 55),
            (14, 56),
            (14, 57),
            (14, 58),
            (14, 59),
            (14, 60),
            (14, 61),
            (14, 62),
            (14, 63),
            (15, 46),
            (15, 47),
            (15, 48),
            (15, 49),
            (15, 50),
            (15, 51),
            (15, 52),
            (15, 53),
            (15, 54),
            (15, 55),
            (15, 56),
            (15, 57),
            (15, 58),
            (15, 59),
            (15, 60),
            (15, 61),
            (15, 62),
            (15, 63),
            (16, 49),
            (16, 50),
            (16, 51),
            (16, 52),
            (16, 53),
            (16, 54),
            (16, 55),
            (16, 56),
            (16, 57),
            (16, 58),
            (16, 59),
            (16, 60),
            (16, 61),
            (16, 62),
            (16, 63),
            (17, 52),
            (17, 53),
            (17, 54),
            (17, 55),
            (17, 56),
            (17, 57),
            (17, 58),
            (17, 59),
            (17, 60),
            (17, 61),
            (17, 62),
            (17, 63),
            (18, 55),
            (18, 56),
            (18, 57),
            (18, 58),
            (18, 59),
            (18, 60),
            (18, 61),
            (18, 62),
            (18, 63),
            (19, 58),
            (19, 59),
            (19, 60),
            (19, 61),
            (19, 62),
            (19, 63),
            (20, 61),
            (20, 62),
            (20, 63),
        }:
            return 25
        elif key in {
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 48),
            (1, 49),
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (1, 60),
            (1, 61),
            (1, 62),
            (1, 63),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (2, 50),
            (2, 51),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (2, 60),
            (2, 61),
            (2, 62),
            (2, 63),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
            (3, 48),
            (3, 49),
            (3, 50),
            (3, 51),
            (3, 52),
            (3, 53),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 57),
            (3, 58),
            (3, 59),
            (3, 60),
            (3, 61),
            (3, 62),
            (3, 63),
            (4, 43),
            (4, 44),
            (4, 45),
            (4, 46),
            (4, 47),
            (4, 48),
            (4, 49),
            (4, 50),
            (4, 51),
            (4, 52),
            (4, 53),
            (4, 54),
            (4, 55),
            (4, 56),
            (4, 57),
            (4, 58),
            (4, 59),
            (4, 60),
            (4, 61),
            (4, 62),
            (4, 63),
            (5, 46),
            (5, 47),
            (5, 48),
            (5, 49),
            (5, 50),
            (5, 51),
            (5, 52),
            (5, 53),
            (5, 54),
            (5, 55),
            (5, 56),
            (5, 57),
            (5, 58),
            (5, 59),
            (5, 60),
            (5, 61),
            (5, 62),
            (5, 63),
            (6, 49),
            (6, 50),
            (6, 51),
            (6, 52),
            (6, 53),
            (6, 54),
            (6, 55),
            (6, 56),
            (6, 57),
            (6, 58),
            (6, 59),
            (6, 60),
            (6, 61),
            (6, 62),
            (6, 63),
            (7, 52),
            (7, 53),
            (7, 54),
            (7, 55),
            (7, 56),
            (7, 57),
            (7, 58),
            (7, 59),
            (7, 60),
            (7, 61),
            (7, 62),
            (7, 63),
            (8, 55),
            (8, 56),
            (8, 57),
            (8, 58),
            (8, 59),
            (8, 60),
            (8, 61),
            (8, 62),
            (8, 63),
            (9, 58),
            (9, 59),
            (9, 60),
            (9, 61),
            (9, 62),
            (9, 63),
            (10, 61),
            (10, 62),
            (10, 63),
        }:
            return 10
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, token):
        if position in {0, 31}:
            return token == ""
        elif position in {1, 16, 20, 23, 25, 26, 29, 30}:
            return token == "2"
        elif position in {2, 3, 4, 5, 6}:
            return token == "3"
        elif position in {
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            21,
            22,
            24,
            27,
            28,
        }:
            return token == "<s>"

    attn_2_0_pattern = select_closest(tokens, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 27}:
            return token == ""
        elif position in {1, 2, 11, 13, 14}:
            return token == "4"
        elif position in {3, 4, 5, 6}:
            return token == "<s>"
        elif position in {7, 9, 15, 19, 24, 25}:
            return token == "0"
        elif position in {8, 10, 18, 28}:
            return token == "1"
        elif position in {12, 21, 22, 23}:
            return token == "2"
        elif position in {16, 26, 29, 31}:
            return token == "3"
        elif position in {17, 20, 30}:
            return token == "5"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, num_mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, position):
        if token in {"0"}:
            return position == 0
        elif token in {"1"}:
            return position == 20
        elif token in {"2"}:
            return position == 3
        elif token in {"3", "5"}:
            return position == 5
        elif token in {"4"}:
            return position == 23
        elif token in {"<s>"}:
            return position == 16

    attn_2_2_pattern = select_closest(positions, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_1_output, num_mlp_0_0_output):
        if attn_0_1_output in {"0"}:
            return num_mlp_0_0_output == 22
        elif attn_0_1_output in {"1"}:
            return num_mlp_0_0_output == 12
        elif attn_0_1_output in {"5", "2"}:
            return num_mlp_0_0_output == 14
        elif attn_0_1_output in {"4", "3"}:
            return num_mlp_0_0_output == 7
        elif attn_0_1_output in {"<s>"}:
            return num_mlp_0_0_output == 21

    attn_2_3_pattern = select_closest(
        num_mlp_0_0_outputs, attn_0_1_outputs, predicate_2_3
    )
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_0_output, attn_0_1_output):
        if attn_0_0_output in {"3", "1", "<s>", "4", "0", "2"}:
            return attn_0_1_output == ""
        elif attn_0_0_output in {"5"}:
            return attn_0_1_output == "5"

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_1_0_output, attn_0_0_output):
        if mlp_1_0_output in {
            0,
            5,
            8,
            10,
            13,
            15,
            16,
            17,
            18,
            20,
            23,
            24,
            27,
            28,
            29,
            31,
        }:
            return attn_0_0_output == "4"
        elif mlp_1_0_output in {1}:
            return attn_0_0_output == "<pad>"
        elif mlp_1_0_output in {2, 3, 4, 7, 9, 11, 12, 14, 19, 26, 30}:
            return attn_0_0_output == ""
        elif mlp_1_0_output in {6}:
            return attn_0_0_output == "0"
        elif mlp_1_0_output in {25, 21, 22}:
            return attn_0_0_output == "<s>"

    num_attn_2_1_pattern = select(attn_0_0_outputs, mlp_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_1_0_output, attn_1_2_output):
        if mlp_1_0_output in {0, 1, 3, 7, 9, 12, 15, 16, 19, 21, 23, 24, 25, 30}:
            return attn_1_2_output == ""
        elif mlp_1_0_output in {2, 4, 5, 6, 8, 10, 13, 14, 17, 18, 20, 22, 27, 28, 29}:
            return attn_1_2_output == "1"
        elif mlp_1_0_output in {11}:
            return attn_1_2_output == "<s>"
        elif mlp_1_0_output in {26, 31}:
            return attn_1_2_output == "2"

    num_attn_2_2_pattern = select(attn_1_2_outputs, mlp_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, token):
        if position in {
            0,
            2,
            3,
            4,
            7,
            9,
            10,
            11,
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
            28,
            29,
            30,
            31,
        }:
            return token == ""
        elif position in {1, 26, 12}:
            return token == "5"
        elif position in {8, 27, 5, 6}:
            return token == "<s>"
        elif position in {15}:
            return token == "<pad>"

    num_attn_2_3_pattern = select(tokens, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_2_output, attn_0_3_output):
        key = (attn_1_2_output, attn_0_3_output)
        return 28

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_0_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_0_output, attn_0_1_output):
        key = (attn_2_0_output, attn_0_1_output)
        if key in {
            ("0", "3"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("2", "3"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "5"),
            ("3", "<s>"),
            ("4", "3"),
            ("5", "3"),
            ("<s>", "3"),
        }:
            return 23
        return 9

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_0_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0}:
            return 1
        return 13

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_1_3_output):
        key = (num_attn_2_3_output, num_attn_1_3_output)
        return 19

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_3_outputs)
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


print(run(["<s>", "1", "1", "2", "0", "1", "0", "2", "3", "2", "0", "0"]))
