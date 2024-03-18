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
        "output/rasp/sort/vocab8maxlen32/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/sort_weights.csv",
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
        if position in {0, 1, 5, 6, 7, 13, 17}:
            return token == "4"
        elif position in {2, 20, 24, 27, 30}:
            return token == "0"
        elif position in {8, 3, 4}:
            return token == "2"
        elif position in {9, 10, 11, 12, 15, 16, 18, 29, 31}:
            return token == "3"
        elif position in {19, 21, 14, 22}:
            return token == "1"
        elif position in {25, 26, 23}:
            return token == "</s>"
        elif position in {28}:
            return token == "<s>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18, 30}:
            return k_position == 19
        elif q_position in {19, 31}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 25
        elif q_position in {24}:
            return k_position == 26
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {26}:
            return k_position == 28
        elif q_position in {27}:
            return k_position == 30
        elif q_position in {28}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 31

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 27}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {8, 6}:
            return k_position == 10
        elif q_position in {28, 7}:
            return k_position == 9
        elif q_position in {9, 12}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11, 29}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 16
        elif q_position in {15}:
            return k_position == 20
        elif q_position in {16}:
            return k_position == 22
        elif q_position in {17}:
            return k_position == 23
        elif q_position in {18}:
            return k_position == 24
        elif q_position in {19}:
            return k_position == 5
        elif q_position in {20, 21}:
            return k_position == 26
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {24, 23}:
            return k_position == 29
        elif q_position in {25, 26}:
            return k_position == 30
        elif q_position in {30}:
            return k_position == 4
        elif q_position in {31}:
            return k_position == 18

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 30, 6, 15}:
            return token == "3"
        elif position in {1, 2, 4, 7, 8, 17, 28, 31}:
            return token == "2"
        elif position in {24, 27, 3}:
            return token == "0"
        elif position in {5, 9, 10, 11, 12, 13, 14, 16, 18, 20}:
            return token == "4"
        elif position in {19}:
            return token == ""
        elif position in {21}:
            return token == "1"
        elif position in {26, 22, 23}:
            return token == "<s>"
        elif position in {25, 29}:
            return token == "</s>"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 1, 23, 26, 28, 30}:
            return token == "0"
        elif position in {2, 3, 21, 22, 25, 27, 31}:
            return token == "1"
        elif position in {4, 5, 7}:
            return token == ""
        elif position in {6}:
            return token == "<pad>"
        elif position in {8, 9, 11, 14, 15, 16, 18, 19, 20}:
            return token == "</s>"
        elif position in {17, 10, 12, 13}:
            return token == "<s>"
        elif position in {24, 29}:
            return token == "2"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 4, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25}:
            return token == "0"
        elif position in {1, 2, 3, 24, 26, 28, 29, 30}:
            return token == ""
        elif position in {27, 5, 31}:
            return token == "<s>"
        elif position in {10, 12, 15}:
            return token == "1"
        elif position in {21}:
            return token == "</s>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {8, 7}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 15
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12, 30}:
            return k_position == 17
        elif q_position in {13}:
            return k_position == 18
        elif q_position in {14, 15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 21
        elif q_position in {17}:
            return k_position == 22
        elif q_position in {18}:
            return k_position == 23
        elif q_position in {19}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {21}:
            return k_position == 26
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {24, 23}:
            return k_position == 29
        elif q_position in {25, 26, 27}:
            return k_position == 30
        elif q_position in {28}:
            return k_position == 9
        elif q_position in {29}:
            return k_position == 31
        elif q_position in {31}:
            return k_position == 2

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 28}:
            return token == "1"
        elif position in {1, 2, 30, 31}:
            return token == "0"
        elif position in {3, 4, 5, 6}:
            return token == ""
        elif position in {7, 10, 11, 13, 14, 15, 19, 21}:
            return token == "<s>"
        elif position in {8, 9, 12, 16, 17, 18, 20}:
            return token == "</s>"
        elif position in {22, 23, 24, 25, 26, 27}:
            return token == "4"
        elif position in {29}:
            return token == "2"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {1, 2, 3, 4, 5, 6, 31}:
            return 14
        elif key in {9, 10}:
            return 31
        elif key in {7}:
            return 9
        elif key in {8}:
            return 13
        return 7

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {0, 5, 7}:
            return 6
        elif key in {3, 4}:
            return 14
        elif key in {1, 2}:
            return 28
        return 26

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 16

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 16

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_1_output, attn_0_0_output):
        if mlp_0_1_output in {0, 17, 18, 22, 29}:
            return attn_0_0_output == "3"
        elif mlp_0_1_output in {8, 1, 28, 9}:
            return attn_0_0_output == "0"
        elif mlp_0_1_output in {2, 5, 6, 10, 12, 13, 15, 19, 20, 26}:
            return attn_0_0_output == "2"
        elif mlp_0_1_output in {11, 3, 31, 23}:
            return attn_0_0_output == ""
        elif mlp_0_1_output in {24, 4, 30}:
            return attn_0_0_output == "</s>"
        elif mlp_0_1_output in {14, 7}:
            return attn_0_0_output == "<s>"
        elif mlp_0_1_output in {16, 25, 27, 21}:
            return attn_0_0_output == "4"

    attn_1_0_pattern = select_closest(attn_0_0_outputs, mlp_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 17, 21, 25, 26}:
            return token == "4"
        elif position in {1, 6}:
            return token == "0"
        elif position in {2, 29, 30}:
            return token == "</s>"
        elif position in {8, 16, 3}:
            return token == "1"
        elif position in {4, 5, 7, 9, 12, 13, 14, 15, 18, 19, 20, 23}:
            return token == ""
        elif position in {10, 22, 27, 28, 31}:
            return token == "3"
        elif position in {11}:
            return token == "2"
        elif position in {24}:
            return token == "<s>"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 11
        elif q_position in {1, 7}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3, 5}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 22
        elif q_position in {8}:
            return k_position == 4
        elif q_position in {9, 28}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 16
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 2
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18, 26}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 9
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 8
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {27}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 30
        elif q_position in {30}:
            return k_position == 31
        elif q_position in {31}:
            return k_position == 28

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_0_output, attn_0_3_output):
        if mlp_0_0_output in {0, 26, 27, 15}:
            return attn_0_3_output == "4"
        elif mlp_0_0_output in {
            1,
            2,
            3,
            4,
            5,
            11,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            25,
            28,
            29,
            30,
        }:
            return attn_0_3_output == ""
        elif mlp_0_0_output in {24, 12, 6}:
            return attn_0_3_output == "</s>"
        elif mlp_0_0_output in {31, 13, 7}:
            return attn_0_3_output == "2"
        elif mlp_0_0_output in {8, 9, 14}:
            return attn_0_3_output == "0"
        elif mlp_0_0_output in {10}:
            return attn_0_3_output == "<pad>"

    attn_1_3_pattern = select_closest(attn_0_3_outputs, mlp_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30}:
            return token == ""
        elif position in {1, 10, 12, 6}:
            return token == "<s>"
        elif position in {2, 4, 5, 7, 8, 9, 11, 31}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {13}:
            return token == "</s>"
        elif position in {28, 14}:
            return token == "<pad>"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 1, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15}:
            return position == 2
        elif mlp_0_0_output in {2}:
            return position == 7
        elif mlp_0_0_output in {5}:
            return position == 12
        elif mlp_0_0_output in {26, 7}:
            return position == 20
        elif mlp_0_0_output in {16}:
            return position == 14
        elif mlp_0_0_output in {17}:
            return position == 26
        elif mlp_0_0_output in {25, 18}:
            return position == 15
        elif mlp_0_0_output in {19}:
            return position == 9
        elif mlp_0_0_output in {20, 29}:
            return position == 8
        elif mlp_0_0_output in {24, 21}:
            return position == 17
        elif mlp_0_0_output in {22, 30}:
            return position == 13
        elif mlp_0_0_output in {27, 23}:
            return position == 16
        elif mlp_0_0_output in {28}:
            return position == 6
        elif mlp_0_0_output in {31}:
            return position == 18

    num_attn_1_1_pattern = select(positions, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_position, k_position):
        if q_position in {0, 2, 3, 4, 8, 12, 30}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {5, 10, 15, 21, 26}:
            return k_position == 0
        elif q_position in {6, 9, 11, 14, 16, 18, 28, 31}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 16
        elif q_position in {25, 13}:
            return k_position == 22
        elif q_position in {17, 27, 20}:
            return k_position == 21
        elif q_position in {19, 29, 22, 23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 28

    num_attn_1_2_pattern = select(positions, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_1_output):
        if position in {0}:
            return attn_0_1_output == "<s>"
        elif position in {1, 2, 5, 6}:
            return attn_0_1_output == "2"
        elif position in {3, 14, 15}:
            return attn_0_1_output == "</s>"
        elif position in {4, 7, 8, 9, 10, 11, 12, 13, 31}:
            return attn_0_1_output == "1"
        elif position in {16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 30}:
            return attn_0_1_output == ""
        elif position in {26, 23}:
            return attn_0_1_output == "<pad>"

    num_attn_1_3_pattern = select(attn_0_1_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_3_output):
        key = (position, attn_1_3_output)
        if key in {
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (6, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "</s>"),
            (7, "<s>"),
            (14, "0"),
            (14, "1"),
            (14, "2"),
            (14, "3"),
            (14, "4"),
            (14, "</s>"),
            (14, "<s>"),
            (27, "0"),
            (27, "1"),
            (27, "2"),
            (27, "3"),
            (27, "4"),
            (27, "</s>"),
            (27, "<s>"),
        }:
            return 28
        elif key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "</s>"),
            (3, "<s>"),
            (25, "1"),
            (25, "3"),
            (25, "</s>"),
            (25, "<s>"),
            (29, "0"),
            (29, "1"),
            (29, "2"),
            (29, "3"),
            (29, "4"),
            (29, "</s>"),
            (29, "<s>"),
            (30, "<s>"),
        }:
            return 5
        elif key in {
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "</s>"),
            (5, "<s>"),
            (31, "<s>"),
        }:
            return 31
        return 11

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_3_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position):
        key = position
        if key in {6, 7, 8, 9, 10, 11, 12, 13, 14, 29, 30}:
            return 14
        elif key in {28}:
            return 9
        elif key in {1}:
            return 31
        return 8

    mlp_1_1_outputs = [mlp_1_1(k0) for k0 in positions]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_0_0_output):
        key = (num_attn_1_0_output, num_attn_0_0_output)
        return 0

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_0_2_output):
        key = (num_attn_1_1_output, num_attn_0_2_output)
        return 9

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 27}:
            return position == 3
        elif mlp_0_0_output in {16, 1}:
            return position == 12
        elif mlp_0_0_output in {2}:
            return position == 17
        elif mlp_0_0_output in {3}:
            return position == 16
        elif mlp_0_0_output in {17, 4}:
            return position == 21
        elif mlp_0_0_output in {5}:
            return position == 19
        elif mlp_0_0_output in {29, 6, 15}:
            return position == 6
        elif mlp_0_0_output in {9, 31, 13, 7}:
            return position == 4
        elif mlp_0_0_output in {8}:
            return position == 27
        elif mlp_0_0_output in {10, 23}:
            return position == 11
        elif mlp_0_0_output in {11}:
            return position == 15
        elif mlp_0_0_output in {12}:
            return position == 2
        elif mlp_0_0_output in {18, 14}:
            return position == 1
        elif mlp_0_0_output in {19}:
            return position == 13
        elif mlp_0_0_output in {20, 28}:
            return position == 10
        elif mlp_0_0_output in {21}:
            return position == 23
        elif mlp_0_0_output in {25, 22}:
            return position == 0
        elif mlp_0_0_output in {24}:
            return position == 22
        elif mlp_0_0_output in {26}:
            return position == 28
        elif mlp_0_0_output in {30}:
            return position == 7

    attn_2_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0, 26, 14}:
            return k_mlp_0_1_output == 6
        elif q_mlp_0_1_output in {1}:
            return k_mlp_0_1_output == 3
        elif q_mlp_0_1_output in {17, 2, 20}:
            return k_mlp_0_1_output == 29
        elif q_mlp_0_1_output in {8, 3}:
            return k_mlp_0_1_output == 22
        elif q_mlp_0_1_output in {4}:
            return k_mlp_0_1_output == 11
        elif q_mlp_0_1_output in {5}:
            return k_mlp_0_1_output == 25
        elif q_mlp_0_1_output in {18, 6, 22}:
            return k_mlp_0_1_output == 7
        elif q_mlp_0_1_output in {7, 19, 23, 24, 25, 27}:
            return k_mlp_0_1_output == 26
        elif q_mlp_0_1_output in {9}:
            return k_mlp_0_1_output == 18
        elif q_mlp_0_1_output in {10}:
            return k_mlp_0_1_output == 10
        elif q_mlp_0_1_output in {11, 12, 30}:
            return k_mlp_0_1_output == 24
        elif q_mlp_0_1_output in {21, 13}:
            return k_mlp_0_1_output == 0
        elif q_mlp_0_1_output in {15}:
            return k_mlp_0_1_output == 9
        elif q_mlp_0_1_output in {16}:
            return k_mlp_0_1_output == 30
        elif q_mlp_0_1_output in {28}:
            return k_mlp_0_1_output == 1
        elif q_mlp_0_1_output in {29}:
            return k_mlp_0_1_output == 5
        elif q_mlp_0_1_output in {31}:
            return k_mlp_0_1_output == 31

    attn_2_1_pattern = select_closest(mlp_0_1_outputs, mlp_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_1_output, attn_1_1_output):
        if mlp_0_1_output in {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 16, 19, 24, 27, 28, 30}:
            return attn_1_1_output == ""
        elif mlp_0_1_output in {6, 7, 12, 20, 25, 26, 31}:
            return attn_1_1_output == "<s>"
        elif mlp_0_1_output in {13, 14}:
            return attn_1_1_output == "0"
        elif mlp_0_1_output in {23, 15}:
            return attn_1_1_output == "</s>"
        elif mlp_0_1_output in {17, 18, 29}:
            return attn_1_1_output == "4"
        elif mlp_0_1_output in {21}:
            return attn_1_1_output == "1"
        elif mlp_0_1_output in {22}:
            return attn_1_1_output == "<pad>"

    attn_2_2_pattern = select_closest(attn_1_1_outputs, mlp_0_1_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0}:
            return k_mlp_0_1_output == 3
        elif q_mlp_0_1_output in {1, 4, 9, 11, 14, 15, 21}:
            return k_mlp_0_1_output == 26
        elif q_mlp_0_1_output in {2, 23}:
            return k_mlp_0_1_output == 18
        elif q_mlp_0_1_output in {3}:
            return k_mlp_0_1_output == 22
        elif q_mlp_0_1_output in {5}:
            return k_mlp_0_1_output == 31
        elif q_mlp_0_1_output in {6}:
            return k_mlp_0_1_output == 13
        elif q_mlp_0_1_output in {7, 20, 27, 28, 29}:
            return k_mlp_0_1_output == 6
        elif q_mlp_0_1_output in {8, 16, 17, 31}:
            return k_mlp_0_1_output == 14
        elif q_mlp_0_1_output in {10, 19}:
            return k_mlp_0_1_output == 7
        elif q_mlp_0_1_output in {12}:
            return k_mlp_0_1_output == 16
        elif q_mlp_0_1_output in {13}:
            return k_mlp_0_1_output == 28
        elif q_mlp_0_1_output in {18}:
            return k_mlp_0_1_output == 8
        elif q_mlp_0_1_output in {24, 22}:
            return k_mlp_0_1_output == 19
        elif q_mlp_0_1_output in {25, 30}:
            return k_mlp_0_1_output == 20
        elif q_mlp_0_1_output in {26}:
            return k_mlp_0_1_output == 11

    attn_2_3_pattern = select_closest(mlp_0_1_outputs, mlp_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, attn_1_2_output):
        if position in {0, 7, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return attn_1_2_output == ""
        elif position in {1}:
            return attn_1_2_output == "1"
        elif position in {2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 31}:
            return attn_1_2_output == "0"
        elif position in {17, 13, 14, 15}:
            return attn_1_2_output == "<s>"
        elif position in {16}:
            return attn_1_2_output == "</s>"
        elif position in {30}:
            return attn_1_2_output == "<pad>"

    num_attn_2_0_pattern = select(attn_1_2_outputs, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_1_1_output, token):
        if num_mlp_1_1_output in {
            0,
            1,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
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
            30,
            31,
        }:
            return token == "2"
        elif num_mlp_1_1_output in {2, 14, 15}:
            return token == "0"
        elif num_mlp_1_1_output in {3}:
            return token == "1"
        elif num_mlp_1_1_output in {29, 5}:
            return token == "3"
        elif num_mlp_1_1_output in {28}:
            return token == "</s>"

    num_attn_2_1_pattern = select(tokens, num_mlp_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_1_2_output):
        if position in {0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 31}:
            return attn_1_2_output == "2"
        elif position in {2, 3}:
            return attn_1_2_output == "<s>"
        elif position in {17, 18, 19, 21, 22}:
            return attn_1_2_output == "</s>"
        elif position in {20, 23, 24, 25, 26, 27, 28, 29, 30}:
            return attn_1_2_output == ""

    num_attn_2_2_pattern = select(attn_1_2_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 14, 28}:
            return token == "0"
        elif mlp_0_0_output in {26, 29, 7}:
            return token == ""
        elif mlp_0_0_output in {10, 12, 13, 15, 17, 18, 23, 24, 31}:
            return token == "<s>"
        elif mlp_0_0_output in {16, 19, 20, 21}:
            return token == "</s>"
        elif mlp_0_0_output in {25, 22}:
            return token == "3"
        elif mlp_0_0_output in {27, 30}:
            return token == "4"

    num_attn_2_3_pattern = select(tokens, mlp_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, position):
        key = (attn_2_2_output, position)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 6),
            ("0", 7),
            ("0", 10),
            ("0", 12),
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("0", 16),
            ("0", 20),
            ("0", 21),
            ("0", 24),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("0", 28),
            ("0", 29),
            ("0", 30),
            ("0", 31),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 13),
            ("1", 14),
            ("1", 15),
            ("1", 16),
            ("1", 21),
            ("1", 27),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 13),
            ("2", 14),
            ("2", 15),
            ("2", 16),
            ("2", 21),
            ("2", 27),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 13),
            ("3", 14),
            ("3", 15),
            ("3", 16),
            ("3", 21),
            ("3", 27),
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 13),
            ("4", 14),
            ("4", 15),
            ("4", 16),
            ("4", 21),
            ("4", 27),
            ("4", 30),
            ("4", 31),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 13),
            ("</s>", 14),
            ("</s>", 16),
            ("</s>", 21),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
            ("<s>", 16),
            ("<s>", 21),
            ("<s>", 27),
        }:
            return 0
        elif key in {("</s>", 27)}:
            return 21
        return 6

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_0_1_output, num_mlp_0_0_output):
        key = (mlp_0_1_output, num_mlp_0_0_output)
        return 4

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, num_mlp_0_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 30

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 1

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_1_2_outputs]
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


print(
    run(
        ["<s>", "0", "4", "4", "2", "4", "3", "2", "3", "2", "4", "2", "2", "4", "</s>"]
    )
)
