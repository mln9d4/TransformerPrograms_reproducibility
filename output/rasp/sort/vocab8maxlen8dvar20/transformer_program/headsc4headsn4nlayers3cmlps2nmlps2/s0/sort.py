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
        "output/rasp/sort/vocab8maxlen8dvar20/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/sort_weights.csv",
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
        if position in {0, 2}:
            return token == "2"
        elif position in {1, 19}:
            return token == "1"
        elif position in {3, 14}:
            return token == "3"
        elif position in {4, 5, 6, 9, 13, 15, 16}:
            return token == "4"
        elif position in {18, 7}:
            return token == "0"
        elif position in {8, 10, 11, 12, 17}:
            return token == ""

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 2}:
            return token == "1"
        elif position in {3, 4, 5, 10, 15, 17, 18}:
            return token == "3"
        elif position in {6}:
            return token == "4"
        elif position in {7, 11, 14, 16, 19}:
            return token == "0"
        elif position in {8, 9, 12, 13}:
            return token == ""

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 4, 6, 10, 12, 13, 15}:
            return token == "4"
        elif position in {8, 1}:
            return token == "0"
        elif position in {17, 2}:
            return token == "1"
        elif position in {3, 7}:
            return token == "2"
        elif position in {5, 9, 11, 14, 16, 18, 19}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 3, 5}:
            return token == "4"
        elif position in {1, 9, 10, 12, 13, 18}:
            return token == "0"
        elif position in {2, 6}:
            return token == "3"
        elif position in {4, 7, 11, 15, 16}:
            return token == "2"
        elif position in {8, 19, 14}:
            return token == "1"
        elif position in {17}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0}:
            return token == "</s>"
        elif position in {1, 2, 3, 6, 7, 10, 12}:
            return token == "1"
        elif position in {4, 9, 11, 14, 16, 19}:
            return token == "3"
        elif position in {5}:
            return token == "<s>"
        elif position in {8, 18, 13}:
            return token == "<pad>"
        elif position in {17, 15}:
            return token == ""

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 4, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19}:
            return token == "2"
        elif position in {3, 5, 6, 7}:
            return token == "1"
        elif position in {13}:
            return token == ""

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 4, 13}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2, 3}:
            return k_position == 5
        elif q_position in {5, 7, 9, 10, 14, 15, 17}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {8, 18, 19}:
            return k_position == 2
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12}:
            return k_position == 3
        elif q_position in {16}:
            return k_position == 15

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 9, 4, 15}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 3, 7}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {10, 6}:
            return k_position == 11
        elif q_position in {8, 11, 12, 13, 14, 16, 17, 18, 19}:
            return k_position == 1

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {4, 5, 6, 7, 10, 11, 13}:
            return 10
        elif key in {1, 9}:
            return 19
        return 12

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {3, 4, 5, 6, 7}:
            return 4
        elif key in {0, 11}:
            return 6
        return 16

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 5

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 14

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, mlp_0_1_output):
        if position in {0, 9}:
            return mlp_0_1_output == 17
        elif position in {1}:
            return mlp_0_1_output == 0
        elif position in {17, 2}:
            return mlp_0_1_output == 11
        elif position in {3, 4, 14}:
            return mlp_0_1_output == 16
        elif position in {16, 5}:
            return mlp_0_1_output == 4
        elif position in {10, 18, 6}:
            return mlp_0_1_output == 5
        elif position in {11, 7}:
            return mlp_0_1_output == 1
        elif position in {8, 13}:
            return mlp_0_1_output == 2
        elif position in {12}:
            return mlp_0_1_output == 6
        elif position in {15}:
            return mlp_0_1_output == 19
        elif position in {19}:
            return mlp_0_1_output == 15

    attn_1_0_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 3
        elif q_position in {1, 2, 7, 8, 11, 16, 17}:
            return k_position == 1
        elif q_position in {3, 5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {9}:
            return k_position == 18
        elif q_position in {10, 13}:
            return k_position == 9
        elif q_position in {12, 14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 8
        elif q_position in {19}:
            return k_position == 14

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {0, 8, 9, 13, 14, 16}:
            return token == ""
        elif position in {1, 2, 11}:
            return token == "0"
        elif position in {3, 12, 7}:
            return token == "2"
        elif position in {4, 5, 6}:
            return token == "3"
        elif position in {10}:
            return token == "4"
        elif position in {17, 18, 19, 15}:
            return token == "1"

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "1"
        elif attn_0_2_output in {"1"}:
            return token == "0"
        elif attn_0_2_output in {"2"}:
            return token == ""
        elif attn_0_2_output in {"3"}:
            return token == "2"
        elif attn_0_2_output in {"4"}:
            return token == "3"
        elif attn_0_2_output in {"<s>", "</s>"}:
            return token == "4"

    attn_1_3_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 2, 3, 4, 8, 10, 13, 14, 17, 18, 19}:
            return token == "1"
        elif position in {1, 9, 11, 15, 16}:
            return token == "2"
        elif position in {12, 5, 7}:
            return token == "0"
        elif position in {6}:
            return token == ""

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {0, 3, 5, 12, 14, 18}:
            return token == "2"
        elif position in {1, 2, 7, 8, 11, 15, 16, 17}:
            return token == "3"
        elif position in {10, 4}:
            return token == "1"
        elif position in {9, 19, 13, 6}:
            return token == ""

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 6, 11, 15, 17}:
            return token == "2"
        elif position in {1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 18, 19}:
            return token == "1"
        elif position in {16}:
            return token == "4"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 1, 7}:
            return token == "1"
        elif position in {2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "0"
        elif position in {5, 6}:
            return token == ""

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("2", "0"),
            ("3", "0"),
            ("4", "0"),
            ("</s>", "0"),
            ("<s>", "0"),
        }:
            return 14
        return 12

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position):
        key = position
        if key in {9, 12, 15, 16, 17, 19}:
            return 1
        elif key in {0, 4, 5, 6, 7, 18}:
            return 6
        elif key in {3, 13}:
            return 9
        return 0

    mlp_1_1_outputs = [mlp_1_1(k0) for k0 in positions]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_0_3_output):
        key = (num_attn_1_3_output, num_attn_0_3_output)
        if key in {(0, 0)}:
            return 9
        return 3

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 13
        return 16

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, mlp_0_0_output):
        if position in {0, 12}:
            return mlp_0_0_output == 17
        elif position in {1, 15}:
            return mlp_0_0_output == 2
        elif position in {2}:
            return mlp_0_0_output == 10
        elif position in {3, 13, 7}:
            return mlp_0_0_output == 18
        elif position in {10, 18, 4}:
            return mlp_0_0_output == 9
        elif position in {5, 6, 14}:
            return mlp_0_0_output == 19
        elif position in {8}:
            return mlp_0_0_output == 3
        elif position in {9}:
            return mlp_0_0_output == 11
        elif position in {11}:
            return mlp_0_0_output == 16
        elif position in {16}:
            return mlp_0_0_output == 1
        elif position in {17}:
            return mlp_0_0_output == 12
        elif position in {19}:
            return mlp_0_0_output == 13

    attn_2_0_pattern = select_closest(mlp_0_0_outputs, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, mlp_0_1_output):
        if position in {0, 16}:
            return mlp_0_1_output == 1
        elif position in {1, 18, 14, 17}:
            return mlp_0_1_output == 2
        elif position in {2, 10}:
            return mlp_0_1_output == 5
        elif position in {19, 3, 7}:
            return mlp_0_1_output == 16
        elif position in {4}:
            return mlp_0_1_output == 3
        elif position in {5}:
            return mlp_0_1_output == 14
        elif position in {9, 13, 6}:
            return mlp_0_1_output == 9
        elif position in {8, 15}:
            return mlp_0_1_output == 17
        elif position in {11}:
            return mlp_0_1_output == 18
        elif position in {12}:
            return mlp_0_1_output == 6

    attn_2_1_pattern = select_closest(mlp_0_1_outputs, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0", "2", "<s>"}:
            return k_token == ""
        elif q_token in {"</s>", "1"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "</s>"
        elif q_token in {"4"}:
            return k_token == "<s>"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_0_output, position):
        if mlp_0_0_output in {0}:
            return position == 9
        elif mlp_0_0_output in {1, 2, 15}:
            return position == 16
        elif mlp_0_0_output in {3}:
            return position == 1
        elif mlp_0_0_output in {4, 7, 9, 10, 14, 16}:
            return position == 2
        elif mlp_0_0_output in {5, 6}:
            return position == 5
        elif mlp_0_0_output in {8}:
            return position == 11
        elif mlp_0_0_output in {11}:
            return position == 0
        elif mlp_0_0_output in {12}:
            return position == 6
        elif mlp_0_0_output in {13}:
            return position == 8
        elif mlp_0_0_output in {17}:
            return position == 17
        elif mlp_0_0_output in {18}:
            return position == 12
        elif mlp_0_0_output in {19}:
            return position == 7

    attn_2_3_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, tokens)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_1_output, token):
        if mlp_0_1_output in {
            0,
            2,
            3,
            4,
            5,
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
        }:
            return token == "0"
        elif mlp_0_1_output in {1}:
            return token == "1"

    num_attn_2_0_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {0, 1, 2, 3, 7, 9, 11, 12, 13, 15, 16, 17, 18, 19}:
            return token == "0"
        elif position in {4, 6, 8, 10, 14}:
            return token == ""
        elif position in {5}:
            return token == "<pad>"

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_1_0_output, token):
        if mlp_1_0_output in {0, 1, 3, 4, 5, 7, 10, 11, 13, 15, 16, 17, 18}:
            return token == "1"
        elif mlp_1_0_output in {2, 6, 12, 14, 19}:
            return token == "0"
        elif mlp_1_0_output in {8}:
            return token == "<s>"
        elif mlp_1_0_output in {9}:
            return token == "</s>"

    num_attn_2_2_pattern = select(tokens, mlp_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, token):
        if position in {0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "1"
        elif position in {5, 6}:
            return token == ""
        elif position in {8, 7}:
            return token == "0"

    num_attn_2_3_pattern = select(tokens, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position):
        key = position
        if key in {1}:
            return 12
        return 19

    mlp_2_0_outputs = [mlp_2_0(k0) for k0 in positions]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position):
        key = position
        if key in {4, 5, 6, 7, 10, 14, 18}:
            return 8
        elif key in {1}:
            return 9
        elif key in {3}:
            return 15
        return 13

    mlp_2_1_outputs = [mlp_2_1(k0) for k0 in positions]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_1_3_output):
        key = (num_attn_2_1_output, num_attn_1_3_output)
        if key in {
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
            (4, 9),
            (4, 10),
            (4, 11),
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
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 23),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (11, 23),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (12, 20),
            (12, 21),
            (12, 22),
            (12, 23),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (14, 17),
            (14, 18),
            (14, 19),
            (14, 20),
            (14, 21),
            (14, 22),
            (14, 23),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (15, 23),
            (16, 18),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (17, 19),
            (17, 20),
            (17, 21),
            (17, 22),
            (17, 23),
            (18, 20),
            (18, 21),
            (18, 22),
            (18, 23),
            (19, 20),
            (19, 21),
            (19, 22),
            (19, 23),
            (20, 21),
            (20, 22),
            (20, 23),
            (21, 22),
            (21, 23),
            (22, 23),
            (23, 23),
        }:
            return 15
        return 2

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output):
        key = num_attn_2_1_output
        return 5

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


print(run(["<s>", "4", "2", "0", "0", "1", "2", "</s>"]))
