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
        "output/rasp/sort/vocab8maxlen16/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/sort_weights.csv",
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
        if position in {0, 2, 7}:
            return token == "0"
        elif position in {1, 11, 15}:
            return token == "<s>"
        elif position in {3}:
            return token == "4"
        elif position in {9, 4, 12}:
            return token == "2"
        elif position in {10, 5, 6}:
            return token == "3"
        elif position in {8, 13, 14}:
            return token == "</s>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {8, 2, 13, 5}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4, 14, 15}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 6}:
            return token == "4"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 4, 5, 7, 8, 11, 12, 13, 14}:
            return token == "3"
        elif position in {9, 10}:
            return token == "1"
        elif position in {15}:
            return token == "2"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 14}:
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
        elif q_position in {13, 15}:
            return k_position == 14

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 8, 15}:
            return token == "0"
        elif position in {1, 2, 3, 4}:
            return token == "2"
        elif position in {5, 6, 7}:
            return token == "1"
        elif position in {9, 10, 11, 13, 14}:
            return token == ""
        elif position in {12}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 14}:
            return token == "1"
        elif position in {1, 2, 3, 10, 15}:
            return token == "0"
        elif position in {4, 5}:
            return token == ""
        elif position in {8, 9, 6, 7}:
            return token == "3"
        elif position in {11, 12, 13}:
            return token == "2"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 2, 3, 4}:
            return token == "0"
        elif position in {5, 6, 7, 8, 9, 10, 12, 13, 14, 15}:
            return token == ""
        elif position in {11}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 9, 10, 11}:
            return token == "1"
        elif position in {1, 12, 13}:
            return token == "0"
        elif position in {2, 3, 15}:
            return token == ""
        elif position in {4}:
            return token == "</s>"
        elif position in {8, 5, 6, 7}:
            return token == "2"
        elif position in {14}:
            return token == "<s>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {2, 3, 4, 5}:
            return 3
        elif key in {1, 15}:
            return 1
        elif key in {0, 7}:
            return 6
        elif key in {6}:
            return 7
        return 12

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {1, 2, 4, 15}:
            return 14
        elif key in {3, 5}:
            return 3
        return 8

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0, 1}:
            return 10
        return 2

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 6
        return 3

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, attn_0_3_output):
        if position in {0, 7, 9, 10, 13}:
            return attn_0_3_output == "3"
        elif position in {1, 15}:
            return attn_0_3_output == "1"
        elif position in {2, 3, 5, 6}:
            return attn_0_3_output == "2"
        elif position in {4}:
            return attn_0_3_output == "<s>"
        elif position in {8}:
            return attn_0_3_output == "0"
        elif position in {11, 14}:
            return attn_0_3_output == ""
        elif position in {12}:
            return attn_0_3_output == "4"

    attn_1_0_pattern = select_closest(attn_0_3_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"0"}:
            return attn_0_1_output == "0"
        elif attn_0_3_output in {"4", "3", "1"}:
            return attn_0_1_output == "</s>"
        elif attn_0_3_output in {"2"}:
            return attn_0_1_output == "<s>"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_1_output == "1"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_1_output == ""

    attn_1_1_pattern = select_closest(attn_0_1_outputs, attn_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, attn_0_2_output):
        if position in {0, 10, 6}:
            return attn_0_2_output == "3"
        elif position in {1, 7}:
            return attn_0_2_output == "0"
        elif position in {9, 2, 13}:
            return attn_0_2_output == ""
        elif position in {3, 5}:
            return attn_0_2_output == "<s>"
        elif position in {4}:
            return attn_0_2_output == "</s>"
        elif position in {8, 11}:
            return attn_0_2_output == "4"
        elif position in {12, 14}:
            return attn_0_2_output == "1"
        elif position in {15}:
            return attn_0_2_output == "2"

    attn_1_2_pattern = select_closest(attn_0_2_outputs, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 13, 14}:
            return mlp_0_1_output == 5
        elif mlp_0_0_output in {1}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {9, 2, 11}:
            return mlp_0_1_output == 3
        elif mlp_0_0_output in {3, 5}:
            return mlp_0_1_output == 8
        elif mlp_0_0_output in {4, 6, 7}:
            return mlp_0_1_output == 14
        elif mlp_0_0_output in {8, 15}:
            return mlp_0_1_output == 4
        elif mlp_0_0_output in {10, 12}:
            return mlp_0_1_output == 7

    attn_1_3_pattern = select_closest(mlp_0_1_outputs, mlp_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, mlp_0_0_output):
        if attn_0_2_output in {"1", "0"}:
            return mlp_0_0_output == 1
        elif attn_0_2_output in {"2", "3", "<s>", "4"}:
            return mlp_0_0_output == 6
        elif attn_0_2_output in {"</s>"}:
            return mlp_0_0_output == 3

    num_attn_1_0_pattern = select(mlp_0_0_outputs, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_1_output, attn_0_0_output):
        if num_mlp_0_1_output in {0, 2, 3, 12, 14, 15}:
            return attn_0_0_output == "<s>"
        elif num_mlp_0_1_output in {1}:
            return attn_0_0_output == "2"
        elif num_mlp_0_1_output in {4, 7, 8, 11, 13}:
            return attn_0_0_output == "0"
        elif num_mlp_0_1_output in {9, 10, 5, 6}:
            return attn_0_0_output == "3"

    num_attn_1_1_pattern = select(
        attn_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_0_output, num_mlp_0_1_output):
        if num_mlp_0_0_output in {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15}:
            return num_mlp_0_1_output == 3
        elif num_mlp_0_0_output in {1}:
            return num_mlp_0_1_output == 7
        elif num_mlp_0_0_output in {12}:
            return num_mlp_0_1_output == 2
        elif num_mlp_0_0_output in {13, 14}:
            return num_mlp_0_1_output == 4

    num_attn_1_2_pattern = select(
        num_mlp_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 13, 3, 5}:
            return token == "1"
        elif position in {1, 2, 4, 6, 8, 9, 12}:
            return token == "<s>"
        elif position in {11, 15, 7}:
            return token == "0"
        elif position in {10, 14}:
            return token == "2"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, position):
        key = (attn_1_3_output, position)
        if key in {
            ("0", 1),
            ("0", 14),
            ("1", 1),
            ("1", 14),
            ("2", 1),
            ("2", 14),
            ("4", 1),
            ("</s>", 1),
            ("</s>", 14),
            ("<s>", 1),
            ("<s>", 14),
        }:
            return 13
        elif key in {
            ("0", 0),
            ("0", 3),
            ("0", 5),
            ("0", 7),
            ("0", 8),
            ("0", 10),
            ("0", 12),
            ("0", 15),
            ("4", 14),
        }:
            return 4
        elif key in {("3", 1)}:
            return 1
        return 15

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position):
        key = position
        if key in {1, 2, 3}:
            return 0
        return 4

    mlp_1_1_outputs = [mlp_1_1(k0) for k0 in positions]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 3
        return 14

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0, 1}:
            return 8
        return 0

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, mlp_0_0_output):
        if position in {0}:
            return mlp_0_0_output == 13
        elif position in {1, 3, 12, 5}:
            return mlp_0_0_output == 8
        elif position in {2, 11}:
            return mlp_0_0_output == 6
        elif position in {4, 7}:
            return mlp_0_0_output == 12
        elif position in {6}:
            return mlp_0_0_output == 3
        elif position in {8, 9, 10, 14}:
            return mlp_0_0_output == 7
        elif position in {13}:
            return mlp_0_0_output == 5
        elif position in {15}:
            return mlp_0_0_output == 11

    attn_2_0_pattern = select_closest(mlp_0_0_outputs, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0}:
            return k_mlp_0_0_output == 13
        elif q_mlp_0_0_output in {1, 3, 7}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {2, 10}:
            return k_mlp_0_0_output == 3
        elif q_mlp_0_0_output in {4, 12, 6}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {5}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {8, 9}:
            return k_mlp_0_0_output == 4
        elif q_mlp_0_0_output in {11}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {13}:
            return k_mlp_0_0_output == 2
        elif q_mlp_0_0_output in {14, 15}:
            return k_mlp_0_0_output == 10

    attn_2_1_pattern = select_closest(mlp_0_0_outputs, mlp_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 3}:
            return token == "</s>"
        elif mlp_0_0_output in {1, 2, 4, 9, 10, 11, 15}:
            return token == ""
        elif mlp_0_0_output in {5}:
            return token == "3"
        elif mlp_0_0_output in {12, 13, 6}:
            return token == "<s>"
        elif mlp_0_0_output in {7}:
            return token == "<pad>"
        elif mlp_0_0_output in {8}:
            return token == "4"
        elif mlp_0_0_output in {14}:
            return token == "0"

    attn_2_2_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_1_1_output, token):
        if mlp_1_1_output in {0, 15}:
            return token == ""
        elif mlp_1_1_output in {1, 13}:
            return token == "</s>"
        elif mlp_1_1_output in {2, 14}:
            return token == "0"
        elif mlp_1_1_output in {3, 5, 6, 9, 11}:
            return token == "3"
        elif mlp_1_1_output in {4}:
            return token == "4"
        elif mlp_1_1_output in {8, 12, 7}:
            return token == "<s>"
        elif mlp_1_1_output in {10}:
            return token == "<pad>"

    attn_2_3_pattern = select_closest(tokens, mlp_1_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, token):
        if position in {0, 1, 2, 3, 4, 8}:
            return token == "1"
        elif position in {15, 5, 6, 7}:
            return token == "0"
        elif position in {9, 10}:
            return token == "<s>"
        elif position in {11, 12, 13, 14}:
            return token == ""

    num_attn_2_0_pattern = select(tokens, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {0, 2, 3, 4, 5, 6, 7}:
            return token == "1"
        elif position in {1}:
            return token == "</s>"
        elif position in {8, 9, 15}:
            return token == "0"
        elif position in {10}:
            return token == "<s>"
        elif position in {11, 12, 13, 14}:
            return token == ""

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, token):
        if position in {0, 1, 2, 3, 4}:
            return token == "0"
        elif position in {5, 6, 7, 8, 9, 15}:
            return token == "2"
        elif position in {10, 11}:
            return token == "<s>"
        elif position in {12, 13, 14}:
            return token == ""

    num_attn_2_2_pattern = select(tokens, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 14, 15}:
            return token == "0"
        elif mlp_0_0_output in {6}:
            return token == "</s>"
        elif mlp_0_0_output in {8, 12}:
            return token == ""

    num_attn_2_3_pattern = select(tokens, mlp_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, position):
        key = (attn_2_3_output, position)
        if key in {
            ("0", 6),
            ("0", 7),
            ("0", 9),
            ("1", 6),
            ("1", 7),
            ("1", 9),
            ("2", 6),
            ("2", 7),
            ("2", 9),
            ("3", 6),
            ("3", 7),
            ("3", 9),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 9),
            ("4", 12),
            ("4", 14),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 9),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 14),
        }:
            return 14
        elif key in {
            ("1", 3),
            ("1", 8),
            ("1", 12),
            ("3", 3),
            ("3", 8),
            ("3", 12),
            ("4", 3),
            ("</s>", 3),
            ("</s>", 8),
            ("</s>", 12),
            ("<s>", 3),
            ("<s>", 12),
            ("<s>", 15),
        }:
            return 10
        return 1

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_3_output, position):
        key = (attn_2_3_output, position)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("2", 0),
            ("2", 1),
            ("2", 2),
            ("3", 0),
            ("3", 1),
            ("3", 2),
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("</s>", 1),
            ("</s>", 2),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 6
        return 5

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_3_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_2_0_output):
        key = (num_attn_1_2_output, num_attn_2_0_output)
        if key in {
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (14, 1),
            (15, 0),
            (15, 1),
            (16, 0),
            (16, 1),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (19, 0),
            (19, 1),
            (20, 0),
            (20, 1),
            (21, 0),
            (21, 1),
            (22, 0),
            (22, 1),
            (23, 0),
            (23, 1),
            (24, 0),
            (24, 1),
            (25, 0),
            (25, 1),
            (26, 0),
            (26, 1),
            (27, 0),
            (27, 1),
            (28, 0),
            (28, 1),
            (29, 0),
            (29, 1),
            (29, 2),
            (30, 0),
            (30, 1),
            (30, 2),
            (31, 0),
            (31, 1),
            (31, 2),
            (32, 0),
            (32, 1),
            (32, 2),
            (33, 0),
            (33, 1),
            (33, 2),
            (34, 0),
            (34, 1),
            (34, 2),
            (35, 0),
            (35, 1),
            (35, 2),
            (36, 0),
            (36, 1),
            (36, 2),
            (37, 0),
            (37, 1),
            (37, 2),
            (38, 0),
            (38, 1),
            (38, 2),
            (39, 0),
            (39, 1),
            (40, 0),
            (40, 1),
            (41, 0),
            (41, 1),
            (42, 0),
            (42, 1),
            (43, 0),
            (43, 1),
            (44, 0),
            (45, 0),
            (46, 0),
            (47, 0),
        }:
            return 14
        elif key in {
            (39, 2),
            (40, 2),
            (41, 2),
            (42, 2),
            (43, 2),
            (44, 1),
            (44, 2),
            (45, 1),
            (45, 2),
            (46, 1),
            (46, 2),
            (47, 1),
            (47, 2),
        }:
            return 8
        elif key in {(0, 0)}:
            return 10
        return 0

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        return 5

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
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


print(run(["<s>", "0", "4", "3", "3", "0", "1", "1", "4", "2", "</s>"]))
