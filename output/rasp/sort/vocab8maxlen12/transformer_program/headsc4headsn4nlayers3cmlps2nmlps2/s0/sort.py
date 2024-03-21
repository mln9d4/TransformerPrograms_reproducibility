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
        "output/rasp/sort/vocab8maxlen12/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/sort_weights.csv",
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
        if position in {0, 4, 6}:
            return token == "4"
        elif position in {1, 18}:
            return token == "0"
        elif position in {2, 7}:
            return token == "2"
        elif position in {3}:
            return token == "3"
        elif position in {9, 5, 14}:
            return token == ""
        elif position in {8, 10, 11, 12, 13, 15, 16, 17, 19}:
            return token == "1"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 7, 11, 12, 14, 17, 19}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {18, 3}:
            return k_position == 4
        elif q_position in {10, 4, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {8, 16}:
            return k_position == 13
        elif q_position in {9, 13}:
            return k_position == 8
        elif q_position in {15}:
            return k_position == 10

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 1, 19, 12}:
            return k_position == 1
        elif q_position in {8, 2}:
            return k_position == 3
        elif q_position in {3, 11, 13, 14, 15}:
            return k_position == 2
        elif q_position in {4, 6, 7}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {9, 18}:
            return k_position == 18
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 16

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 4, 6}:
            return token == "4"
        elif position in {1, 7, 9, 11, 12, 13}:
            return token == "0"
        elif position in {2, 5, 10, 16, 17}:
            return token == "2"
        elif position in {3, 8, 14, 18, 19}:
            return token == "1"
        elif position in {15}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 6
        elif q_position in {1, 8, 9, 10, 11, 14, 16, 17, 19}:
            return k_position == 4
        elif q_position in {2, 18, 5, 7}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {12, 6}:
            return k_position == 2
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 14

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "0"
        elif q_token in {"</s>", "4", "3"}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == "</s>"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 3, 4}:
            return k_position == 6
        elif q_position in {1, 12}:
            return k_position == 3
        elif q_position in {2, 7}:
            return k_position == 4
        elif q_position in {10, 5, 14, 15}:
            return k_position == 7
        elif q_position in {18, 6}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {9, 11}:
            return k_position == 16
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 2

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 17, 19, 13}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {8, 3}:
            return k_position == 6
        elif q_position in {9, 4, 15}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 18
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {10, 11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {16}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 9

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {3, 4, 5, 6, 8}:
            return 10
        elif key in {1}:
            return 16
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        return 16

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        return 18

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 7

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {0, 2, 6, 10, 11, 12, 14, 15, 18}:
            return token == ""
        elif num_mlp_0_0_output in {1, 13, 9, 17}:
            return token == "1"
        elif num_mlp_0_0_output in {16, 3, 5}:
            return token == "0"
        elif num_mlp_0_0_output in {4, 7}:
            return token == "4"
        elif num_mlp_0_0_output in {8, 19}:
            return token == "2"

    attn_1_0_pattern = select_closest(tokens, num_mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 2, 18}:
            return token == "0"
        elif position in {8, 1, 12, 9}:
            return token == "1"
        elif position in {3, 13}:
            return token == "3"
        elif position in {4}:
            return token == "<s>"
        elif position in {5}:
            return token == "</s>"
        elif position in {6, 7, 10, 11, 14, 16, 17}:
            return token == ""
        elif position in {15}:
            return token == "2"
        elif position in {19}:
            return token == "<pad>"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 10}:
            return token == "4"
        elif mlp_0_0_output in {1, 2, 12}:
            return token == ""
        elif mlp_0_0_output in {16, 3, 4, 5}:
            return token == "1"
        elif mlp_0_0_output in {6, 7, 8, 11, 13, 15, 17, 19}:
            return token == "0"
        elif mlp_0_0_output in {9, 18}:
            return token == "</s>"
        elif mlp_0_0_output in {14}:
            return token == "2"

    attn_1_2_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, attn_0_0_output):
        if attn_0_1_output in {"0"}:
            return attn_0_0_output == "</s>"
        elif attn_0_1_output in {"1"}:
            return attn_0_0_output == "0"
        elif attn_0_1_output in {"2", "<s>"}:
            return attn_0_0_output == ""
        elif attn_0_1_output in {"3"}:
            return attn_0_0_output == "1"
        elif attn_0_1_output in {"</s>", "4"}:
            return attn_0_0_output == "<s>"

    attn_1_3_pattern = select_closest(attn_0_0_outputs, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "1"
        elif attn_0_0_output in {"</s>", "1", "<s>", "2", "3"}:
            return token == "0"
        elif attn_0_0_output in {"4"}:
            return token == ""

    num_attn_1_0_pattern = select(tokens, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {0, 1, 3, 17}:
            return token == "1"
        elif position in {2}:
            return token == ""
        elif position in {10, 4, 14}:
            return token == "0"
        elif position in {11, 5, 15}:
            return token == "2"
        elif position in {6}:
            return token == "<pad>"
        elif position in {7, 8, 12, 16, 18, 19}:
            return token == "3"
        elif position in {9, 13}:
            return token == "</s>"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, num_mlp_0_0_output):
        if attn_0_3_output in {"0"}:
            return num_mlp_0_0_output == 16
        elif attn_0_3_output in {"1"}:
            return num_mlp_0_0_output == 12
        elif attn_0_3_output in {"2"}:
            return num_mlp_0_0_output == 2
        elif attn_0_3_output in {"3"}:
            return num_mlp_0_0_output == 18
        elif attn_0_3_output in {"4", "<s>"}:
            return num_mlp_0_0_output == 6
        elif attn_0_3_output in {"</s>"}:
            return num_mlp_0_0_output == 5

    num_attn_1_2_pattern = select(
        num_mlp_0_0_outputs, attn_0_3_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 1, 7, 8, 11, 14, 16, 19}:
            return token == "1"
        elif position in {2, 3, 9, 10, 12, 13, 17, 18}:
            return token == "0"
        elif position in {4, 5, 6}:
            return token == ""
        elif position in {15}:
            return token == "</s>"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_1_output, position):
        key = (attn_0_1_output, position)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
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
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 14),
            ("1", 15),
            ("1", 16),
            ("1", 17),
            ("1", 18),
            ("1", 19),
            ("3", 2),
            ("3", 11),
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 11),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 11),
        }:
            return 7
        return 3

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_0_2_output):
        key = (position, attn_0_2_output)
        if key in {
            (0, "3"),
            (0, "4"),
            (0, "</s>"),
            (1, "4"),
            (1, "</s>"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (3, "4"),
            (3, "</s>"),
            (4, "1"),
            (4, "3"),
            (4, "4"),
            (4, "</s>"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "</s>"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (7, "1"),
            (7, "3"),
            (7, "4"),
            (7, "</s>"),
            (8, "3"),
            (8, "4"),
            (8, "</s>"),
            (9, "3"),
            (9, "4"),
            (9, "</s>"),
            (10, "1"),
            (10, "3"),
            (10, "4"),
            (10, "</s>"),
            (11, "3"),
            (11, "4"),
            (11, "</s>"),
            (12, "3"),
            (12, "4"),
            (12, "</s>"),
            (13, "3"),
            (13, "4"),
            (13, "</s>"),
            (14, "3"),
            (14, "4"),
            (14, "</s>"),
            (15, "1"),
            (15, "3"),
            (15, "4"),
            (15, "</s>"),
            (16, "1"),
            (16, "3"),
            (16, "4"),
            (16, "</s>"),
            (17, "3"),
            (17, "4"),
            (17, "</s>"),
            (18, "3"),
            (18, "4"),
            (18, "</s>"),
            (19, "3"),
            (19, "4"),
            (19, "</s>"),
        }:
            return 4
        elif key in {
            (0, "1"),
            (1, "1"),
            (1, "3"),
            (2, "1"),
            (3, "1"),
            (3, "3"),
            (8, "1"),
            (9, "1"),
            (11, "1"),
            (14, "1"),
            (17, "1"),
            (18, "1"),
            (19, "1"),
        }:
            return 13
        elif key in {(12, "1"), (13, "1")}:
            return 19
        return 14

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_0_2_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_1_0_output):
        key = (num_attn_1_3_output, num_attn_1_0_output)
        return 1

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 7

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(num_mlp_0_0_output, attn_1_3_output):
        if num_mlp_0_0_output in {0}:
            return attn_1_3_output == "1"
        elif num_mlp_0_0_output in {8, 1, 18}:
            return attn_1_3_output == "<s>"
        elif num_mlp_0_0_output in {2, 3, 4, 5, 6, 7, 10, 14, 15, 17}:
            return attn_1_3_output == ""
        elif num_mlp_0_0_output in {9}:
            return attn_1_3_output == "2"
        elif num_mlp_0_0_output in {11, 12, 13}:
            return attn_1_3_output == "3"
        elif num_mlp_0_0_output in {16, 19}:
            return attn_1_3_output == "4"

    attn_2_0_pattern = select_closest(
        attn_1_3_outputs, num_mlp_0_0_outputs, predicate_2_0
    )
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_2_output, mlp_1_0_output):
        if attn_1_2_output in {"4", "0", "<s>"}:
            return mlp_1_0_output == 5
        elif attn_1_2_output in {"</s>", "1"}:
            return mlp_1_0_output == 10
        elif attn_1_2_output in {"2"}:
            return mlp_1_0_output == 14
        elif attn_1_2_output in {"3"}:
            return mlp_1_0_output == 1

    attn_2_1_pattern = select_closest(mlp_1_0_outputs, attn_1_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_1_output, num_mlp_0_1_output):
        if mlp_0_1_output in {0, 8, 13}:
            return num_mlp_0_1_output == 3
        elif mlp_0_1_output in {1}:
            return num_mlp_0_1_output == 15
        elif mlp_0_1_output in {16, 2}:
            return num_mlp_0_1_output == 1
        elif mlp_0_1_output in {3}:
            return num_mlp_0_1_output == 5
        elif mlp_0_1_output in {4, 6, 14, 15, 18}:
            return num_mlp_0_1_output == 16
        elif mlp_0_1_output in {9, 5}:
            return num_mlp_0_1_output == 7
        elif mlp_0_1_output in {11, 7}:
            return num_mlp_0_1_output == 2
        elif mlp_0_1_output in {10, 19}:
            return num_mlp_0_1_output == 10
        elif mlp_0_1_output in {12}:
            return num_mlp_0_1_output == 12
        elif mlp_0_1_output in {17}:
            return num_mlp_0_1_output == 0

    attn_2_2_pattern = select_closest(
        num_mlp_0_1_outputs, mlp_0_1_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, tokens)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_1_0_output, mlp_1_1_output):
        if attn_1_0_output in {"0"}:
            return mlp_1_1_output == 2
        elif attn_1_0_output in {"1", "<s>"}:
            return mlp_1_1_output == 4
        elif attn_1_0_output in {"</s>", "2"}:
            return mlp_1_1_output == 1
        elif attn_1_0_output in {"3"}:
            return mlp_1_1_output == 12
        elif attn_1_0_output in {"4"}:
            return mlp_1_1_output == 6

    attn_2_3_pattern = select_closest(mlp_1_1_outputs, attn_1_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_0_1_output):
        if attn_1_2_output in {"0"}:
            return attn_0_1_output == "<s>"
        elif attn_1_2_output in {"1", "4", "<s>", "2", "3"}:
            return attn_0_1_output == "0"
        elif attn_1_2_output in {"</s>"}:
            return attn_0_1_output == ""

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_1_output, num_mlp_0_1_output):
        if mlp_0_1_output in {0, 1, 2, 3, 4, 5, 12, 15, 16, 18, 19}:
            return num_mlp_0_1_output == 1
        elif mlp_0_1_output in {6, 7}:
            return num_mlp_0_1_output == 18
        elif mlp_0_1_output in {8}:
            return num_mlp_0_1_output == 2
        elif mlp_0_1_output in {9}:
            return num_mlp_0_1_output == 5
        elif mlp_0_1_output in {10}:
            return num_mlp_0_1_output == 15
        elif mlp_0_1_output in {11}:
            return num_mlp_0_1_output == 6
        elif mlp_0_1_output in {17, 13}:
            return num_mlp_0_1_output == 17
        elif mlp_0_1_output in {14}:
            return num_mlp_0_1_output == 12

    num_attn_2_1_pattern = select(
        num_mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_1_output, attn_1_2_output):
        if attn_1_1_output in {"0"}:
            return attn_1_2_output == "<s>"
        elif attn_1_1_output in {"1", "3"}:
            return attn_1_2_output == "0"
        elif attn_1_1_output in {"2", "4", "<s>"}:
            return attn_1_2_output == "1"
        elif attn_1_1_output in {"</s>"}:
            return attn_1_2_output == ""

    num_attn_2_2_pattern = select(attn_1_2_outputs, attn_1_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_1_output, attn_0_2_output):
        if attn_0_1_output in {"0", "</s>", "1", "4", "2"}:
            return attn_0_2_output == "0"
        elif attn_0_1_output in {"3"}:
            return attn_0_2_output == ""
        elif attn_0_1_output in {"<s>"}:
            return attn_0_2_output == "1"

    num_attn_2_3_pattern = select(attn_0_2_outputs, attn_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position):
        key = position
        if key in {0, 1, 2, 8, 11, 13, 15, 18}:
            return 7
        return 19

    mlp_2_0_outputs = [mlp_2_0(k0) for k0 in positions]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position):
        key = position
        if key in {1}:
            return 13
        return 5

    mlp_2_1_outputs = [mlp_2_1(k0) for k0 in positions]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_2_1_output):
        key = (num_attn_2_3_output, num_attn_2_1_output)
        return 14

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {1, 2}:
            return 6
        elif key in {0}:
            return 4
        return 11

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_0_2_outputs]
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


#print(run(["<s>", "1", "0", "4", "</s>"]))
