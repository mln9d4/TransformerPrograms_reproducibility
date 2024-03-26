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
        "output/rasp/most_freq/vocab8maxlen8dvar20/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/most_freq_weights.csv",
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
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {8, 18}:
            return k_position == 16
        elif q_position in {9, 17}:
            return k_position == 13
        elif q_position in {10, 14, 15}:
            return k_position == 12
        elif q_position in {16, 11}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {13}:
            return k_position == 8
        elif q_position in {19}:
            return k_position == 15

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1}:
            return token == "2"
        elif position in {2, 8, 9, 10, 11, 14, 15, 16, 17, 18}:
            return token == "5"
        elif position in {3, 4, 5}:
            return token == "0"
        elif position in {6}:
            return token == "1"
        elif position in {7}:
            return token == "4"
        elif position in {19, 12, 13}:
            return token == ""

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3, 7}:
            return k_position == 4
        elif q_position in {4, 5}:
            return k_position == 6
        elif q_position in {8, 17}:
            return k_position == 16
        elif q_position in {9, 13}:
            return k_position == 18
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12, 15}:
            return k_position == 9
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {16, 19}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 15

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2, 4, 5}:
            return k_position == 2
        elif q_position in {3, 8, 9, 10, 11, 14, 16, 19}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 17
        elif q_position in {13}:
            return k_position == 5
        elif q_position in {17, 18, 15}:
            return k_position == 16

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 3, 4, 5}:
            return token == "1"
        elif position in {1}:
            return token == "3"
        elif position in {8, 2, 7}:
            return token == "<s>"
        elif position in {18, 6}:
            return token == ""
        elif position in {9, 10, 11, 12, 13, 15}:
            return token == "2"
        elif position in {16, 17, 19, 14}:
            return token == "5"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1, 10, 18, 15}:
            return token == "2"
        elif position in {2, 3, 8, 11, 12, 13, 14, 16, 17, 19}:
            return token == ""
        elif position in {4, 5, 6, 7}:
            return token == "<s>"
        elif position in {9}:
            return token == "1"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 14, 17}:
            return token == "5"
        elif position in {2, 3, 4, 6, 7, 8, 9, 11, 13, 15, 16, 19}:
            return token == "<s>"
        elif position in {10, 18, 12, 5}:
            return token == ""

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 19}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {8, 13, 15}:
            return token == "5"
        elif position in {17, 18, 11}:
            return token == ""
        elif position in {16}:
            return token == "1"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, position):
        key = (attn_0_1_output, position)
        if key in {
            ("0", 1),
            ("0", 8),
            ("0", 9),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("0", 16),
            ("0", 17),
            ("0", 18),
            ("0", 19),
            ("1", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("5", 1),
            ("<s>", 1),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 14),
            ("<s>", 16),
            ("<s>", 17),
        }:
            return 5
        elif key in {
            ("0", 6),
            ("1", 6),
            ("2", 6),
            ("3", 0),
            ("3", 6),
            ("4", 6),
            ("5", 0),
            ("5", 6),
            ("<s>", 6),
        }:
            return 14
        elif key in {
            ("0", 0),
            ("1", 0),
            ("1", 19),
            ("2", 0),
            ("2", 8),
            ("2", 9),
            ("2", 11),
            ("2", 12),
            ("2", 17),
            ("2", 18),
            ("2", 19),
            ("4", 0),
            ("<s>", 0),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 18),
            ("<s>", 19),
        }:
            return 15
        elif key in {
            ("0", 7),
            ("1", 7),
            ("2", 7),
            ("3", 7),
            ("4", 7),
            ("5", 7),
            ("<s>", 7),
        }:
            return 9
        elif key in {
            ("5", 2),
            ("5", 3),
            ("5", 4),
            ("5", 5),
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
        }:
            return 4
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, position):
        key = (attn_0_1_output, position)
        if key in {
            ("1", 4),
            ("1", 5),
            ("2", 5),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("4", 5),
            ("5", 4),
            ("5", 5),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
            ("<s>", 16),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 19),
        }:
            return 7
        elif key in {
            ("1", 0),
            ("1", 3),
            ("1", 6),
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
            ("5", 6),
        }:
            return 17
        elif key in {("0", 1), ("3", 1), ("4", 1), ("5", 1), ("<s>", 1)}:
            return 19
        elif key in {
            ("0", 7),
            ("1", 7),
            ("2", 7),
            ("3", 7),
            ("4", 7),
            ("5", 7),
            ("<s>", 7),
        }:
            return 4
        elif key in {
            ("0", 2),
            ("1", 2),
            ("2", 2),
            ("3", 2),
            ("4", 2),
            ("5", 2),
            ("<s>", 2),
        }:
            return 10
        elif key in {
            ("0", 3),
            ("0", 4),
            ("0", 5),
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
        }:
            return 15
        elif key in {
            ("1", 1),
            ("2", 0),
            ("2", 1),
            ("2", 3),
            ("2", 6),
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
        }:
            return 16
        return 18

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        if key in {0, 1}:
            return 0
        return 11

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 6

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_3_output, position):
        if attn_0_3_output in {"5", "0"}:
            return position == 7
        elif attn_0_3_output in {"1", "3"}:
            return position == 1
        elif attn_0_3_output in {"2"}:
            return position == 11
        elif attn_0_3_output in {"4"}:
            return position == 0
        elif attn_0_3_output in {"<s>"}:
            return position == 2

    attn_1_0_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 1, 2, 3, 4}:
            return token == "1"
        elif position in {5, 15}:
            return token == "<s>"
        elif position in {10, 6}:
            return token == "0"
        elif position in {7, 8, 9, 11, 12, 13, 14, 16, 17, 19}:
            return token == ""
        elif position in {18}:
            return token == "4"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, mlp_0_1_output):
        if position in {0}:
            return mlp_0_1_output == 2
        elif position in {1}:
            return mlp_0_1_output == 4
        elif position in {2, 6, 7}:
            return mlp_0_1_output == 7
        elif position in {3, 4, 5}:
            return mlp_0_1_output == 16
        elif position in {8, 9, 11, 12, 13, 15, 16, 17, 18, 19}:
            return mlp_0_1_output == 15
        elif position in {10}:
            return mlp_0_1_output == 14
        elif position in {14}:
            return mlp_0_1_output == 19

    attn_1_2_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 2, 6, 7}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {3, 4, 5}:
            return k_position == 6
        elif q_position in {8, 10, 11, 13, 15}:
            return k_position == 18
        elif q_position in {9, 12, 14, 16, 17, 18, 19}:
            return k_position == 15

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_0_output):
        if position in {0, 14, 7}:
            return attn_0_0_output == "5"
        elif position in {1, 18}:
            return attn_0_0_output == "1"
        elif position in {2, 3, 4, 5, 6, 9}:
            return attn_0_0_output == ""
        elif position in {8, 10, 12}:
            return attn_0_0_output == "4"
        elif position in {16, 17, 11, 15}:
            return attn_0_0_output == "2"
        elif position in {19, 13}:
            return attn_0_0_output == "3"

    num_attn_1_0_pattern = select(attn_0_0_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 3, 4, 5, 6, 7, 15, 17, 18}:
            return token == "3"
        elif mlp_0_1_output in {1, 8, 10, 12, 13, 16, 19}:
            return token == "2"
        elif mlp_0_1_output in {2, 11}:
            return token == "<s>"
        elif mlp_0_1_output in {9}:
            return token == ""
        elif mlp_0_1_output in {14}:
            return token == "5"

    num_attn_1_1_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 2, 10}:
            return token == "<s>"
        elif mlp_0_1_output in {1, 8, 13, 16, 19}:
            return token == "3"
        elif mlp_0_1_output in {3, 4, 5, 7, 15, 17, 18}:
            return token == "4"
        elif mlp_0_1_output in {6}:
            return token == "2"
        elif mlp_0_1_output in {9, 11, 12, 14}:
            return token == ""

    num_attn_1_2_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 2, 5}:
            return token == "2"
        elif mlp_0_1_output in {1, 13}:
            return token == "3"
        elif mlp_0_1_output in {3, 7}:
            return token == "5"
        elif mlp_0_1_output in {4, 8, 10, 11, 16, 19}:
            return token == ""
        elif mlp_0_1_output in {6}:
            return token == "4"
        elif mlp_0_1_output in {9, 12, 14, 15, 17, 18}:
            return token == "0"

    num_attn_1_3_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("0", 7),
            ("0", 11),
            ("0", 14),
            ("0", 17),
            ("0", 18),
            ("4", 0),
            ("4", 1),
            ("4", 9),
            ("4", 13),
            ("4", 16),
            ("5", 0),
            ("5", 1),
            ("5", 9),
            ("5", 13),
            ("5", 16),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 9),
            ("<s>", 13),
            ("<s>", 16),
        }:
            return 19
        elif key in {
            ("0", 0),
            ("0", 1),
            ("0", 8),
            ("0", 9),
            ("0", 13),
            ("0", 16),
            ("0", 19),
        }:
            return 7
        return 10

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_2_output, attn_1_0_output):
        key = (attn_0_2_output, attn_1_0_output)
        if key in {
            ("1", "1"),
            ("1", "3"),
            ("1", "<s>"),
            ("5", "1"),
            ("5", "3"),
            ("5", "<s>"),
        }:
            return 5
        return 8

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0, 1, 2, 3, 4, 5}:
            return 2
        return 6

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        if key in {
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
            (9, 0),
            (9, 1),
            (10, 0),
            (10, 1),
            (10, 2),
            (11, 0),
            (11, 1),
            (11, 2),
            (12, 0),
            (12, 1),
            (12, 2),
            (13, 0),
            (13, 1),
            (13, 2),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
        }:
            return 18
        return 3

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, mlp_0_0_output):
        if token in {"1", "2", "0", "4"}:
            return mlp_0_0_output == 4
        elif token in {"3"}:
            return mlp_0_0_output == 12
        elif token in {"5"}:
            return mlp_0_0_output == 5
        elif token in {"<s>"}:
            return mlp_0_0_output == 18

    attn_2_0_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"1", "5", "0", "2", "4"}:
            return k_token == "<s>"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"1", "5", "0", "<s>", "2", "4"}:
            return k_token == "<s>"
        elif q_token in {"3"}:
            return k_token == "3"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 7, 11, 12, 13, 14, 15, 16, 19}:
            return k_position == 18
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2, 4, 5}:
            return k_position == 2
        elif q_position in {3, 6}:
            return k_position == 3
        elif q_position in {8, 9, 18, 17}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 9

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_1_output, attn_0_1_output):
        if mlp_0_1_output in {0}:
            return attn_0_1_output == "1"
        elif mlp_0_1_output in {1, 6, 8, 13, 16, 19}:
            return attn_0_1_output == "3"
        elif mlp_0_1_output in {2}:
            return attn_0_1_output == "5"
        elif mlp_0_1_output in {3, 7, 9, 11, 15, 17, 18}:
            return attn_0_1_output == "2"
        elif mlp_0_1_output in {4, 5}:
            return attn_0_1_output == "0"
        elif mlp_0_1_output in {10, 12, 14}:
            return attn_0_1_output == ""

    num_attn_2_0_pattern = select(attn_0_1_outputs, mlp_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "<s>"
        elif attn_0_0_output in {"1", "5", "3", "2", "4"}:
            return token == "0"
        elif attn_0_0_output in {"<s>"}:
            return token == ""

    num_attn_2_1_pattern = select(tokens, attn_0_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_0_output, token):
        if attn_0_0_output in {"1", "5", "0", "3", "2"}:
            return token == "4"
        elif attn_0_0_output in {"4"}:
            return token == "0"
        elif attn_0_0_output in {"<s>"}:
            return token == "1"

    num_attn_2_2_pattern = select(tokens, attn_0_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_0_output, token):
        if attn_0_0_output in {"1", "5", "0", "2", "4"}:
            return token == "3"
        elif attn_0_0_output in {"3"}:
            return token == "2"
        elif attn_0_0_output in {"<s>"}:
            return token == "<s>"

    num_attn_2_3_pattern = select(tokens, attn_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_2_output, token):
        key = (attn_0_2_output, token)
        if key in {("1", "1")}:
            return 18
        return 12

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, tokens)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(token, attn_0_0_output):
        key = (token, attn_0_0_output)
        if key in {("4", "4"), ("<s>", "4")}:
            return 3
        elif key in {("5", "5"), ("<s>", "5")}:
            return 12
        return 17

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(tokens, attn_0_0_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_3_output, num_attn_1_0_output):
        key = (num_attn_1_3_output, num_attn_1_0_output)
        return 4

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_0_output, num_attn_2_1_output):
        key = (num_attn_1_0_output, num_attn_2_1_output)
        return 9

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_1_outputs)
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
