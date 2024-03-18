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


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/sort/sort_weights.csv", index_col=[0, 1], dtype={"feature": str}
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
            return token == "3"
        elif position in {1, 2}:
            return token == "1"
        elif position in {3, 4, 5}:
            return token == "2"
        elif position in {6}:
            return token == "4"
        elif position in {7}:
            return token == "0"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 5, 7}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 1

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 3, 4}:
            return token == "3"
        elif position in {1, 2, 7}:
            return token == "0"
        elif position in {5, 6}:
            return token == "4"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 2, 7}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {5, 6}:
            return k_position == 4

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_1_output):
        key = (position, attn_0_1_output)
        if key in {
            (0, "</s>"),
            (0, "<s>"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "1"),
            (5, "</s>"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (6, "<s>"),
            (7, "</s>"),
            (7, "<s>"),
        }:
            return 6
        elif key in {
            (0, "0"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "<s>"),
            (2, "0"),
            (3, "0"),
            (5, "0"),
            (7, "0"),
        }:
            return 1
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_1_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 4, 5}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 0

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 3, 4}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 0
        elif q_position in {7}:
            return k_position == 6

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1}:
            return token == "1"
        elif position in {2, 3, 4}:
            return token == "<s>"
        elif position in {5}:
            return token == "4"
        elif position in {6}:
            return token == "3"
        elif position in {7}:
            return token == "<pad>"

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {2, 3, 4, 7}:
            return token == "</s>"
        elif position in {5}:
            return token == "<s>"
        elif position in {6}:
            return token == "4"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_2_output):
        key = (position, attn_1_2_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (1, "0"),
            (1, "1"),
            (2, "0"),
            (2, "1"),
            (3, "0"),
            (3, "1"),
            (3, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "<s>"),
            (7, "0"),
            (7, "1"),
        }:
            return 6
        elif key in {
            (0, "2"),
            (1, "2"),
            (2, "2"),
            (2, "<s>"),
            (3, "2"),
            (4, "2"),
            (7, "2"),
        }:
            return 3
        elif key in {
            (0, "<s>"),
            (1, "<s>"),
            (5, "0"),
            (5, "<s>"),
            (6, "<s>"),
            (7, "</s>"),
            (7, "<s>"),
        }:
            return 2
        elif key in {(1, "3"), (1, "4"), (1, "</s>")}:
            return 4
        elif key in {(6, "0"), (6, "1")}:
            return 7
        return 0

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_2_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
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
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                one_scores,
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


print(run(["<s>", "0", "0", "1", "2", "3", "1", "</s>"]))
