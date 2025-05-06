"""
Reference: https://github.com/UpstageAI/UpScore/blob/v1.5.0/upscore/evaluation/upscore.py
"""

import argparse
import copy
import glob
import os
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import yaml
from Levenshtein import distance
from scipy.optimize import linear_sum_assignment

from .utils import (
    defaultdict_to_dict,
    get_f1_recall_precision_accuracy_upscore,
    get_ontology_keys,
    get_TP_FP_FN,
    print_upscore,
    read_non_group_and_group_entities,
    sort_indices_based_on_scores,
    update_eval_criteria,
)


def get_non_group_confusion_matrix(
    gold: Dict[str, Any],
    pred: Dict[str, Any],
    confusion_matrix: Dict[str, Any],
    empty_token: str,
    value_delimiter: str,
    split_merged_value: bool,
    include_empty_token_in_score_calculation: bool,
    eval_criteria: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool]:
    """
    Calculates the confusion matrix at non_group for given 'gold' and 'predicted' dictionaries.
    The confusion matrix here includes True Positives (TP), False Positives (FP), and False Negatives (FN)
    for each key in the union of keys from both gold and predicted dictionaries.

    Args:
        gold (Dict[str, Any]): A dictionary with true values.
        pred (Dict[str, Any]): A dictionary with predicted values.
        confusion_matrix (Dict[str, Any]): A dictionary to store the confusion matrix data.
        empty_token (str): A string representing the empty token symbol.
        value_delimiter (str): A string representing the delimiter for values.
        split_merged_value (bool): A flag indicating if the merged values should be split.
        include_empty_token_in_score_calculation (bool): A flag indicating if the empty token should be included in the score calculation.
        eval_criteria (Dict[str, Dict[str, Any]]): A dictionary containing the evaluation criteria for each key.

    Returns:
        The first element is the confusion matrix dictionary updated with the current key's TP, FP, and FN quantities.
        The second element is a boolean representing whether a perfect match has been achieved for all keys.

    """
    exact_match_only = []
    if not gold and not pred:
        return confusion_matrix, exact_match_only

    # Remove characters based on the eval_criteria
    for g_k, g_v_list in gold.items():
        criterion = eval_criteria[g_k]
        if split_merged_value:
            for g_v_idx, g_v in enumerate(g_v_list):
                split_values = g_v.split(value_delimiter)
                for v_idx, v in enumerate(split_values):
                    for ch in criterion["remove_chars"]:
                        split_values[v_idx] = split_values[v_idx].replace(ch, "")
                merged_value = value_delimiter.join(split_values)
                gold[g_k][g_v_idx] = merged_value
        else:
            for g_v_idx in range(len(g_v_list)):
                for ch in criterion["remove_chars"]:
                    gold[g_k][g_v_idx] = gold[g_k][g_v_idx].replace(ch, "")

    for p_k, p_v_list in pred.items():
        criterion = eval_criteria[p_k]
        if split_merged_value:
            for p_v_idx, p_v in enumerate(p_v_list):
                split_values = p_v.split(value_delimiter)
                for v_idx, v in enumerate(split_values):
                    for ch in criterion["remove_chars"]:
                        split_values[v_idx] = split_values[v_idx].replace(ch, "")
                merged_value = value_delimiter.join(split_values)
                pred[p_k][p_v_idx] = merged_value
        else:
            for p_v_idx in range(len(p_v_list)):
                for ch in criterion["remove_chars"]:
                    pred[p_k][p_v_idx] = pred[p_k][p_v_idx].replace(ch, "")

    union_keys = set(gold.keys()).union(set(pred.keys()))

    for key in list(union_keys):
        TP, FP, FN = 0, 0, 0
        if key in gold:
            if split_merged_value:
                if key not in pred:
                    pred_values = []
                else:
                    pred_values = [
                        v
                        for merged_value in pred[key]
                        for v in merged_value.split(value_delimiter)
                        if v != empty_token or include_empty_token_in_score_calculation
                    ]
                gold_values = [
                    v
                    for merged_value in gold[key]
                    for v in merged_value.split(value_delimiter)
                    if v != empty_token or include_empty_token_in_score_calculation
                ]
            else:
                if key not in pred:
                    pred_values = []
                else:
                    pred_values = [
                        merged_value
                        for merged_value in pred[key]
                        if merged_value != empty_token or include_empty_token_in_score_calculation
                    ]
                gold_values = [
                    merged_value
                    for merged_value in gold[key]
                    if merged_value != empty_token or include_empty_token_in_score_calculation
                ]
            TP, FP, FN = get_TP_FP_FN(gold_values, pred_values, eval_criteria[key])
            wrong_match = min(FP, FN)
            no_match = max(FP, FN) - wrong_match

            if FP < 0 or FN < 0:
                raise ValueError("False Positive (FP) and False Negative (FN) should not be less than zero.")
            confusion_matrix[key]["TP"] += TP
            confusion_matrix[key]["FP"] += FP
            confusion_matrix[key]["FN"] += FN
            confusion_matrix[key]["wrong_match"] += wrong_match
            confusion_matrix[key]["no_match"] += no_match
            confusion_matrix[key]["label"] += len(gold_values)
        else:
            # key not in gold. it is a false positive
            pred_values = [v for v in pred[key] if v != empty_token or include_empty_token_in_score_calculation]
            FP = len(pred_values)
            confusion_matrix[key]["FP"] += FP
            confusion_matrix[key]["no_match"] += FP

        if TP > 0 and FP + FN == 0:
            exact_match_only.append(True)
        else:
            exact_match_only.append(False)

    return confusion_matrix, exact_match_only


def get_group_confusion_matrix(
    gold: List[Dict[str, Any]],
    pred: List[Dict[str, Any]],
    confusion_matrix: Dict[str, Any],
    doc_name: str,
    empty_token: str,
    value_delimiter: str,
    grouping_strategy: str,
    split_merged_value: bool,
    include_empty_token_in_score_calculation: bool,
    eval_criteria: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool]:
    """
    Calculates the confusion matrix at group level for given 'gold' and 'predicted' dictionaries.
    The confusion matrix here includes True Positives (TP), False Positives (FP), and False Negatives (FN)

    Args:
        gold: A list of dictionaries containing the true values.
        pred: A list of dictionaries containing the predicted values.
        confusion_matrix: A dictionary that stores the scores for True Positives(TP), False Positives(FP), and False Negatives(FN).
        doc_name: Name of the document. Used for keying the confusion matrix
        empty_token: A string that represents a missing or empty prediction.
        value_delimiter: A string that is used to separate the different values in a non_group prediction or gold annotation.
        grouping_strategy: A string used to define the strategy for grouping entities.
        split_merged_value: A boolean that if True, the function will split merged values.
        include_empty_token_in_score_calculation (bool): A flag indicating if the empty token should be included in the score calculation.
        eval_criteria: A dictionary containing the evaluation criteria for each key.

    Returns:
        confusion_matrix: A dictionary that stores the scores for True Positives(TP), False Positives(FP), and False Negatives(FN).
        exact_match_only: A list of boolean values indicating whether the prediction is an exact match.
    """

    exact_match_only = []
    if not gold and not pred:
        return confusion_matrix, exact_match_only

    # Remove characters based on the eval_criteria
    for group_id, gold_row in gold.items():
        for g_row_idx, g_kvs in enumerate(gold_row):
            for g_k, g_v in g_kvs.items():
                criterion = eval_criteria[g_k]
                if split_merged_value:
                    split_values = g_v.split(value_delimiter)
                    for v_idx, v in enumerate(split_values):
                        for ch in criterion["remove_chars"]:
                            split_values[v_idx] = split_values[v_idx].replace(ch, "")
                    merged_value = value_delimiter.join(split_values)
                    gold[group_id][g_row_idx][g_k] = merged_value
                else:
                    for ch in criterion["remove_chars"]:
                        gold[group_id][g_row_idx][g_k] = gold[group_id][g_row_idx][g_k].replace(ch, "")

    for group_id, pred_row in pred.items():
        for p_row_idx, p_kvs in enumerate(pred_row):
            for p_k, p_v in p_kvs.items():
                criterion = eval_criteria[p_k]
                if split_merged_value:
                    split_values = p_v.split(value_delimiter)
                    for v_idx, v in enumerate(split_values):
                        for ch in criterion["remove_chars"]:
                            split_values[v_idx] = split_values[v_idx].replace(ch, "")
                    merged_value = value_delimiter.join(split_values)
                    pred[group_id][p_row_idx][p_k] = merged_value
                else:
                    for ch in criterion["remove_chars"]:
                        pred[group_id][p_row_idx][p_k] = pred[group_id][p_row_idx][p_k].replace(ch, "")

    union_keys = set(gold.keys()).union(set(pred.keys()))
    for group_id in list(union_keys):
        TP, FP, FN, label = 0, 0, 0, 0

        if group_id not in gold:
            # count wrong prediction. group_id not in gold
            for p_idx, p_kv in enumerate(pred[group_id]):
                # grouping count wrong prediction, no label(=gold)
                confusion_matrix[f"{group_id}.grouping"]["FP"] += 1
                confusion_matrix[f"{group_id}.grouping"]["no_match"] += 1
                for p_k, p_v in p_kv.items():
                    if split_merged_value:
                        pred_v = [
                            pv
                            for pv in p_v.split(value_delimiter)
                            if pv != empty_token or include_empty_token_in_score_calculation
                        ]
                        FP += len(pred_v)
                        confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += len(pred_v)
                        confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += len(pred_v)
                    else:
                        if p_v != empty_token or include_empty_token_in_score_calculation:
                            FP += 1
                            confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += 1
                            confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += 1
        else:
            # build score matrix
            group_match_score_matrix = defaultdict(lambda: defaultdict(int))

            # greedy search for max exact match score
            for g_idx, g_kv in enumerate(gold[group_id]):
                for p_idx, p_kv in enumerate(pred[group_id]):
                    group_match_score_matrix[g_idx][p_idx] += 0
                    for p_k, p_v in p_kv.items():
                        # exist pred_key in gold
                        if p_k in g_kv:
                            criterion = eval_criteria[p_k]

                            if split_merged_value:
                                # split merged value
                                gold_v = [
                                    gv
                                    for gv in g_kv[p_k].split(value_delimiter)
                                    if gv != empty_token or include_empty_token_in_score_calculation
                                ]
                                pred_v = [
                                    pv
                                    for pv in p_v.split(value_delimiter)
                                    if pv != empty_token or include_empty_token_in_score_calculation
                                ]

                                if criterion["match"] == "exact_match":
                                    for v in pred_v:
                                        if v in gold_v:
                                            group_match_score_matrix[g_idx][p_idx] += 1
                                            gold_v.remove(v)
                                elif criterion["match"] == "partial_match":
                                    for pv in pred_v:
                                        min_ned = 1
                                        min_gv_idx = -1
                                        for gv_idx, gv in enumerate(gold_v):
                                            denom = max(len(pv), len(gv))
                                            if denom == 0:
                                                ned = 0
                                            else:
                                                ned = distance(pv, gv) / denom
                                            if ned < min_ned:
                                                min_ned = ned
                                                min_gv_idx = gv_idx
                                        if min_ned <= criterion["ned_th"]:
                                            group_match_score_matrix[g_idx][p_idx] += 1
                                            gold_v.pop(min_gv_idx)
                                else:
                                    raise ValueError(f"Unknown match criterion: {criterion['match']}")
                            else:
                                if criterion["match"] == "exact_match":
                                    if p_v == g_kv[p_k] and (
                                        p_v != empty_token or include_empty_token_in_score_calculation
                                    ):
                                        group_match_score_matrix[g_idx][p_idx] += 1
                                elif criterion["match"] == "partial_match":
                                    denom = max(len(p_v), len(g_kv[p_k]))
                                    if denom == 0:
                                        ned = 0
                                    else:
                                        ned = distance(p_v, g_kv[p_k]) / denom

                                    if ned <= criterion["ned_th"]:
                                        group_match_score_matrix[g_idx][p_idx] += 1
                                else:
                                    raise ValueError(f"Unknown match criterion: {criterion['match']}")

            # sort score matrix by grouping_strategy algorithm
            sorted_key_value_pairs = sort_indices_based_on_scores(group_match_score_matrix, strategy=grouping_strategy)

            remove_g_idx = []
            remove_p_idx = []
            # find max exact match score
            for (g_idx, p_idx), max_em_score in sorted_key_value_pairs:
                if g_idx not in remove_g_idx and p_idx not in remove_p_idx:
                    remove_g_idx.append(g_idx)
                    remove_p_idx.append(p_idx)

                    TP += max_em_score
                    gold_v = gold[group_id][g_idx]
                    pred_v = pred[group_id][p_idx]

                    gold_v = {
                        k: v for k, v in gold_v.items() if v != empty_token or include_empty_token_in_score_calculation
                    }
                    pred_v = {
                        k: v for k, v in pred_v.items() if v != empty_token or include_empty_token_in_score_calculation
                    }

                    # count exact match and wrong prediction grouping
                    all_matched = False
                    if len(gold_v.keys()) == len(pred_v.keys()):
                        key_matched_list = [False] * len(gold_v.keys())
                        for k_idx, k in enumerate(gold_v.keys()):
                            if k not in pred_v:
                                break

                            criterion = eval_criteria[k]
                            if criterion["match"] == "exact_match":
                                if gold_v[k] == pred_v[k]:
                                    key_matched_list[k_idx] = True
                            elif criterion["match"] == "partial_match":
                                denom = max(len(gold_v[k]), len(pred_v[k]))
                                if denom == 0:
                                    ned = 0
                                else:
                                    ned = distance(gold_v[k], pred_v[k]) / denom
                                if ned <= criterion["ned_th"]:
                                    key_matched_list[k_idx] = True

                        if all(key_matched_list):
                            all_matched = True

                    if all_matched:
                        confusion_matrix[f"{group_id}.grouping"]["TP"] += 1
                    else:
                        confusion_matrix[f"{group_id}.grouping"]["FP"] += 1
                        confusion_matrix[f"{group_id}.grouping"]["FN"] += 1
                        confusion_matrix[f"{group_id}.grouping"]["wrong_match"] += 1
                    # count label(=gold)
                    confusion_matrix[f"{group_id}.grouping"]["label"] += 1

                    for p_k, p_v in pred_v.items():
                        if p_k in gold_v:
                            criterion = eval_criteria[p_k]

                            if split_merged_value:
                                gold_list = gold_v[p_k].split(value_delimiter)
                                pred_list = p_v.split(value_delimiter)
                                group_TP, group_FP, group_FN = get_TP_FP_FN(gold_list, pred_list, criterion)
                                group_wrong_match = min(group_FP, group_FN)
                                group_no_match = max(group_FP, group_FN) - group_wrong_match
                                FP += group_FP
                                FN += group_FN
                                label += len(gold_list)
                                confusion_matrix[f"{p_k}.{group_id}.group"]["TP"] += group_TP
                                confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += group_FP
                                confusion_matrix[f"{p_k}.{group_id}.group"]["FN"] += group_FN
                                confusion_matrix[f"{p_k}.{group_id}.group"]["wrong_match"] += group_wrong_match
                                confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += group_no_match
                                confusion_matrix[f"{p_k}.{group_id}.group"]["label"] += len(gold_list)
                            else:
                                # Calculate matching based on the criterion
                                matched = False
                                if criterion["match"] == "exact_match":
                                    if p_v == gold_v[p_k]:
                                        confusion_matrix[f"{p_k}.{group_id}.group"]["TP"] += 1
                                        matched = True
                                elif criterion["match"] == "partial_match":
                                    denom = max(len(p_v), len(gold_v[p_k]))
                                    if denom == 0:
                                        ned = 0
                                    else:
                                        ned = distance(p_v, gold_v[p_k]) / denom
                                    if ned <= criterion["ned_th"]:
                                        confusion_matrix[f"{p_k}.{group_id}.group"]["TP"] += 1
                                        matched = True
                                else:
                                    raise ValueError(f"Unknown match criterion: {criterion['match']}")

                                if not matched:
                                    FP += 1
                                    FN += 1
                                    confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += 1
                                    confusion_matrix[f"{p_k}.{group_id}.group"]["FN"] += 1
                                    confusion_matrix[f"{p_k}.{group_id}.group"]["wrong_match"] += 1

                                label += 1

                                confusion_matrix[f"{p_k}.{group_id}.group"]["label"] += 1
                        else:
                            # predict key is not in gold
                            if split_merged_value:
                                group_FP = len(p_v.split(value_delimiter))
                                FP += group_FP
                                confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += group_FP
                                confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += group_FP
                            else:
                                FP += 1
                                confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += 1
                                confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += 1

                    missing_keys = set(gold_v.keys()) - set(pred_v.keys())

                    for m_k in missing_keys:
                        if split_merged_value:
                            group_FN = len(gold_v[m_k].split(value_delimiter))
                            FN += group_FN
                            label += group_FN
                            confusion_matrix[f"{m_k}.{group_id}.group"]["FN"] += group_FN
                            confusion_matrix[f"{m_k}.{group_id}.group"]["no_match"] += group_FN
                            confusion_matrix[f"{m_k}.{group_id}.group"]["label"] += group_FN
                        else:
                            FN += 1
                            label += 1
                            confusion_matrix[f"{m_k}.{group_id}.group"]["FN"] += 1
                            confusion_matrix[f"{m_k}.{group_id}.group"]["no_match"] += 1
                            confusion_matrix[f"{m_k}.{group_id}.group"]["label"] += 1

            # count wrong prediction
            not_removed_p_idx = []
            not_removed_g_idx = []
            for p_idx, p_kv in enumerate(pred[group_id]):
                if p_idx in remove_p_idx:
                    continue
                not_removed_p_idx.append(p_idx)
                # grouping count wrong prediction
                confusion_matrix[f"{group_id}.grouping"]["FP"] += 1
                confusion_matrix[f"{group_id}.grouping"]["no_match"] += 1
                for p_k, p_v in p_kv.items():
                    if split_merged_value:
                        pred_v = [
                            pv
                            for pv in p_v.split(value_delimiter)
                            if pv != empty_token or include_empty_token_in_score_calculation
                        ]
                        FP += len(pred_v)
                        confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += len(pred_v)
                        confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += len(pred_v)
                    else:
                        if p_v != empty_token or include_empty_token_in_score_calculation:
                            FP += 1
                            confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += 1
                            confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += 1

            # count miss prediction
            for g_idx, g_kv in enumerate(gold[group_id]):
                if g_idx in remove_g_idx:
                    continue
                not_removed_g_idx.append(g_idx)

                all_empty = True
                for g_k, g_v in g_kv.items():
                    if split_merged_value:
                        gold_v = [
                            gv
                            for gv in g_v.split(value_delimiter)
                            if gv != empty_token or include_empty_token_in_score_calculation
                        ]
                        FN += len(gold_v)
                        label += len(gold_v)
                        confusion_matrix[f"{g_k}.{group_id}.group"]["FN"] += len(gold_v)
                        confusion_matrix[f"{g_k}.{group_id}.group"]["no_match"] += len(gold_v)
                        confusion_matrix[f"{g_k}.{group_id}.group"]["label"] += len(gold_v)
                        all_empty = all_empty and len(gold_v) == 0
                    else:
                        if g_v != empty_token or include_empty_token_in_score_calculation:
                            FN += 1
                            label += 1
                            confusion_matrix[f"{g_k}.{group_id}.group"]["FN"] += 1
                            confusion_matrix[f"{g_k}.{group_id}.group"]["no_match"] += 1
                            confusion_matrix[f"{g_k}.{group_id}.group"]["label"] += 1
                            all_empty = False

                # grouping count miss prediction and label(=gold)
                if not all_empty:
                    confusion_matrix[f"{group_id}.grouping"]["FN"] += 1
                    confusion_matrix[f"{group_id}.grouping"]["no_match"] += 1
                    confusion_matrix[f"{group_id}.grouping"]["label"] += 1
        if FP < 0 or FN < 0:
            raise ValueError("False Positive (FP) and False Negative (FN) should not be less than zero.")
        if TP > 0 and FN + FP == 0:
            exact_match_only.append(True)
        else:
            exact_match_only.append(False)

    return confusion_matrix, exact_match_only


def get_upscore(
    confusion_matrix: Dict[str, Any],
    score_scale: int = 100,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Computes a suite of evaluation metrics (precision, recall, micro_f1, macro_f1, accuracy)
    for the given confusion matrix, as well as the overall total.

    Args:
        confusion_matrix (Dict[str, Any]): A dictionary representing the confusion matrix,
        where each key corresponds to a specific classification category and the value is
        another dictionary with counts of 'True Positives', 'False Positives', 'False Negatives',
        and total labels.
        score_scale (int): A scaling factor for the scores. Default is 100.

    Returns:
        entity_upscore (Dict[str, Dict[str, int]]): The first dictionary of this tuple where each key represents
        a specific classification category and corresponding value is a dictionary with calculated scores of precision,
        recall, micro_f1, macro_f1, and accuracy.
        total_upscore (Dict[str, Dict[str, int]]): The second dictionary of this tuple where the key represents the type
        of total score (total.non_group or total.group) and the corresponding value is a dictionary of calculated scores
        (precision, recall, micro_f1, macro_f1, accuracy).
    """
    entity_upscore = defaultdict(lambda: defaultdict(int))
    total_upscore = defaultdict(lambda: defaultdict(int))

    (
        total_upscore_TP,
        total_upscore_FN,
        total_upscore_FP,
        total_upscore_wrong_match,
        total_upscore_no_match,
        total_upscore_label,
    ) = (0, 0, 0, 0, 0, 0)
    non_group_TP, non_group_FN, non_group_FP, non_group_wrong_match, non_group_no_match, non_group_label = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    group_TP, group_FN, group_FP, group_wrong_match, group_no_match, group_label = 0, 0, 0, 0, 0, 0
    grouping_TP, grouping_FN, grouping_FP, grouping_wrong_match, grouping_no_match, grouping_label = 0, 0, 0, 0, 0, 0
    # Calculate metric score by key

    for key, conf_mat in confusion_matrix.items():
        TP, FN, FP, wrong_match, no_match, label = (
            conf_mat["TP"],
            conf_mat["FN"],
            conf_mat["FP"],
            conf_mat["wrong_match"],
            conf_mat["no_match"],
            conf_mat["label"],
        )
        if key.endswith(".group"):
            group_TP += TP
            group_FN += FN
            group_FP += FP
            group_wrong_match += wrong_match
            group_no_match += no_match
            group_label += label
        elif key.endswith(".grouping"):
            grouping_TP += TP
            grouping_FN += FN
            grouping_FP += FP
            grouping_wrong_match += wrong_match
            grouping_no_match += no_match
            grouping_label += label
        else:
            non_group_TP += TP
            non_group_FN += FN
            non_group_FP += FP
            non_group_wrong_match += wrong_match
            non_group_no_match += no_match
            non_group_label += label

        entity_upscore = get_f1_recall_precision_accuracy_upscore(
            metrics=entity_upscore,
            key=key,
            label=label,
            TP=TP,
            FN=FN,
            FP=FP,
            wrong_match=wrong_match,
            no_match=no_match,
            score_scale=score_scale,
        )

    # Calculate the overall metric score
    group_f1_list = [entity_upscore[k]["micro_f1"] for k, v in entity_upscore.items() if "group_id" in v]
    non_group_f1_list = [entity_upscore[k]["micro_f1"] for k, v in entity_upscore.items() if not k.endswith(".group")]
    grouping_f1_list = [
        entity_upscore[k]["micro_f1"] for k, v in entity_upscore.items() if not k.endswith(".grouping")
    ]
    total_upscore_f1_list = []

    if non_group_label > 0:
        total_upscore = get_f1_recall_precision_accuracy_upscore(
            metrics=total_upscore,
            key="total.non_group",
            label=non_group_label,
            TP=non_group_TP,
            FN=non_group_FN,
            FP=non_group_FP,
            wrong_match=non_group_wrong_match,
            no_match=non_group_no_match,
            f1_list=non_group_f1_list,
            score_scale=score_scale,
        )
        total_upscore_label += non_group_label
        total_upscore_TP += non_group_TP
        total_upscore_FN += non_group_FN
        total_upscore_FP += non_group_FP
        total_upscore_wrong_match += non_group_wrong_match
        total_upscore_no_match += non_group_no_match
        total_upscore_f1_list += non_group_f1_list

    if group_label > 0:
        # group entity score
        total_upscore = get_f1_recall_precision_accuracy_upscore(
            metrics=total_upscore,
            key="total.group",
            label=group_label,
            TP=group_TP,
            FN=group_FN,
            FP=group_FP,
            wrong_match=group_wrong_match,
            no_match=group_no_match,
            f1_list=group_f1_list,
            score_scale=score_scale,
        )
        # total grouping score
        total_upscore = get_f1_recall_precision_accuracy_upscore(
            metrics=total_upscore,
            key="total.grouping",
            label=grouping_label,
            TP=grouping_TP,
            FN=grouping_FN,
            FP=grouping_FP,
            wrong_match=grouping_wrong_match,
            no_match=grouping_no_match,
            f1_list=grouping_f1_list,
            score_scale=score_scale,
        )
        total_upscore_label += group_label
        total_upscore_TP += group_TP
        total_upscore_FN += group_FN
        total_upscore_FP += group_FP
        total_upscore_wrong_match += group_wrong_match
        total_upscore_no_match += group_no_match
        total_upscore_f1_list += group_f1_list

    # total entity score (non_group + group)
    total_upscore = get_f1_recall_precision_accuracy_upscore(
        metrics=total_upscore,
        key="total.upscore",
        label=total_upscore_label,
        TP=total_upscore_TP,
        FN=total_upscore_FN,
        FP=total_upscore_FP,
        wrong_match=total_upscore_wrong_match,
        no_match=total_upscore_no_match,
        f1_list=total_upscore_f1_list,
        score_scale=score_scale,
    )

    # check key in total upscore
    for key in ["total.non_group", "total.group", "total.upscore", "total.grouping"]:
        if key not in total_upscore:
            total_upscore[key] = None

    return entity_upscore, total_upscore


def upscore(
    gold_csv_files: List[str],
    pred_csv_files: List[str],
    shortlist: List[str],
    empty_token: str,
    grouping_strategy: str,
    value_delimiter: str,
    entity_type_delimiter: str,
    split_merged_value: bool,
    include_empty_token_in_score_calculation: bool,
    print_score: bool = True,
    score_scale: int = 100,
    eval_criteria: Union[Dict[str, Any], None] = None,
) -> Tuple[str, Dict, Dict]:
    """
    Calculates the performance of model's predictions at three levels: non_group-level, group-level, and document-level.

    Args:
        gold_csv_files (List[str]): A list of file paths for the gold standard (actual) entities in CSV format.
        pred_csv_files (List[str]): A list of file paths for the predicted entities in CSV format.
        shortlist (List[str]): A list of used keys for calculating upscore.
        empty_token (str): A string representing the empty token symbol.
        grouping_strategy (str): A string representing the strategy for grouping entities.
        value_delimiter (str): A string representing the delimiter for values.
        entity_type_delimiter (str): A string representing the delimiter for entity types.
        split_merged_value (bool): A flag indicating if the merged values should be split.
        include_empty_token_in_score_calculation (bool): A flag indicating if the empty token should be included in the score calculation.
        print_score (bool): A flag indicating whether to print the scores. Default is True.
        score_scale (int): A scaling factor for the scores. The scale value can be 100 or 1. Default is 100.
        eval_criteria (Union[Dict[str, Any], None]): A dictionary containing the evaluation criteria for each key.

    Returns:
        markdown_text (str): A markdown formatted text representing tables with the calculated metrics and scores
        at non_group-level, group-level, and document-level.
        entity_upscore (Dict[str, Dict[str, int]]): The first dictionary of this tuple where each key represents
        a specific classification category and corresponding value is a dictionary with calculated scores of precision,
        recall, micro_f1, macro_f1, and accuracy.
        total_upscore (Dict[str, Dict[str, int]]): The second dictionary of this tuple where the key represents the type
        of total score (total.non_group or total.group) and the corresponding value is a dictionary of calculated scores
        (precision, recall, micro_f1, macro_f1, accuracy).
    """

    if not (score_scale == 100 or score_scale == 1):
        raise ValueError("score_scale should be 100 or 1")

    # Update eval_criteria by adding criterion for keys which are not in eval_criteria
    eval_criteria = update_eval_criteria(eval_criteria, shortlist)

    confusion_matrix = defaultdict(lambda: defaultdict(int))

    document_exact_match = defaultdict(int)
    for gold_csv, pred_csv in zip(gold_csv_files, pred_csv_files):
        # sanity check
        assert os.path.basename(gold_csv) == os.path.basename(pred_csv)

        filename = os.path.basename(gold_csv)
        doc_name, extention = os.path.splitext(filename)

        # Load non_group_entities, group_entities
        gold_non_group_entities, gold_group_entities = read_non_group_and_group_entities(
            gold_csv, entity_type_delimiter, empty_token, include_empty_token_in_score_calculation, shortlist
        )
        pred_non_group_entities, pred_group_entities = read_non_group_and_group_entities(
            pred_csv, entity_type_delimiter, empty_token, include_empty_token_in_score_calculation, shortlist
        )

        # find group level entity confusion metric
        confusion_matrix, group_match = get_group_confusion_matrix(
            gold_group_entities,
            pred_group_entities,
            confusion_matrix,
            doc_name,
            empty_token,
            value_delimiter,
            grouping_strategy,
            split_merged_value,
            include_empty_token_in_score_calculation,
            eval_criteria=eval_criteria,
        )

        # find non_group level entity confusion metric
        confusion_matrix, non_group_match = get_non_group_confusion_matrix(
            gold_non_group_entities,
            pred_non_group_entities,
            confusion_matrix,
            empty_token,
            value_delimiter,
            split_merged_value,
            include_empty_token_in_score_calculation,
            eval_criteria=eval_criteria,
        )

        if all(group_match) and all(non_group_match):
            if len(group_match) + len(non_group_match) > 0:
                document_exact_match[doc_name] = 1
            # TODO: empty document calc
            else:
                print(f"empty document : {doc_name}")
                # document_exact_match[doc_name] = 0

        else:
            document_exact_match[doc_name] = 0

    entity_upscore, total_upscore = get_upscore(confusion_matrix, score_scale)
    for k, v in document_exact_match.items():
        entity_upscore[f"{k}.document"]["exact_match"] = v

    total_upscore["total.document"]["exact_match"] = sum(document_exact_match.values())
    total_upscore["total.document"]["num_of_document"] = len(document_exact_match)
    total_upscore["total.document"]["accuracy"] = (
        sum(document_exact_match.values()) / len(document_exact_match) * score_scale
    )

    if score_scale == 100:
        decimal_point = 2
    elif score_scale == 1:
        decimal_point = 4
    else:
        raise ValueError("score_scale should be 100 or 1")
    markdown_text = print_upscore(
        entity_upscore,
        total_upscore,
        print_score=print_score,
        decimal_point=decimal_point,
        eval_criteria=eval_criteria,
    )

    # Convert defaultdict to dict
    entity_upscore = defaultdict_to_dict(entity_upscore)
    total_upscore = defaultdict_to_dict(total_upscore)

    return markdown_text, entity_upscore, total_upscore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv_folder", type=str, required=True)
    parser.add_argument("--pred_csv_folder", type=str, required=True)
    parser.add_argument("--document_type", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--use_case", type=str, required=True)
    parser.add_argument("--ontology_folder", type=str, required=True)
    parser.add_argument("--eval_criteria_file", type=str, default=None)
    parser.add_argument("--empty_token", type=str, default="<empty>")
    parser.add_argument("--grouping_strategy", type=str, default="hungarian", choices=["max_em_score", "hungarian"])
    parser.add_argument("--include_empty_token_in_score_calculation", action="store_true")
    parser.add_argument("--split_merged_value", action="store_true")
    parser.add_argument("--entity_type_delimiter", type=str, default="\t")
    parser.add_argument("--value_delimiter", type=str, default="||")

    args = parser.parse_args()

    # Create the directory
    os.makedirs(os.path.join(args.output_folder, args.use_case, args.document_type), exist_ok=True)

    # Get csv file path
    gold_csv_files = sorted(glob.glob(os.path.join(args.gold_csv_folder, "*.csv")))
    pred_csv_files = sorted(glob.glob(os.path.join(args.pred_csv_folder, "*.csv")))
    # sanity check
    assert len(gold_csv_files) == len(pred_csv_files)

    # Get ontology file path
    ontology_files = sorted(glob.glob(os.path.join(args.ontology_folder, "*.yaml")))

    # Sanity check: ensure that some ontology files are found
    if len(ontology_files) == 0:
        raise ValueError(
            "No ontology files found. Please check the ontology_folder, use_case, and document_type arguments."
        )

    # Get eval_criteria
    if args.eval_criteria_file:
        with open(args.eval_criteria_file, "r") as f:
            eval_criteria = yaml.load(f, Loader=yaml.FullLoader)
    else:
        eval_criteria = {}

    for ontology_file_path in ontology_files:
        ontology_level, ontology_keys = get_ontology_keys(ontology_file_path)
        # calculation upscore
        result, _, _ = upscore(
            gold_csv_files,
            pred_csv_files,
            ontology_keys,
            args.empty_token,
            args.grouping_strategy,
            args.value_delimiter,
            args.entity_type_delimiter,
            args.split_merged_value,
            args.include_empty_token_in_score_calculation,
            eval_criteria,
        )
        output_file = os.path.join(args.output_folder, args.use_case, args.document_type, f"{ontology_level}.md")
        with open(output_file, "w") as file:
            file.write(result)