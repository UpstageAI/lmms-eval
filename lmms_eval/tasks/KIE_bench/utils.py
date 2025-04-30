import os
import json
import numpy as np
from PIL import Image
from pdf2image import convert_from_path


dir_name = os.path.dirname(os.path.abspath(__file__))

def KIE_bench_doc_to_visual(item):
    """
    item example: {'test_id': 'IEX-101', 'bench_type': 'RealKIE', 'doc_type': 'charities', 
                    'doc_name': '00c5f02bb4d35366915b1d44fa610452', 
                    'file_path': 'data/v3_0/realkie_big/charities/files/00c5f02bb4d35366915b1d44fa610452.pdf', 
                    'schema_path': 'data/v3_0/realkie_big/charities/schema/00c5f02bb4d35366915b1d44fa610452.json', 
                    'gold_path': 'data/v3_0/realkie_big/charities/gold_result/00c5f02bb4d35366915b1d44fa610452.json', 
                    'num_pages': 12, 'num_entities': 19}
    """
    image_path = item["file_path"]
    if image_path.endswith(".pdf"):
        images = convert_from_path(image_path)
    else:
        image = Image.open(image_path).convert("RGB")
        return [image]
    return images

def KIE_bench_doc_to_text(item, lmms_eval_specific_kwargs=None):
    # TODO: apply response_format
    schema_path = item["schema_path"]
    with open(schema_path, "r") as f:
        schema = json.load(f)

    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{schema}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"

    return question



def KIE_bench_doc_to_target(item):
    gold_path = item["gold_path"]
    with open(gold_path, "r") as f:
        gold = json.load(f)
    return str(gold)

def parse_pred_ans(pred_ans):
    """
    Args:
        pred_ans: a string of the predicted answer
    Returns:
        a string of the predicted answer
    """
    # TODO: implement prediction parsing algorithm
    return pred_ans

def KIE_bench_process_results(item, results):
    """
    Args:
        item: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """

    pred = results[0]
    pred_ans = parse_pred_ans(pred)

    # TODO: implement gt parsing algorithm (See "info-extractor-engine" repo)
    gt_path = item["gold_path"]
    with open(gt_path, "r") as f:
        gt = json.load(f)
    gt_ans = str(gt)

    # TODO: implement UpScore (See "info-extractor-engine" repo)
    score = np.random.rand()
    test_id = item["test_id"]
    bench_type = item["bench_type"] 
    doc_type = item["doc_type"]
    key_name = "UpScore"

    return {key_name: {"test_id": test_id, "bench_type": bench_type, "doc_type": doc_type, "score": score}}

def KIE_bench_aggregate_results(results):
    """
    Args:
        results: a list of dictionaries returned by process_results
    Returns:
        A score
    """
    score_list = []
    for result in results:
        score_list.append(result["score"])
    score = np.mean(score_list)
    return score