import os
import json
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path
import glob
from typing import Dict, Any, Union
import pandas as pd
from lmms_eval.tasks.KIE_bench.UpScore.upscore import upscore
import re

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

def parse_pred_ans(pred, schema):
    """
    Args:
        pred: a string of the predicted answer
        schema: a dictionary of the schema
    Returns:
        a string of the predicted answer
    """
    # 1) grab everything between the outer braces
    m = re.search(r"\{(.+)\}", pred, flags=re.DOTALL)
    if not m:
        # TODO: handle error case where no dictionary found in prediction result
        print("\n\n")
        print("WARNING: no dictionary found in prediction result")
        print("\n =========== Prediction =========== \n")
        print(pred)
        print("\n =========== Schema =========== \n")
        print(schema)
        print("\n =============================== \n")
        return {}
    body = m.group(1)

    # 2) a regex that matches either
    #      'key': [val1, val2, …]
    #    or
    #      'key': [someValue]
    #  we capture the list-contents in "list" or the lone value in "val"
    pattern = re.compile(
        r"'(?P<key>[^']+)'\s*:\s*"
        r"(?:\[(?P<list>[^\]]*)\]"
        r"|(?P<val>[^,\}]+))"
    )

    out = {}
    for m in pattern.finditer(body):
        key = m.group("key")
        if m.group("list") is not None:
            # split on commas, strip whitespace and surrounding quotes
            items = [x.strip().strip("'\"")
                     for x in m.group("list").split(",")
                     if x.strip()]
            out[key] = items
        else:
            v = m.group("val").strip().strip("'\"")
            out[key] = [v]

    return out

def KIE_bench_process_results(item, results):
    """
    Args:
        item: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    # Parse prediction based on schema
    pred = results[0]
    schema_path = item["schema_path"]
    with open(schema_path, "r") as f:
        schema = json.load(f)
    pred_result = parse_pred_ans(pred, schema)

    # Parse ground truth
    gt_path = item["gold_path"]
    with open(gt_path, "r") as f:
        gold_result = json.load(f)

    score = eval_upscore(f"./UpScore_results/{item['test_id']}/", schema, gold_result, pred_result, f"{item['test_id']}.csv")

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

def eval_upscore(
    output_path: str, schema: Dict[str, Any], gold_result: Dict[str, Any], pred_result: Dict[str, Any], file_name: str
) -> None:
    entity_type_delimiter = "\t"
    value_delimiter = " || "
    empty_token = "<empty>"
    split_merged_value = True

    gold_csv_files, pred_csv_files = convert_to_csv_files(
        schema=schema,
        output_dir=output_path,
        gold_result=gold_result,
        pred_result=pred_result,
        entity_type_delimiter=entity_type_delimiter,
        value_delimiter=value_delimiter,
        empty_token=empty_token,
        file_name=file_name,
    )

    eval_criteria = {}
    shortlist = []
    for k, v in schema["properties"].items():
        if (v["type"] == "array") and (v["items"]["type"] == "object"):
            for sub_k in v["items"]["properties"].keys():
                shortlist.append(sub_k)
                eval_criteria[sub_k] = {}
                eval_criteria[sub_k]["remove_chars"] = [" ", "\\"]
                eval_criteria[sub_k]["match"] = "exact_match"
        else:
            shortlist.append(k)
            eval_criteria[k] = {}
            eval_criteria[k]["remove_chars"] = [" ", ".", ",", "$", "£", "\\"]
            eval_criteria[k]["match"] = "exact_match"

    result, entity_level_upscore, total_upscore = upscore(
        gold_csv_files=gold_csv_files,
        pred_csv_files=pred_csv_files,
        shortlist=shortlist,
        empty_token=empty_token,
        grouping_strategy="hungarian",
        value_delimiter=value_delimiter,
        entity_type_delimiter=entity_type_delimiter,
        split_merged_value=split_merged_value,
        include_empty_token_in_score_calculation=True,
        print_score=False,
        eval_criteria=eval_criteria,
    )

    upscore_result_file = os.path.join(output_path, "result.md")
    with open(upscore_result_file, "w", encoding="utf-8") as f:
        f.write(result)

    entitiy_level_upscore_result_file = os.path.join(output_path, "entity_level_KIEval.json")
    with open(entitiy_level_upscore_result_file, "w", encoding="utf-8") as f:
        json.dump(entity_level_upscore, f, indent=4, ensure_ascii=False)

    total_upscore_result_file = os.path.join(output_path, "total_KIEval.json")
    with open(total_upscore_result_file, "w", encoding="utf-8") as f:
        json.dump(total_upscore, f, indent=4, ensure_ascii=False)

    score = total_upscore["total.upscore"]["upscore"]
    return score



def postprocess(x: Union[str, int]) -> Union[str, int]:
    """
    Args:
        x: a string or an integer
    Returns:
        a processed string or an integer

    This code is from "UpstageAI/info-extractor-engine" repo
    """
    processed_x = x.lower() if isinstance(x, str) else x
    if not isinstance(processed_x, str):
        processed_x = processed_x[0].replace("\n", " ")
    else:
        processed_x = processed_x.replace("\n", " ")
    return processed_x


def write_csv(
    output_path: str, grouped_dict: Dict[str, Any], non_grouped_dict: Dict[str, Any], entity_type_delimiter: str = ","
) -> None:
    """
    Args:
        output_path: a string of the output path
        grouped_dict: a dictionary of the grouped result
        non_grouped_dict: a dictionary of the non-grouped result
        entity_type_delimiter: a string of the entity type delimiter

    This code is from "UpstageAI/info-extractor-engine" repo
    """
    with open(output_path, "w") as f:
        if len(grouped_dict) > 0:
            for num, group in enumerate(grouped_dict.values()):
                f.write(f"{num}\n")
                grouped_df = pd.DataFrame(group)
                grouped_df = grouped_df.applymap(lambda x: postprocess(x))
                grouped_df.to_csv(f, index=False, sep=entity_type_delimiter)
                f.write("\n")
        else:
            f.write("\n\n")
        f.write("-1\n")
        non_grouped_df = pd.DataFrame.from_dict(non_grouped_dict, orient="index", columns=["-1"]).reset_index()
        non_grouped_df["-1"] = non_grouped_df["-1"].apply(lambda x: postprocess(x))
        non_grouped_df.to_csv(f, index=False, sep=entity_type_delimiter, header=False)


def convert_to_csv_files(
    schema: Dict[str, Any],
    output_dir: str,
    gold_result: Dict[str, Any] = None,
    pred_result: Dict[str, Any] = None,
    empty_token: str = "",
    value_delimiter: str = " || ",
    entity_type_delimiter: str = ",",
    file_name: str = "test.csv",
) -> None:
    """
    Args:
        schema: a dictionary of the schema
        output_dir: a string of the output directory
        gold_result: a dictionary of the gold result
        pred_result: a dictionary of the predicted result
        empty_token: a string of the empty token
        value_delimiter: a string of the value delimiter

    This code is from "UpstageAI/info-extractor-engine" repo
    """
    gold_non_grouped_dict = {}
    pred_non_grouped_dict = {}
    gold_grouped_dict = {}
    pred_grouped_dict = {}
    for key, value in schema["properties"].items():
        if (value["type"] == "array") and (value["items"]["type"] == "object"):
            if key in gold_result.keys() and gold_result[key]:
                gold_grouped_dict[key] = [
                    {
                        sub_key: item.get(sub_key, empty_token) if item.get(sub_key, empty_token) else empty_token
                        for sub_key in value["items"]["properties"].keys()
                    }
                    for item in gold_result[key]
                ]
            if key in pred_result.keys() and pred_result[key]:
                pred_grouped_dict[key] = [
                    {
                        sub_key: item.get(sub_key, empty_token) if item.get(sub_key, empty_token) else empty_token
                        for sub_key in value["items"]["properties"].keys()
                    }
                    for item in pred_result[key]
                ]
        else:
            # non_group
            if key in gold_result.keys() and gold_result[key]:
                if isinstance(gold_result[key], str):
                    gold_result[key] = [gold_result[key]]
                if isinstance(gold_result[key], list) and all(not v for v in gold_result[key]):
                    continue
                gold_non_grouped_dict[key] = value_delimiter.join(gold_result[key])
            if key in pred_result.keys() and pred_result[key]:
                if isinstance(pred_result[key], str):
                    pred_result[key] = [pred_result[key]]
                if isinstance(pred_result[key], list) and all(not v for v in pred_result[key]):
                    continue
                pred_non_grouped_dict[key] = value_delimiter.join(pred_result[key])

    os.makedirs(Path(output_dir) / "gold_csv", exist_ok=True)
    os.makedirs(Path(output_dir) / "pred_csv", exist_ok=True)

    write_csv(
        Path(output_dir) / "gold_csv" / f"{file_name}.csv",
        gold_grouped_dict,
        gold_non_grouped_dict,
        entity_type_delimiter,
    )
    write_csv(
        Path(output_dir) / "pred_csv" / f"{file_name}.csv",
        pred_grouped_dict,
        pred_non_grouped_dict,
        entity_type_delimiter,
    )

    gold_csv_files = sorted(glob.glob(os.path.join(output_dir, "gold_csv", "*.csv")))
    pred_csv_files = sorted(glob.glob(os.path.join(output_dir, "pred_csv", "*.csv")))
    assert len(gold_csv_files) == len(pred_csv_files)

    return gold_csv_files, pred_csv_files