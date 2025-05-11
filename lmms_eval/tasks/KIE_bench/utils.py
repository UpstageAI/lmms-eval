import os
import json
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import uuid
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

def KIE_bench_doc_to_visual_VLM_LLM_IE(item):
    """
    For vllm serving, save the temporary images and return the image paths to the item
    """
    image_path = item["file_path"]
    if image_path.endswith(".pdf"):
        images = convert_from_path(image_path)
        os.makedirs("temp_images", exist_ok=True)
        image_paths = []
        for image in images:
            image_path = f"temp_images/{uuid.uuid4()}.png"
            image.save(image_path)
            image_paths.append(image_path)
    else:
        image = Image.open(image_path).convert("RGB")
        image_path = f"temp_images/{uuid.uuid4()}.png"
        image.save(image_path)
        image_paths = [image_path]

    return image_paths



def KIE_bench_doc_to_text(item, lmms_eval_specific_kwargs=None):
    schema_path = item["schema_path"]
    with open(schema_path, "r") as f:
        schema = json.load(f)

    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{schema}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"

    return question

def KIE_bench_doc_to_text_VLM_LLM_IE(item, lmms_eval_specific_kwargs=None):
    schema_path = item["schema_path"]
    with open(schema_path, "r") as f:
        schema = json.load(f)

    schema_text = json.dumps(schema) # json.dumps로 변환하여 문자열로 저장
    if "vlm_user_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["vlm_user_prompt"] != "":
        vlm_user_prompt = lmms_eval_specific_kwargs["vlm_user_prompt"]
    else:
        vlm_user_prompt = ""
    if "llm_pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["llm_pre_prompt"] != "":
        llm_pre_prompt = lmms_eval_specific_kwargs["llm_pre_prompt"]
    else:
        llm_pre_prompt = ""
    if "llm_post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["llm_post_prompt"] != "":
        llm_post_prompt = lmms_eval_specific_kwargs["llm_post_prompt"]
    else:
        llm_post_prompt = ""
    if "DocEV_DP_user_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["DocEV_DP_user_prompt"] != "":
        DocEV_DP_user_prompt = lmms_eval_specific_kwargs["DocEV_DP_user_prompt"]
    else:
        DocEV_DP_user_prompt = ""

    context_dict = {
        "schema": schema_text,
        "vlm_user_prompt": vlm_user_prompt,
        "llm_pre_prompt": llm_pre_prompt,
        "llm_post_prompt": llm_post_prompt,
        "DocEV_DP_user_prompt": DocEV_DP_user_prompt
    }

    return json.dumps(context_dict) # string타입만 가능하므로 json.dumps로 우회

def _read_DP_HTML(DP_html_path):
    DP_html_dir = os.path.dirname(DP_html_path)
    DP_html_list = [os.path.join(DP_html_dir, p) for p in os.listdir(DP_html_dir) if p.startswith(os.path.basename(DP_html_path).split('.')[0])]
    DP_html_list = sorted(DP_html_list)
    html_text = ""
    for DP_html in DP_html_list:
        with open(DP_html, "r") as f:
            html_text += f.read() + "\n"
    return html_text

def KIE_bench_doc_to_text_DP_LLM_IE(item, lmms_eval_specific_kwargs=None):
    schema_path = item["schema_path"]
    with open(schema_path, "r") as f:
        schema = json.load(f)

    schema_text = json.dumps(schema) # json.dumps로 변환하여 문자열로 저장
    vlm_user_prompt = "NO VLM INFERENCE"

    if "llm_pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["llm_pre_prompt"] != "":
        llm_pre_prompt = lmms_eval_specific_kwargs["llm_pre_prompt"]
    else:
        llm_pre_prompt = ""
    if "llm_post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["llm_post_prompt"] != "":
        llm_post_prompt = lmms_eval_specific_kwargs["llm_post_prompt"]
    else:
        llm_post_prompt = ""

    # read DP and set the prompt
    DP_html_path = item["file_path"].replace("/v3.1/","/v3.1/dp_result/html/") # TODO: v3.1은 bench version update되면 수정해야함. 하드코딩된 부분을 추후 외부에서 입력하도록 수정 필요.
    html_text = _read_DP_HTML(DP_html_path)
    with open("./temp_html.txt", "w") as f:
        f.write(DP_html_path)
        f.write(html_text)

    context_dict = {
        "schema": schema_text,
        "vlm_user_prompt": vlm_user_prompt,
        "llm_pre_prompt": llm_pre_prompt,
        "llm_post_prompt": llm_post_prompt,
        "vlm_output": html_text
    }

    return json.dumps(context_dict) # string타입만 가능하므로 json.dumps로 우회


def KIE_bench_doc_to_target(item):
    # UpScore 계산에는 사용하지 않습니다.
    # lmms eval 에 자체적으로 구현되어있는 metric을 사용하는 경우에 필요하며,
    # 만약 KIE bench 데이터셋을 이용해서 다른 metric을 사용하고자 하는 경우에 구현이 필요합니다.
    return ""

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
    try:
        # If the prediction is already in json format, use json.loads (structured output of VLM_LLM_IE)
        pred_result = json.loads(pred)
    except Exception as e:
        # If the prediction is not in json format, use parse_pred_ans
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