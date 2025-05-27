"""
ref1: https://github.com/UpstageAI/docev-data-engine/blob/main/dp2ufx/utils.py
ref2: https://github.com/UpstageAI/docev-data-engine/blob/main/dp2ufx/ufx_for_pretraining.py

Usage:
```
python KIE_bench_dp2html.py <dp json files dir> <dp html files dir>
```

이 코드는 다음과 같은 작업을 수행합니다.
1. 주어진 디렉토리에서 json 파일들을 찾아 목록으로 반환합니다.
2. 각 json 파일을 파싱하여 html 파일로 변환합니다.
3. 변환된 html 파일을 주어진 디렉토리에 저장합니다.

reference 코드에서 변경된 부분:
- ufx format 대신 html 파일 자체를 저장합니다.
- 이미지 사이즈 정보를 표기하지 않습니다.
- 데이터 필터링 과정 제거
"""

import json
import os
from typing import List

from bs4 import BeautifulSoup, Tag
from p_tqdm import p_map
import argparse
import os
import sys
from functools import partial

from p_tqdm import p_map

sys.path.append(os.path.dirname(os.path.abspath(__file__)))




def _find_json_files_in_subdir(args):
    input_dir, subdir = args
    json_files = []
    subdir_path = os.path.join(input_dir, subdir)

    for root, _, files in os.walk(subdir_path):
        for file in files:
            if file.endswith(".json"):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, input_dir)
                json_files.append(rel_path)

    return json_files


def find_files(input_dir: str) -> List[str]:
    """
    input_dir의 하위 디렉토리들을 재귀적으로 탐색하여 json 파일들을 찾아내고, 파일 리스트를 반환합니다.
    """
    # 최상위 하위 디렉토리 목록 생성 (최상위 디렉토리에 있는 파일은 무시)
    subdirs = [name for name in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, name))]

    # 빠른 탐색을 위해 p_map을 사용하여 각 하위 디렉토리에서 병렬로 검색
    results = p_map(_find_json_files_in_subdir, [(input_dir, subdir) for subdir in subdirs])

    # 리스트 평탄화
    json_files = [item for sublist in results for item in sublist]

    return json_files



def _remove_alt_tag(soup):
    """
    HTML 내 모든 태그에서 alt 속성을 제거합니다.
    """
    for tag in soup.find_all(True):
        if tag.has_attr("alt"):
            del tag["alt"]
    return soup

def _remove_font_size_style_value(soup):
    """
    모든 태그의 style 어트리뷰트에서 font-size 속성을 제거하고,
    style이 비면 style 어트리뷰트 자체를 삭제합니다.
    """
    for tag in soup.find_all(True):
        style = tag.get("style", None)
        if style:
            # font-size 속성만 제거
            styles = [s.strip() for s in style.split(';') if s.strip()]
            filtered = [s for s in styles if not s.strip().startswith("font-size")]
            if filtered:
                tag["style"] = "; ".join(filtered)
            else:
                del tag["style"]
    return soup

# 테이블 ID에 매핑된 표 전사 HTML을 찾아서 원본 HTML에 삽입하는 함수입니다.
# json_data 내 'paragraphs'에서 label이 'Table'인 항목을 골라 전사 내용을 매핑합니다.
def _insert_table_transcription(data, soup):
    """
    JSON 데이터에서 'Table'로 라벨된 전사(transcription) HTML을 추출하여,
    html_string 내 대응하는 <table id="..."> 요소에 <thead>와 <tbody>를
    올바른 순서(ID 기준)에 삽입한 뒤 전체 HTML을 문자열로 반환합니다.

    Args:
        json_data (dict): 'paragraphs' 키로 매핑된 ID별 사전(dict)을 포함합니다.
                          각 사전에는 'label'과 'transcription'이 있어야 합니다.
        html_string (str): 테이블 전사를 삽입해야 할 원본 HTML 문자열입니다.

    Returns:
        str: 전사 내용이 삽입된 전체 HTML 문자열.
    """
    # 1) 'Table' 레이블을 가진 항목만 골라 {id: transcription_html} 매핑 생성
    table_transcriptions = {
        pid: pdata['transcription'].strip()
        for pid, pdata in data.get('paragraphs', {}).items()
        if pdata.get('label', '').strip().lower() == 'table' and pdata.get('transcription')
    }

    # 2) 모든 <table id="..."> 요소 순회
    for table_tag in soup.find_all('table', id=True):
        tid = table_tag['id']  # 테이블 ID
        if tid not in table_transcriptions:
            continue  # 전사 데이터가 없으면 건너뜁니다.

        # 3) 전사 문자열을 파싱하여 조각(fragment) 생성
        frag = BeautifulSoup(table_transcriptions[tid], 'html.parser')

        # 4) 기존 <thead>, <tbody> 및 직계 하위의 <br> 태그 제거
        for node in table_tag.find_all(['thead', 'tbody', 'br'], recursive=False):
            node.decompose()

        # 5) ID 기반의 순서 삽입 인덱스를 계산하기 위해 숫자 변환 시도
        try:
            table_idx = int(tid)
        except ValueError:
            table_idx = None

        insertion_index = None
        if table_idx is not None:
            # 6-1) 직계 자식 중 ID를 가진 태그들의 (위치, 숫자ID) 목록 수집
            id_children = []
            for idx, child in enumerate(table_tag.contents):
                if isinstance(child, Tag) and child.has_attr('id'):
                    try:
                        cid = int(child['id'])
                        id_children.append((idx, cid))
                    except ValueError:
                        pass

            # 6-2) 테이블 ID보다 큰 자식이 있으면 해당 위치 이전에 삽입
            greater = [idx for idx, cid in id_children if cid > table_idx]
            if greater:
                insertion_index = min(greater)
            else:
                # 6-3) 그렇지 않으면 작은 자식 중 가장 뒤(ID 큰 순) 이후에 삽입
                lesser = [idx for idx, cid in id_children if cid < table_idx]
                if lesser:
                    insertion_index = max(lesser) + 1

        # 6) 전사 조각 노드를 원본 테이블에 삽입 또는 추가
        nodes = list(frag.contents)
        if insertion_index is None:
            # 계산된 인덱스 없으면 맨 끝에 추가
            for node in nodes:
                table_tag.append(node)
        else:
            # 역순으로 삽입하여 원래 순서 유지
            for node in reversed(nodes):
                table_tag.insert(insertion_index, node)

    # 7) 수정된 BeautifulSoup 객체 반환
    return soup

def _serialize_with_order(soup: BeautifulSoup) -> str:
    """
    최상위 태그에 한해서 id, data-x1, data-y1, data-x2, data-y2 순으로
    attrs를 직접 문자열로 조합해 내보냅니다.
    나머지 부분은 formatter=None 옵션을 사용해 원본 HTML/text를 그대로 반환합니다.
    """
    parts = []
    for element in soup.contents:
        # 1) 최상위 태그(id & bbox 속성 갖고 있는) 특별 처리
        if isinstance(element, Tag) and element.has_attr("id") \
           and all(k in element.attrs for k in ("data-x1", "data-y1", "data-x2", "data-y2")):

            ordered_keys = ["id", "data-x1", "data-y1", "data-x2", "data-y2"]
            attr_strs = []
            # 지정된 순서대로 추가
            for k in ordered_keys:
                if k in element.attrs:
                    attr_strs.append(f'{k}="{element.attrs[k]}"')
            # 나머지 속성들 추가
            for k, v in element.attrs.items():
                if k not in ordered_keys:
                    attr_strs.append(f'{k}="{v}"')

            # inner HTML 추출 (이스케이프 없이 원본 그대로)
            inner_html = element.encode_contents(formatter=None).decode()

            parts.append(
                f'<{element.name} ' + " ".join(attr_strs) +
                f'>{inner_html}</{element.name}>'
            )
        else:
            # 2) Tag 인 경우: decode(formatter=None) → raw HTML
            if isinstance(element, Tag):
                parts.append(element.decode(formatter=None))
            # NavigableString 등 단순 문자열인 경우: str(element) 그대로
            else:
                parts.append(str(element))

    return "".join(parts)

def _get_enriched_html(data):
    html_str = data.get("html", "")
    if html_str:
        try:
            soup = BeautifulSoup(html_str, "html.parser")
        except Exception as e:
            # 파싱 실패 샘플은 빈 문자열로 처리
            return ""
    else:
        return ""


    # 2. alt 어트리뷰트 모두 제거
    soup = _remove_alt_tag(soup)

    # 3. style에서 font-size 제거
    soup = _remove_font_size_style_value(soup)

    # 4. 테이블 트랜스크립션 삽입
    soup = _insert_table_transcription(data, soup)

    return _serialize_with_order(soup)


# 주요 변경 사항:
# - 이미지 사이즈 정보를 표기하지 않습니다.
# - 데이터 필터링 과정 제거
def parse(args, input_dir):
    idx, rel_path = args
    abs_path = os.path.join(input_dir, rel_path)

    # json 파일 파싱
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise Exception(f"Error parsing json file: {abs_path}")

    # 명세에 따라 필드 추출
    text = _get_enriched_html(data)

    return {
        "id": str(idx),
        "rel_path": rel_path,
        "text": text,
    }


# ref 코드와 비교하여 주요 변경 사항:
# 1. ufx format 대신 html 파일 자체를 저장합니다.
# 2. 데이터 필터링 과정 제거
# 3. 데이터 저장 과정 제거
def convert(args):
    """
    JSON 데이터 파일을 읽어들여, html 파일 자체를 저장합니다.
    """
    print(f"1. JSON 파일 리스트 작성: {args.input_dp_dir}...")
    rel_paths = find_files(args.input_dp_dir)
    print(f"- 파일 갯수: {len(rel_paths)}")

    print("2. 데이터 추출 전처리...")
    parse_func = partial(parse, input_dir=args.input_dp_dir)
    params = [(idx, rel_path) for idx, rel_path in enumerate(rel_paths)]
    results = p_map(parse_func, params, num_cpus=args.num_cpus)

    print("3. html 파일 저장...")
    for result in results:
        if result["text"]:
            html_path = os.path.join(args.output_dir, result['rel_path'].replace(".json", ".html"))
            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
        else:
            # Empty result in json file
            print(f"Error: {os.path.join(args.input_dp_dir, result['rel_path'])} has no text")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("")
            continue

    print(f"- 저장 완료: {len(results)} 샘플")



def main():
    parser = argparse.ArgumentParser(description="UFX Pretraining Data Converter")
    parser.add_argument("input_dp_dir", type=str, help="Input directory to search for DP output JSON files")
    parser.add_argument("output_dir", type=str, help="Output directory to save the result")
    parser.add_argument("--num_cpus", type=int, default=50, help="Number of CPUs to use")
    args = parser.parse_args()


    convert(args)


if __name__ == "__main__":
    main()