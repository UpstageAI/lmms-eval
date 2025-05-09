"""
VLM-LLM-IE에서 캡셔닝에 따른 성능 향상을 확인하기 위해 DP 결과를 저장합니다.
기존 DP pipeline 코드에 pdf file handling 기능을 추가합니다.

ref: https://github.com/UpstageAI/docev-data-engine/tree/sehwan/feat/dp-pipeline/document-parse
"""
# ref: https://github.com/UpstageAI/dp-pipeline/blob/main/dp_pipeline/inference.py

import argparse
import base64
import json
import os
import multiprocessing
from typing import Dict, List, Union
import numpy as np
import cv2
from hydra.utils import instantiate
import torch
from tqdm import tqdm
import yaml
import traceback
from ocr_utils.file_io import find_images, find_files
from glob import glob
from p_tqdm import p_map
from functools import partial
import gc
import signal
from pdf2image import convert_from_path



MAX_SIZE = 2048 * 2048

_PIPELINE_MODULE_PREFIX = "dp_pipeline.pipelines."
_PLACEHOLDER_MODULE_PREFIX = "dp_pipeline.placeholders."

_CFG_INPUT_KEY = "inputs"
_CFG_OUTPUT_KEY = "outputs"
_CFG_PLACEHOLDER_KEY = "placeholders"
_CFG_PIPELINE_KEY = "pipelines"


def find_pdfs(root, recursive: bool = False, extensions: Union[str, List[str], None] = None):
    """주어진 root directory에서 pdf 파일을 모두 찾는다.

    Args:
        root: 파일을 찾을 디렉토리
        recursive: True일 경우 하위 디렉토리까지 검색에 포함한다.
        extensions: 찾을 이미지의 파일 확장자들. 기본적으로 대상이 되는 확장자 목록은 다음과 같다:
            {'.pdf'}.
    """
    pdf_extensions = extensions or {'.pdf'}
    return find_files(root, recursive=recursive, extensions=pdf_extensions)


def dumper(obj):
    try:
        return obj.toJSON()
    except AttributeError:
        return obj.__dict__


class DPPipeline:
    def __init__(
        self,
        version: str = "1.3.1_base",
    ):
        self.base_path = "/app/docfm/checkpoints/dp"
        self.version = version
        self.load()

    def reload(self):
        self.ready = False
        torch.cuda.empty_cache()  # Clear GPU memory cache
        del self.pipelines
        gc.collect()  # Run garbage collection
        torch.cuda.reset_peak_memory_stats()  # Reset CUDA memory statistics
        # Build pipelines
        self.pipelines = {}
        for name, options in self.config[_CFG_PIPELINE_KEY].items():
            pipeline = instantiate(options, name=name)
            self.pipelines[name] = pipeline
        self.ready = True

    def _load_config(self):
        config_dir = f"{self.base_path}/v{self.version}/OP_CONFIG_DIR"
        config_path = os.path.join(config_dir, "config.yaml")

        assert os.path.isfile(config_path), "config is not in base_path"
        os.environ["OP_CONFIG_DIR"] = config_dir
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # set model_dir
        model_dir = f"{self.base_path}/v{self.version}/OP_MODEL_DIR"
        os.environ["OP_MODEL_DIR"] = model_dir
        return config

    def _build_placeholders(self):
        placeholders = {}
        for name, options in self.config[_CFG_PLACEHOLDER_KEY].items():
            options["_target_"] = _PLACEHOLDER_MODULE_PREFIX + options["_target_"]
            placeholder = instantiate(options, name=name)
            placeholders[name] = placeholder
        return placeholders

    def _build_pipelines(self):
        pipelines = {}
        for name, options in self.config[_CFG_PIPELINE_KEY].items():
            options["_target_"] = _PIPELINE_MODULE_PREFIX + options["_target_"]
            pipeline = instantiate(options, name=name)
            pipelines[name] = pipeline
        return pipelines

    def load(self):
        # load configure
        self.config = self._load_config()

        # Build placeholders
        self.placeholders = self._build_placeholders()

        # Build pipelines
        self.pipelines = self._build_pipelines()

        self.ready = True

    def predict(self, request: Dict) -> Dict:
        for input_key, placeholder_key in self.config[_CFG_INPUT_KEY].items():
            input_obj = request.get(input_key)
            self.placeholders[placeholder_key].set_data(input_obj)

            if placeholder_key == "image" and "image_tensor" in self.placeholders:
                self.placeholders["image_tensor"].set_data(
                    torch.from_numpy(self.placeholders[placeholder_key].data).permute(2,0,1).to('cuda').to(torch.float32)
                )

        with torch.inference_mode():
            for name, pipeline in self.pipelines.items():
                if request.get("ufo") != None and ("detector" in name or "recognizer" in name):
                    continue
                pipeline.forward(self.placeholders)

        outputs = {}
        for output_key in self.config[_CFG_OUTPUT_KEY]:
            output = self.placeholders[output_key].get_data()
            outputs.update(output)
        return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="1.3.1_base")
    # parser.add_argument("--input_dir", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="/app/docfm/datasets/document_datasets/mint-1t/pdf/image_norm/CC-MAIN-2023-23-shard-0")
    parser.add_argument("--output_dir", type=str, default="/app/docfm/datasets/document_datasets/mint-1t/ufso/v1.3.1/CC-MAIN-2023-23-shard-0")
    parser.add_argument("--num_cpus", type=int, help="Number of CPUs to use for parallel processing")
    parser.add_argument("--num_chunk", type=int, default=8, help="Number of chunks to split the input directory")
    parser.add_argument("--start_chunk_index", type=int, default=1, help="Number of start chunk index")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use for parallel inference")
    parser.add_argument("--usePatchify", action=argparse.BooleanOptionalAction)
    parser.add_argument("--withCharBox", action=argparse.BooleanOptionalAction)
    parser.add_argument("--withConfidenceScore", action=argparse.BooleanOptionalAction)
    parser.add_argument("--adjustBoxAngles", action=argparse.BooleanOptionalAction)
    parser.add_argument("--omitRelations", action=argparse.BooleanOptionalAction)
    parser.add_argument("--withDOS", action=argparse.BooleanOptionalAction)
    parser.add_argument("--omitSerializer", action=argparse.BooleanOptionalAction)
    parser.add_argument("--languageTag", type=str)
    parser.add_argument("--confidenceThres", type=float, default=0.4)
    parser.add_argument("--alignOrientation", action=argparse.BooleanOptionalAction)
    parser.add_argument("--useOcrTextLine", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    assert args.num_chunk > 0, "num_chunk must be greater than 0"
    assert args.start_chunk_index >= 0 and args.start_chunk_index < args.num_chunk, "start_chunk_index must be in range of num_chunk"

    args.num_cpus = args.num_cpus if args.num_cpus else multiprocessing.cpu_count() // args.num_chunk

    json_input = {}
    for key, value in vars(args).items():
        if key in ["version", "image_dir", "ufo_path", "output_dir", "num_processes"]:
            continue
        json_input[key] = value

    return args, json_input


def find_leaf_directories(directory, num_chunks, start_chunk_index):
    """Find all leaf directories (directories with no subdirectories) using os.walk.
    """
    root_dir_length = len(directory.split(os.sep))
    leaf_dirs = []
    index = 0
    for root, dirs, _ in os.walk(directory, followlinks=True):
        # If there are no subdirectories, this is a leaf
        if not dirs:
            if index % num_chunks == start_chunk_index:
                parts = root.split(os.sep)
                leaf_dirs.append(os.path.join(*parts[root_dir_length - 1:]))
            index += 1
    return leaf_dirs


def process_and_encode(image_path):
    with open(image_path, "rb") as f:
        image_binary = f.read()
    nparr = np.frombuffer(image_binary, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = image_np.shape[:2]

    if h * w > MAX_SIZE * MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode("utf-8")


def init_worker(version):
    """Process initialization function. Initialize GPU memory settings and dp_pipeline instance for each process."""
    global model_pipeline
    model_pipeline = DPPipeline(version=version)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Prediction timed out")


def process_image(args_tuple):
    """Function to process each image. Uses the global dp_pipeline instance."""
    global model_pipeline
    image_path, leaf_dir, args, json_input = args_tuple
    
    save_root_dir = os.path.join(args.output_dir, leaf_dir)
    os.makedirs(save_root_dir, exist_ok=True)
    
    save_path = os.path.join(save_root_dir, os.path.basename(image_path) + ".json")
    if os.path.exists(save_path):
        return image_path, True
    
    try:
        # Process image and predict
        current_json_input = json_input.copy()
        current_json_input["image"] = process_and_encode(image_path)
        
        # Set timeout for prediction (30 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            output = model_pipeline.predict(current_json_input)
            # Disable alarm
            signal.alarm(0)
        except TimeoutError:
            # Prediction is hanging, reload model
            # signal.alarm(0)  # Disable alarm
            print(f"Prediction hanging for image: {image_path}, reloading model...")
            raise RuntimeError("Prediction timed out")
        
        # Save results
        json.dump(output, open(save_path, "w"), default=dumper, ensure_ascii=False, indent=2)
        return image_path, True
    except Exception as e:
        signal.alarm(0)  # Ensure alarm is disabled
        print(f"Error processing image: {image_path}")
        traceback.print_exception(type(e), e, e.__traceback__)
        if type(e) == RuntimeError:
            model_pipeline.reload()
        return image_path, False


def process_batch(image_batch, leaf_dir, args, json_input):
    """Function to process image batches"""
    # Prepare arguments for each image
    args_tuples = [(image_path, leaf_dir, args, json_input) for image_path in image_batch]
    
    # Process results
    results = []
    for args_tuple in args_tuples:
        result = process_image(args_tuple)
        results.append(result)
    
    return results


def main(args, json_input):
    # Settings for parallel processing
    num_processes = min(args.num_processes, multiprocessing.cpu_count())
    print(f"Using {num_processes} processes for parallel inference")
    
    # Find all top-level subdirectories
    sub_dirs = [d for d in glob(os.path.join(args.input_dir, '*')) if os.path.isdir(d)]

    # 1. Find leaf directories for each subdirectory in parallel
    leaf_dirs_lists = p_map(
        find_leaf_directories,
        sub_dirs,
        [args.num_chunk] * len(sub_dirs),
        [args.start_chunk_index] * len(sub_dirs),
        num_cpus=args.num_cpus,
        desc="Finding leaf directories"
    )

    # Create multiprocessing pool and initialize
    # Each process has its own dp_pipeline instance
    pool = multiprocessing.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(args.version,)
    )
    
    try:
        for leaf_dirs in leaf_dirs_lists:
            for leaf_dir in tqdm(leaf_dirs, total=len(leaf_dirs), desc="Inferencing leaf directories"):
                # Find all image paths in the current directory
                full_leaf_path = os.path.join(args.input_dir, leaf_dir)

                # pdf to image conversion
                pdf_paths = find_pdfs(full_leaf_path)
                image_dir = full_leaf_path.replace(args.input_dir, os.path.join(args.output_dir, "pdf_images/"))
                os.makedirs(image_dir, exist_ok=True)
                for pdf_path in pdf_paths:
                    images = convert_from_path(pdf_path)
                    for i, image in enumerate(images):
                        image.save(os.path.join(image_dir, f"{os.path.basename(pdf_path)}_{i}.jpg"))

                # image paths
                image_paths = find_images(image_dir)
                image_paths.extend(find_images(full_leaf_path))
                
                # Split images to match the number of processes
                chunks = [[] for _ in range(num_processes)]
                for i, image_path in enumerate(image_paths):
                    process_idx = i % num_processes
                    chunks[process_idx].append(image_path)
                
                # Batch process with each process
                process_func = partial(process_batch, leaf_dir=leaf_dir, args=args, json_input=json_input)
                results = []
                for res in pool.imap_unordered(process_func, chunks):
                    results.extend(res)
    finally:
        # Close pool after task completion
        pool.close()
        pool.join()
        print(f"Inference completed for {args.input_dir}_{args.num_chunk}_{args.start_chunk_index}")


if __name__ == "__main__":
    args, json_input = parse_args()
    main(args, json_input)