import os
from datasets import Dataset, DatasetDict

# ref: https://huggingface.co/datasets/upstage/kie-bench

def aggregate_dataset(base_path, bench_name, test_id=0, HF_dataset=[]):
    """
    item example: {'test_id': 'IEX-101', 'bench_type': 'RealKIE', 'doc_type': 'charities', 
                    'doc_name': '00c5f02bb4d35366915b1d44fa610452', 
                    'file_path': 'data/v3_0/realkie_big/charities/files/00c5f02bb4d35366915b1d44fa610452.pdf', 
                    'schema_path': 'data/v3_0/realkie_big/charities/schema/00c5f02bb4d35366915b1d44fa610452.json', 
                    'gold_path': 'data/v3_0/realkie_big/charities/gold_result/00c5f02bb4d35366915b1d44fa610452.json', 
                    'num_pages': 12, 'num_entities': 19}
    """


    if bench_name == "KIE-universal":
        bench_type = "KIE-universal"
        bench_path = os.path.join(base_path, "universal")
        doc_type = None
    elif bench_name == "OmniAI-OCR":
        bench_type = "OmniAI-OCR"
        bench_path = os.path.join(base_path, "public", "default", "omniai_ocr_benchmark")
        doc_type = None
    elif bench_name == "RealKIE/charities":
        bench_type = "RealKIE"
        doc_type = "charities"
        bench_path = os.path.join(base_path, "public", "default", "RealKIE", "charities")
    elif bench_name == "RealKIE/fcc_invoices":
        bench_type = "RealKIE"
        doc_type = "fcc_invoices"
        bench_path = os.path.join(base_path, "public", "default", "RealKIE", "fcc_invoices")
    elif bench_name == "RealKIE/nda":
        bench_type = "RealKIE"
        doc_type = "nda"
        bench_path = os.path.join(base_path, "public", "default", "RealKIE", "nda")
    elif bench_name == "Sensible-Insurance":
        bench_type = "Sensible-Insurance"
        bench_path = os.path.join(base_path, "documents", "sensible_insurance")        
        doc_type = None
    else:
        raise ValueError(f"Invalid benchmark name: {bench_name}")


    # universal
    image_dir = os.path.join(bench_path, "files")
    schema_dir = os.path.join(bench_path, "schema")
    gold_result_dir = os.path.join(bench_path, "gold_result")

    items = os.listdir(image_dir)

    # Process each item in the directory
    for item in items:
        doc_name = os.path.basename(item).split(".")[0]
        image_file_path = os.path.join(image_dir, item)
        schema_file_path = os.path.join(schema_dir, doc_name + ".json")
        gold_file_path = os.path.join(gold_result_dir, doc_name + ".json")

        if bench_name == "KIE-universal":
            doc_name_split = doc_name.split("_")
            if doc_name_split[0].isdigit():
                # Handle the case when the first part is a number
                doc_type = doc_name_split[1]
            else:
                doc_type = '_'.join(doc_name_split[:-1])
        elif bench_name == "OmniAI-OCR" or bench_name == "Sensible-Insurance":
            # TODO: No information about doc_type in OmniAI-OCR folder but it exists in HF dataset of upstage/KIE-bench. 
            # Since not used for lmms-eval, we set it to ""
            doc_type = "" 
        elif doc_type is None:
            raise ValueError(f"Invalid Value for doc_type: {doc_type}")


        HF_dataset.append({
            "test_id": f"IEX-{test_id:03d}",
            "bench_type": bench_type,
            "doc_type": doc_type,
            "doc_name": doc_name,
            "file_path": image_file_path,
            "schema_path": schema_file_path,
            "gold_path": gold_file_path
        })
        test_id += 1


    return HF_dataset, test_id

if __name__ == "__main__":
    base_path = "/app/docfm/datasets/benchmark/key_information_extraction/v3.1"

    # universal
    HF_dataset = []
    HF_dataset, test_id = aggregate_dataset(base_path, "KIE-universal", test_id=0, HF_dataset=HF_dataset)
    HF_dataset, test_id = aggregate_dataset(base_path, "RealKIE/charities", test_id, HF_dataset)
    HF_dataset, test_id = aggregate_dataset(base_path, "RealKIE/fcc_invoices", test_id, HF_dataset)
    HF_dataset, test_id = aggregate_dataset(base_path, "RealKIE/nda", test_id, HF_dataset)
    HF_dataset, test_id = aggregate_dataset(base_path, "Sensible-Insurance", test_id, HF_dataset)
    HF_dataset, test_id = aggregate_dataset(base_path, "OmniAI-OCR", test_id, HF_dataset)


    # Create the dataset
    dataset = Dataset.from_list(HF_dataset)
    # Convert the dataset to a DatasetDict with "test" as the key
    dataset_dict = DatasetDict({"test": dataset})
    dataset = dataset_dict

    # Save the dataset
    dataset_path = os.path.join('/'.join(base_path.split('/')[:-1]), os.path.basename(base_path)+'_HuggingFace')
    os.makedirs(dataset_path, exist_ok=True)
    dataset.save_to_disk(dataset_path)
    print(f'Saved dataset: {dataset_path}')