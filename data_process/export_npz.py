import os
import json
import pickle
from transformers import AutoTokenizer,BertTokenizer


def export(json_path, output_path):
    print("start")
    with open(json_path, 'r', encoding='utf-8') as filename:
        caption_all = json.load(filename)
    folders = {}
    for caption in caption_all:
        f = os.path.split(caption["file_path"])[0]
        if f not in folders:
            folders[f] = 0
    tokenizer = AutoTokenizer.from_pretrained("E:\\ss\\bert-base-chinese")
    encoded_text = {
        "train": {"caption_id": [],
                  "attention_mask": [],
                  "images_path": [],
                  "labels": []},
        "test": {"caption_id": [],
                 "attention_mask": [],
                 "images_path": [],
                 "labels": []},
        "val": {"caption_id": [],
                "attention_mask": [],
                "images_path": [],
                "labels": []}
    }

    seen_id = []
    for image in caption_all:
        # print(os.path.split(image["file_path"]))
        folder = os.path.split(image["file_path"])[0]
        # print(folder)
        if image["id"] not in seen_id:
            # print(image["id"])
            seen_id.append(image["id"])
            folders[folder] += 1
        if folders[folder] % 10 == 1:
            stage = 'test'
        elif folders[folder] % 10 == 2:
            stage = "val"
        else:
            stage = "train"

        for text in image["captions"]:
            encoded = tokenizer(text, max_length=128, padding="max_length", truncation=True,return_tensors='pt')
            image_path = image["file_path"]

            encoded_text[stage]["caption_id"].append(encoded["input_ids"])
            encoded_text[stage]["attention_mask"].append(encoded["attention_mask"])
            encoded_text[stage]["images_path"].append(image_path)
            encoded_text[stage]["labels"].append(image["id"])

    for stage in encoded_text:
        encoded_text[stage]["caption_id"] = (encoded_text[stage]["caption_id"])
        encoded_text[stage]["attention_mask"] = (encoded_text[stage]["attention_mask"])
        encoded_text[stage]["images_path"] = (encoded_text[stage]["images_path"])
        encoded_text[stage]["labels"] = (encoded_text[stage]["labels"])
        file_path = os.path.join(output_path, f"{stage}_64.npz")
        with open(file_path, "wb") as f_pkl:
            pickle.dump(encoded_text[stage], f_pkl)
    # print(len(encoded_text["train"]))
    # print(len(encoded_text["train"]["labels"]))
    # print(len(encoded_text["train"]["caption_id"]))
    # print(len(encoded_text["train"]["attention_mask"]))
    # print(len(encoded_text["train"]["images_path"]))
    # print(len(encoded_text["val"]["labels"]))
    # print(len(encoded_text["test"]["labels"]))
    print("end")

    return encoded_text

if __name__ == "__main__":
    export('E:\ss\datasets\CUHK-PEDES\sample_cn_100.json','E:\ss\datasets\BERT_encode')