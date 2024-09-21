"""
import pandas as pd
import json

def csv2json(input_path,output_path):

    # 读取CSV文件
    csv_file_path = 'fix.csv'  # 替换为实际的CSV文件路径
    df = pd.read_csv(csv_file_path)

    # 替换 file_path 列中所有的反斜杠为正斜杠
    df['file_path'] = df['file_path'].apply(lambda x: x.replace('\\', '/'))

    # 检查并修复 captions 列中的数据
    def safe_json_loads(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {text}")
            return None  # or you could return an empty list, etc.

    df['captions'] = df['captions'].apply(safe_json_loads)

    # 过滤掉解析失败的行
    df = df[df['captions'].notnull()]

    # 将DataFrame转换为字典列表
    records = df.to_dict(orient='records')

    # 保存为JSON文件，确保不转义反斜杠
    json_file_path = 'output_file.json'  # 指定保存的JSON文件名
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=4)

    print(f"JSON文件已保存至: {json_file_path}")
if __name__ == "__main__":
    csv2json("input_path","output_path")
"""

import pandas as pd
import json

def csv2json(input_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 替换 file_path 列中所有的反斜杠为正斜杠
    df['file_path'] = df['file_path'].apply(lambda x: x.replace('\\', '/'))

    # 检查并修复 captions 列中的数据
    def safe_json_loads(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {text}")
            return None  # 或者可以返回一个空列表等其他适合的值

    df['captions'] = df['captions'].apply(safe_json_loads)

    # 过滤掉解析失败的行
    df = df[df['captions'].notnull()]

    # 将DataFrame转换为字典列表
    records = df.to_dict(orient='records')

    # 保存为JSON文件，确保不转义反斜杠
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=4)

    print(f"JSON文件已保存至: {output_path}")

if __name__ == "__main__":
    # 需要指定输出JSON文件的完整路径，包括文件名
    csv2json("E:\\ss\\datasets\\dataset_info_cn.csv", "E:\\ss\\datasets\\output_file.json")
