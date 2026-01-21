import torch

def collate(batch):
    # 長さで降順ソート
    batch = sorted(batch, key=lambda x: len(x["input_ids"]), reverse=True)

    # 最大長
    max_len = len(batch[0]["input_ids"])

    # input_ids をパディング
    input_ids = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
        input_ids.append(ids)

    input_ids = torch.stack(input_ids)

    # label をまとめる
    labels = torch.stack([item["label"] for item in batch])

    return {
        "input_ids": input_ids,
        "label": labels
    }

# 入力データ
batch = [
    {
        'text': 'hide new secretions from the parental units',
        'label': torch.tensor([0.]),
        'input_ids': torch.tensor([5785, 66, 113845, 18, 12, 15095, 1594])
    },
    {
        'text': 'contains no wit , only labored gags',
        'label': torch.tensor([0.]),
        'input_ids': torch.tensor([3475, 87, 15888, 90, 27695, 42637])
    },
    {
        'text': 'that loves its characters and communicates something rather beautiful about human nature',
        'label': torch.tensor([1.]),
        'input_ids': torch.tensor([4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964])
    },
    {
        'text': 'remains utterly satisfied to remain the same throughout',
        'label': torch.tensor([0.]),
        'input_ids': torch.tensor([987, 14528, 4941, 873, 12, 208, 898])
    }
]

result = collate(batch)
print(result)
