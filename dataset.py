"""MS COCO 2017 data loading, image preprocessing, and DistilBERT tokenization."""
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer


class COCOCaptionDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, max_length: int = 64, max_samples: int = None):
        self.img_dir = img_dir
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        with open(ann_file) as f:
            data = json.load(f)

        id_to_file = {img['id']: img['file_name'] for img in data['images']}
        self.samples = [
            {'image_file': id_to_file[ann['image_id']], 'caption': ann['caption']}
            for ann in data['annotations']
            if ann['image_id'] in id_to_file
        ]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self.transform(
            Image.open(os.path.join(self.img_dir, sample['image_file'])).convert('RGB')
        )
        tokens = self.tokenizer(
            sample['caption'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
        }


def get_dataloader(
    img_dir: str,
    ann_file: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    max_samples: int = None,
) -> DataLoader:
    return DataLoader(
        COCOCaptionDataset(img_dir, ann_file, max_samples=max_samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
