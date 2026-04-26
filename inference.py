"""Run on-device inference via Qualcomm AI Hub.
Encodes a query image and text, then finds the closest match on-device.

Usage:
    python inference.py --image_job <image_job_id> --text_job <text_job_id>
    python inference.py --image_job abc123 --text_job def456 --query "a dog on the beach"
"""
import argparse
import numpy as np
from PIL import Image
from transformers import DistilBertTokenizer
import qai_hub


# ── Image preprocessing (must match training transform) ───────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))   # HWC → CHW
    return np.expand_dims(arr, axis=0)   # (1, 3, 224, 224)


# ── Text preprocessing (must match training tokenizer) ────────────────────────
def preprocess_text(text: str, max_length: int = 64):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np',
    )
    # truncate int64 → int32 (matches --truncate_64bit_io compile option)
    input_ids     = tokens['input_ids'].astype(np.int32)
    attention_mask = tokens['attention_mask'].astype(np.int32)
    return input_ids, attention_mask


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(
    image_job_id: str,
    text_job_id: str,
    image_path: str,
    query: str,
    device_name: str,
) -> None:
    device = qai_hub.Device(device_name)

    # Retrieve compiled models from previous compile jobs
    image_model = qai_hub.get_job(image_job_id).get_target_model()
    text_model  = qai_hub.get_job(text_job_id).get_target_model()

    # ── Image encoding ────────────────────────────────────────────────────────
    print(f'Image: {image_path}')
    image_input = preprocess_image(image_path)

    image_inference_job = qai_hub.submit_inference_job(
        model=image_model,
        device=device,
        inputs=dict(image=[image_input]),
    )
    image_output = image_inference_job.download_output_data()
    print(f'Image output keys: {list(image_output.keys())}')
    image_embedding = list(image_output.values())[0][0]   # first output
    print(f'Image embedding shape: {image_embedding.shape}')

    # ── Text encoding ─────────────────────────────────────────────────────────
    print(f'Query: "{query}"')
    input_ids, attention_mask = preprocess_text(query)

    text_inference_job = qai_hub.submit_inference_job(
        model=text_model,
        device=device,
        inputs=dict(
            input_ids=[input_ids],
            attention_mask=[attention_mask],
        ),
    )
    text_output = text_inference_job.download_output_data()
    print(f'Text output keys: {list(text_output.keys())}')
    text_embedding = list(text_output.values())[0][0]    # first output
    print(f'Text embedding shape: {text_embedding.shape}')

    # ── Cosine similarity ─────────────────────────────────────────────────────
    img_norm = image_embedding / np.linalg.norm(image_embedding)
    txt_norm = text_embedding  / np.linalg.norm(text_embedding)
    similarity = float(np.dot(img_norm.flatten(), txt_norm.flatten()))
    print(f'\nCosine similarity: {similarity:.4f}')
    print('(1.0 = perfect match, 0.0 = unrelated, -1.0 = opposite)')


# ── Profile job (optional: detailed latency breakdown) ────────────────────────
def run_profile(image_job_id: str, text_job_id: str, device_name: str) -> None:
    device = qai_hub.Device(device_name)

    image_model = qai_hub.get_job(image_job_id).get_target_model()
    text_model  = qai_hub.get_job(text_job_id).get_target_model()

    print('Submitting profile jobs...')
    img_profile = qai_hub.submit_profile_job(model=image_model, device=device)
    txt_profile = qai_hub.submit_profile_job(model=text_model,  device=device)

    print(f'Image encoder profile job: {img_profile.job_id}')
    print(f'Text  encoder profile job: {txt_profile.job_id}')
    print('Monitor at https://app.aihub.qualcomm.com')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_job',  required=True,
                        help='Compile job ID for the image encoder')
    parser.add_argument('--text_job',   required=True,
                        help='Compile job ID for the text encoder')
    parser.add_argument('--image',      default='images/dog.jpg',
                        help='Path to query image')
    parser.add_argument('--query',      default='a photo of a dog',
                        help='Natural language query text')
    parser.add_argument('--device',     default='Samsung Galaxy S24')
    parser.add_argument('--profile',    action='store_true',
                        help='Submit profile jobs instead of inference')
    args = parser.parse_args()

    if args.profile:
        run_profile(args.image_job, args.text_job, args.device)
    else:
        run_inference(args.image_job, args.text_job, args.image, args.query, args.device)
