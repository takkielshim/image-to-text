"""Export trained model to ONNX and submit compile jobs to Qualcomm AI Hub.
Produces INT8-quantised QNN models targeting the Snapdragon XR2 Gen 2.
"""
import torch
import torch.onnx
import qai_hub

from model import ImageTextModel


def export_onnx(checkpoint: str = 'checkpoints/final_model.pt', embed_dim: int = 256) -> None:
    model = ImageTextModel(embed_dim=embed_dim)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    # --- Image encoder ---
    dummy_img = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model.image_encoder,
        dummy_img,
        'image_encoder.onnx',
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={'image': {0: 'batch'}, 'embedding': {0: 'batch'}},
        opset_version=17,
    )
    print('Exported image_encoder.onnx')

    # --- Text encoder ---
    dummy_ids = torch.zeros(1, 64, dtype=torch.long)
    dummy_mask = torch.ones(1, 64, dtype=torch.long)
    torch.onnx.export(
        model.text_encoder,
        (dummy_ids, dummy_mask),
        'text_encoder.onnx',
        input_names=['input_ids', 'attention_mask'],
        output_names=['embedding'],
        dynamic_axes={
            'input_ids': {0: 'batch'},
            'attention_mask': {0: 'batch'},
            'embedding': {0: 'batch'},
        },
        opset_version=17,
    )
    print('Exported text_encoder.onnx')


def submit_to_ai_hub() -> None:
    """Upload ONNX models to Qualcomm AI Hub and compile for Snapdragon XR2 Gen 2."""
    print('Uploading models to Qualcomm AI Hub...')

    img_hub_model = qai_hub.upload_model('image_encoder.onnx')
    img_job = qai_hub.submit_compile_job(
        model=img_hub_model,
        device=qai_hub.Device('Snapdragon XR2 Gen 2'),
        options='--quantize_full_type int8 --quantize_io',
    )
    print(f'Image encoder compile job submitted: {img_job.job_id}')

    txt_hub_model = qai_hub.upload_model('text_encoder.onnx')
    txt_job = qai_hub.submit_compile_job(
        model=txt_hub_model,
        device=qai_hub.Device('Snapdragon XR2 Gen 2'),
        options='--quantize_full_type int8 --quantize_io',
    )
    print(f'Text encoder compile job submitted: {txt_job.job_id}')

    print('Monitor jobs at https://app.aihub.qualcomm.com')


if __name__ == '__main__':
    export_onnx()
    submit_to_ai_hub()
