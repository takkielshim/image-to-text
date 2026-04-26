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

    # --- Image encoder (static shape: batch=1, 3x224x224) ---
    dummy_img = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model.image_encoder,
        dummy_img,
        'image_encoder.onnx',
        input_names=['image'],
        output_names=['embedding'],
        opset_version=17,
    )
    print('Exported image_encoder.onnx  [1, 3, 224, 224]')

    # --- Text encoder (static shape: batch=1, seq_len=64) ---
    dummy_ids = torch.zeros(1, 64, dtype=torch.long)
    dummy_mask = torch.ones(1, 64, dtype=torch.long)
    torch.onnx.export(
        model.text_encoder,
        (dummy_ids, dummy_mask),
        'text_encoder.onnx',
        input_names=['input_ids', 'attention_mask'],
        output_names=['embedding'],
        opset_version=17,
    )
    print('Exported text_encoder.onnx  [1, 64]')


def submit_to_ai_hub(device_name: str = 'Samsung Galaxy S24') -> None:
    """Upload ONNX models to Qualcomm AI Hub and compile for the target device.
    Run `qai_hub.get_devices()` to list all available device names.
    """
    print(f'Uploading models to Qualcomm AI Hub... (device: {device_name})')

    img_hub_model = qai_hub.upload_model('image_encoder.onnx')
    img_job = qai_hub.submit_compile_job(
        model=img_hub_model,
        device=qai_hub.Device(device_name),
        options='--quantize_full_type int8 --quantize_io',
    )
    print(f'Image encoder compile job submitted: {img_job.job_id}')

    txt_hub_model = qai_hub.upload_model('text_encoder.onnx')
    txt_job = qai_hub.submit_compile_job(
        model=txt_hub_model,
        device=qai_hub.Device(device_name),
        # Use QNN context binary runtime instead of TFLite:
        # - TFLite int8: crashes (QuantizeMultiplierSmallerThanOneExp on SUB)
        # - TFLite w8a16: CAST and SELECT_V2 unsupported
        # - qnn_context_binary int8: QNN handles scale differently, may pass
        options='--target_runtime qnn_context_binary --quantize_full_type int8 --quantize_io --truncate_64bit_io',
    )
    print(f'Text encoder compile job submitted: {txt_job.job_id}')

    print('Monitor jobs at https://app.aihub.qualcomm.com')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/final_model.pt')
    parser.add_argument('--embed_dim',  type=int, default=256)
    parser.add_argument('--device',     default='Samsung Galaxy S24',
                        help='AI Hub device name (run qai_hub.get_devices() to list)')
    parser.add_argument('--list_devices', action='store_true',
                        help='Print available AI Hub devices and exit')
    args = parser.parse_args()

    if args.list_devices:
        import qai_hub
        for d in qai_hub.get_devices():
            print(d.name)
    else:
        export_onnx(checkpoint=args.checkpoint, embed_dim=args.embed_dim)
        submit_to_ai_hub(device_name=args.device)
