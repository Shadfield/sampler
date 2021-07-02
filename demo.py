from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

import gradio as gr
from modules import sampler

# Models
models = [
    'Downsample',
    'Random',
    'MixedAdaptiveRandom',
    'LevelCross',
    'SAUCE',
    'DeepSAUCE'
]
weight_dir = Path('./weights')


def sample(image, sample_rate, mode):

    # Setup input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.tensor(image, device=device).permute(2, 0, 1).unsqueeze(0)
    img = img.to(torch.float) / 255
    droprate = torch.tensor(1 - sample_rate, device=device)

    heatmaps, samplemaps = {}, {}
    for mod in models:
        # Load model + weights
        this_model = getattr(sampler, mod)()
        if (weight_dir/f"{mod}.pt").exists():
            this_model.load_state_dict(torch.load(weight_dir/f"{mod}.pt", map_location='cpu'))
        this_model = this_model.to(device)
        this_model.droprate = droprate

        # Run model
        hmap, smap = this_model(img)
        heatmaps[mod] = to_pil_image(hmap.squeeze(0))
        samplemaps[mod] = to_pil_image(smap.squeeze(0))

    return samplemaps.values() if mode == 'Sample maps' else heatmaps.values()


iface = gr.Interface(
    fn=sample,
    inputs=[
        gr.inputs.Image(shape=(224, 224)),
        gr.inputs.Slider(0, 1.),
        gr.inputs.Radio(['Proability maps', 'Sample maps'])
    ],
    outputs=[gr.outputs.Image("pil", label=mod) for mod in models]
)
iface.launch(share=False)
