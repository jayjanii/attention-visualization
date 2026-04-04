import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from transformers import ViTForImageClassification, AutoImageProcessor
    from PIL import Image
    import requests

    return (
        AutoImageProcessor,
        Image,
        ViTForImageClassification,
        requests,
        torch,
    )


@app.cell
def _(AutoImageProcessor, ViTForImageClassification, torch):
    model_name = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTForImageClassification.from_pretrained(
        model_name,
        output_attentions=True
    ).to(device)
    model.eval()
    return device, model, processor


@app.cell
def _(Image, requests):
    url = "http://farm7.staticflickr.com/6074/6052102674_a8e19571e5_z.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image
    return (image,)


@app.cell
def _(device, image, model, processor, torch):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    return (outputs,)


@app.cell
def _(model, outputs):
    predicted_class = outputs.logits.argmax(-1).item()
    print(model.config.id2label[predicted_class])
    return


@app.cell
def _(outputs):
    print(f"Number of layers: {len(outputs.attentions)}")
    print(f"Attention shape per layer: {outputs.attentions[0].shape}")
    return


@app.cell
def _(outputs):
    # shape: [1, batch size
    #         12, attention heads
    #         197, token i
    #         197] token j
    #
    # "how much does token i attend to token j"


    # get attention weights from last layer
    attn = outputs.attentions[-1]


    # Extract CLS token attention to all other tokens j
    cls_attn = attn[0, :, 0, :]

    print(cls_attn.shape)
    return (cls_attn,)


@app.cell
def _(cls_attn):
    cls_attn_mean = cls_attn.mean(dim=0)
    print(cls_attn_mean.shape)
    return (cls_attn_mean,)


@app.cell
def _(cls_attn_mean):
    patch_attn = cls_attn_mean[1:]
    print(patch_attn.shape)
    return (patch_attn,)


@app.cell
def _(patch_attn, torch):
    unflat_patch_attn = torch.unflatten(patch_attn, dim=0, sizes=(14,14))
    print(unflat_patch_attn.shape)
    return (unflat_patch_attn,)


@app.cell
def _(unflat_patch_attn):
    import torch.nn.functional as F

    patch_attn_upsampled = F.interpolate(
        unflat_patch_attn.unsqueeze(0).unsqueeze(0),
        size=(224,224),
        mode="bilinear"
    )

    patch_attn_upsampled = patch_attn_upsampled.squeeze()
    print(patch_attn_upsampled.shape)
    return F, patch_attn_upsampled


@app.cell
def _(image, patch_attn_upsampled):
    import matplotlib.pyplot as plt
    import numpy as np


    image_resized = np.array(image.resize((224,224)))

    # Convert attention to numpy
    attn_map = patch_attn_upsampled.cpu().numpy()

    # Normalize to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Attention map
    axes[1].imshow(attn_map, cmap="hot")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image_resized)
    axes[2].imshow(attn_map, cmap="hot", alpha=0.75)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("attention.png")
    plt.gca()
    return image_resized, plt


@app.cell
def _(device, outputs, torch):
    attn_layers = [a[0].mean(dim=0) for a in outputs.attentions]
    rollout = torch.eye(197).to(device)

    for attn_layer in attn_layers:
        attn_layer = attn_layer.to(device)
        attn_with_residual = attn_layer + torch.eye(197).to(device)
        attn_normalized = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        rollout = attn_normalized @ rollout

    cls_rollout = rollout[0, 1:]
    print(rollout.shape)
    print(cls_rollout.shape)
    return (cls_rollout,)


@app.cell
def _(F, cls_rollout, torch):
    unflat_cls_rollout = torch.unflatten(cls_rollout, dim=0, sizes=(14,14))
    print(unflat_cls_rollout.shape)

    cls_rollout_upsampled = F.interpolate(
        unflat_cls_rollout.unsqueeze(0).unsqueeze(0),
        size=(224,224),
        mode="bilinear"
    )

    cls_rollout_upsampled = cls_rollout_upsampled.squeeze()
    print(cls_rollout_upsampled.shape)
    return (cls_rollout_upsampled,)


@app.cell
def _(cls_rollout_upsampled, image_resized, plt):
    # Convert attention to numpy
    attn_map2 = cls_rollout_upsampled.cpu().numpy()

    # Normalize to [0, 1]
    attn_map2 = (attn_map2 - attn_map2.min()) / (attn_map2.max() - attn_map2.min())

    # Plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes2[0].imshow(image_resized)
    axes2[0].set_title("Original")
    axes2[0].axis("off")

    # Attention map
    axes2[1].imshow(attn_map2, cmap="hot")
    axes2[1].set_title("Attention Map")
    axes2[1].axis("off")

    # Overlay
    axes2[2].imshow(image_resized)
    axes2[2].imshow(attn_map2, cmap="hot", alpha=0.6)
    axes2[2].set_title("Overlay")
    axes2[2].axis("off")

    plt.tight_layout()
    plt.savefig("attention2.png")
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
