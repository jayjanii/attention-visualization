import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from transformers import ViTForImageClassification, AutoImageProcessor
    from PIL import Image
    import requests
    import torch.nn.functional as F

    return (
        AutoImageProcessor,
        F,
        Image,
        ViTForImageClassification,
        requests,
        torch,
    )


@app.cell
def _(AutoImageProcessor, ViTForImageClassification, torch):
    model_name = "google/vit-large-patch16-224"
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
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
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
    return attn, cls_attn


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
def _(F, unflat_patch_attn):
    patch_attn_upsampled = F.interpolate(
        unflat_patch_attn.unsqueeze(0).unsqueeze(0),
        size=(224,224),
        mode="bilinear"
    )

    patch_attn_upsampled = patch_attn_upsampled.squeeze()
    print(patch_attn_upsampled.shape)
    return (patch_attn_upsampled,)


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
    return image_resized, np, plt


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
    import urllib.request
    import zipfile
    import os

    os.makedirs("data", exist_ok=True)

    if not os.path.exists("data/annotations"):
        print("Downloading annotations...")
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "data/annotations.zip"
        )
        with zipfile.ZipFile("data/annotations.zip", "r") as z:
            z.extractall("data")
        print("Done!")
    return (os,)


@app.cell
def _():
    from pycocotools.coco import COCO

    coco = COCO("data/annotations/instances_val2017.json")

    categories = ["cat", "horse", "car", "bicycle"]
    images_per_category = 25

    image_urls = {}

    for category in categories:
        cat_ids = coco.getCatIds(catNms=[category])
        img_ids = coco.getImgIds(catIds=cat_ids)[:images_per_category]
        imgs = coco.loadImgs(img_ids)
        image_urls[category] = [img["coco_url"] for img in imgs]

    # Quick sanity check
    for cat, urls in image_urls.items():
        print(f"{cat}: {len(urls)} images")
    return image_urls, imgs


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Refactor Existing Mapping Code into Functions
    """)
    return


@app.cell
def _(F, attn, torch):
    def last_layer_attention_map(image, device, model, processor):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class = outputs.logits.argmax(-1).item()

        attn = outputs.attentions[-1]

        cls_attn = attn[0, :, 0, :]
        cls_attn_mean = cls_attn.mean(dim=0)
        patch_attn = cls_attn_mean[1:]
        unflat_patch_attn = torch.unflatten(patch_attn, dim=0, sizes=(14,14))

        patch_attn_upsampled = F.interpolate(
            unflat_patch_attn.unsqueeze(0).unsqueeze(0),
            size=(224,224),
            mode="bilinear"
        )
    
        patch_attn_upsampled = patch_attn_upsampled.squeeze()

        return patch_attn_upsampled

    def rollout_attention_map(image, device, model, processor):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class = outputs.logits.argmax(-1).item()

        attn_layers = [a[0].mean(dim=0) for a in outputs.attentions]
        rollout = torch.eye(197).to(device)
    
        for attn_layer in attn_layers:
            attn_layer = attn_layer.to(device)
            attn_with_residual = attn_layer + torch.eye(197).to(device)
            attn_normalized = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
            rollout = attn_normalized @ rollout
    
        cls_rollout = rollout[0, 1:]

        cls_attn = attn[0, :, 0, :]
        cls_attn_mean = cls_attn.mean(dim=0)
        patch_attn = cls_attn_mean[1:]

        unflat_cls_rollout = torch.unflatten(cls_rollout, dim=0, sizes=(14,14))

        cls_rollout_upsampled = F.interpolate(
            unflat_cls_rollout.unsqueeze(0).unsqueeze(0),
            size=(224,224),
            mode="bilinear"
        )
    
        cls_rollout_upsampled = cls_rollout_upsampled.squeeze()

        return cls_rollout_upsampled

    return last_layer_attention_map, rollout_attention_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Heatmaps given [224, 224] shaped tensor map
    """)
    return


@app.cell
def _(np, plt):
    def generate_heatmap(image, attn_map, prediction, save_path):
        image_resized = np.array(image.resize((224, 224)))

        attn_map = attn_map.cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Predicted: {prediction}", fontsize=14, fontweight="bold", y=1.02)

        axes[0].imshow(image_resized)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(attn_map, cmap="hot")
        axes[1].set_title("Attention Map")
        axes[1].axis("off")

        axes[2].imshow(image_resized)
        axes[2].imshow(attn_map, cmap="hot", alpha=0.75)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return (generate_heatmap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing
    """)
    return


@app.cell
def _(imgs):
    imgs
    return


@app.cell
def _(
    Image,
    device,
    generate_heatmap,
    image_urls,
    last_layer_attention_map,
    model,
    os,
    processor,
    requests,
    rollout_attention_map,
    torch,
):
    def _():
        os.makedirs("output/last_layer", exist_ok=True)
        os.makedirs("output/rollout", exist_ok=True)

        for category, urls in image_urls.items():
            for i, url in enumerate(urls):
                image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
                with torch.no_grad():
                    outputs = model(**inputs)

                predicted_class = outputs.logits.argmax(-1).item()
                prediction = model.config.id2label[predicted_class]
            
                last_layer_map = last_layer_attention_map(image, device, model, processor)
                rollout_map = rollout_attention_map(image, device, model, processor)
            
                generate_heatmap(image, last_layer_map, prediction, f"output/last_layer/{category}_{i}.png")
                generate_heatmap(image, rollout_map, prediction, f"output/rollout/{category}_{i}.png")
            print(f"Done: {category} {i+1}/25")

        return
    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
