"""Phase 0 AuM reproduction: converted from examples/inference/inference.ipynb.
Loads the official AudioSet-pretrained VGGSound AuM checkpoint and runs
inference on the 5 bundled sample clips, comparing predictions to the
ground-truth labels shipped by the AuM authors themselves.

Verified working 2026-10-04 on a Thunder Compute RTX A6000 (compute
capability 8.6): 4/5 correct (80%), see
results/phase0_aum_vggsound_5sample_result.json and DECISIONS.md for the
exact environment recipe and why this is a pipeline-correctness check, not
a full dataset-level reproduction of the reported 46.78% VGGSound accuracy.

Run this FROM INSIDE a clone of third_party/Audio-Mamba-AuM, specifically
from its examples/inference/ directory (relies on relative imports/paths
and the 5 bundled sample wavs + datafiles/ that ship with that repo):

    cd third_party/Audio-Mamba-AuM/examples/inference
    cp <this file> run_inference.py
    python3 run_inference.py
"""
import csv
import json
import sys

import numpy as np
import torch
import torchaudio

sys.path.append("../../")
import src.models as models


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


data_args = Namespace(
    num_mel_bins=128,
    target_length=1024,
    mean=-5.0767093,
    std=4.4533687,
)

model_args = Namespace(
    model_type="base",
    n_classes=309,
    imagenet_pretrain=False,
    imagenet_pretrain_path=None,
    aum_pretrain=True,
    aum_pretrain_path="models/aum-base_audioset-vggsound.pth",
    aum_variant="Fo-Bi",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

embed_dim = {"base": 768, "small": 384, "tiny": 192}[model_args.model_type]
bimamba_type = {"Fo-Fo": "none", "Fo-Bi": "v1", "Bi-Bi": "v2"}[model_args.aum_variant]

AuM = models.AudioMamba(
    spectrogram_size=(data_args.num_mel_bins, data_args.target_length),
    patch_size=(16, 16),
    strides=(16, 16),
    embed_dim=embed_dim,
    num_classes=model_args.n_classes,
    imagenet_pretrain=model_args.imagenet_pretrain,
    imagenet_pretrain_path=model_args.imagenet_pretrain_path,
    aum_pretrain=model_args.aum_pretrain,
    aum_pretrain_path=model_args.aum_pretrain_path,
    bimamba_type=bimamba_type,
)
AuM.to(model_args.device)
AuM.eval()
print("Model loaded OK on", model_args.device)

index_dict, label_dict = {}, {}
with open("datafiles/class_labels_indices.csv", "r") as f:
    for row in csv.DictReader(f):
        index_dict[row["mid"]] = row["index"]
        label_dict[row["index"]] = row["display_name"]

eval_data = json.load(open("datafiles/eval.json"))["data"]

results = []
correct = 0
with torch.no_grad():
    for item in eval_data:
        waveform, sr = torchaudio.load(item["wav"])
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=data_args.num_mel_bins,
            dither=0.0,
            frame_shift=10,
        )
        n_frames = fbank.shape[0]
        p = data_args.target_length - n_frames
        if p > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0:
            fbank = fbank[0:data_args.target_length, :]
        fbank = (fbank - data_args.mean) / (data_args.std * 2)
        fbank = fbank.unsqueeze(0).to(model_args.device)

        output = torch.sigmoid(AuM(fbank)).cpu().numpy()[0]
        pred_idx = int(np.argmax(output))
        pred_label = label_dict[str(pred_idx)]
        true_label = label_dict[index_dict[item["labels"]]]
        is_correct = pred_label == true_label
        correct += int(is_correct)
        results.append(
            {
                "wav": item["wav"],
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob": float(output[pred_idx]),
                "correct": is_correct,
            }
        )
        status = "OK" if is_correct else "WRONG"
        print("{}: true={!r} pred={!r} ({:.3f}) {}".format(item["wav"], true_label, pred_label, output[pred_idx], status))

acc = correct / len(eval_data)
print("\nTop-1 accuracy on {} bundled samples: {:.3f} ({}/{})".format(len(eval_data), acc, correct, len(eval_data)))
json.dump(
    {"checkpoint": model_args.aum_pretrain_path, "n": len(eval_data), "top1_accuracy": acc, "results": results},
    open("aum_vggsound_5sample_result.json", "w"),
    indent=2,
)
