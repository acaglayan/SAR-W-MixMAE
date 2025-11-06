# THIRD_PARTY.md — File Mapping and License Status

This document enumerates third-party origins, statuses (verbatim/modified/new), and notes.

| File/Path | Source | Status | License Tag | Notes |
|---|---|---|---|---|
| util/pos_embed.py | MixMIM (SenseTime) | Verbatim | SPDX: NOASSERTION | Redistributed with permission (2025-10-15) |
| util/lr_sched.py | MixMIM (SenseTime) | Verbatim | SPDX: NOASSERTION | Redistributed with permission (2025-10-15) |
| util/lr_decay.py | MixMIM (SenseTime) | Verbatim | SPDX: NOASSERTION | Redistributed with permission (2025-10-15) |
| util/datasets.py | MixMIM (SenseTime) | Verbatim | SPDX: NOASSERTION | Redistributed with permission (2025-10-15) |
| util/crop.py | MixMIM (SenseTime) | Verbatim | SPDX: NOASSERTION | Redistributed with permission (2025-10-15) |
| util/misc.py | MixMIM (SenseTime) | Modified | SPDX: NOASSERTION | Migrated AMP to torch.amp.autocast('cuda') and update ckpt load for new np |
| main_pretrain.py | MixMIM (SenseTime) | Modified | SPDX: NOASSERTION | Weighting flags, exponential ckpt save, recon inference, ckpt comparison, AMP |
| main_finetune.py | MixMIM (SenseTime) | Modified | SPDX: NOASSERTION | Switched to multi-label finetune flow, multi-metrics printing |
| engine_pretrain.py | MixMIM (SenseTime) | Modified | SPDX: NOASSERTION | Applies token-weighted reconstruction loss, signature change, AMP |
| engine_finetune.py | MixMIM (SenseTime) | Modified | SPDX: NOASSERTION | Multi-label loss; torchmetrics AP/F1/Precision; AMP API; returns macro/micro metrics |
| models_mixmim.py | MixMIM (SenseTime) | Modified | SPDX: NOASSERTION | Retains MixMIM core, 2-ch patchify/unpatchify, weighted forward_loss |
| models_mixmim_ft.py | MixMIM (SenseTime) | Modified | SPDX: NOASSERTION | Adapted head/stem for 2-ch multi-label |
| models_sen12_ft.py | MixMIM (SenseTime) | New (derived) | SPDX: NOASSERTION | New file derived from MixMIM FT; pairwise head: feature diff + pooling + binary cls |

All other files (below) are original to this repository and licensed under **MIT**:

- sarwmix/bigearthnetv1.py
- sarwmix/bigearthnetv2.py
- sarwmix/helper.py
- sarwmix/sen12_clean_data.py
- sarwmix/sen12_data_alignment.py
- sarwmix/sen12_data_prep.py
- sarwmix/sen12_dataset_utils.py
- sarwmix/sen12flood_loader.py
- sarwmix/weighting.py
- scripts/*, datasets/* (as applicable)

