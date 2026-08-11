"""Boltz-1 run glue — ported from the fork's ``main.predict`` and ``aas-boltz-runner``.

Ported so that we get a **model handle** to hook, instead of the one-shot ``predict()`` CLI:

  process_inputs (once/protein)  →  BoltzInferenceDataModule  →  load_boltz_model (once, reused)
  →  for each condition: install hooks · Trainer.predict · BoltzWriter → mmCIF · remove hooks

Anchors: fork ``src/nutz_and_boltz/main.py`` (`check_inputs`:86, `process_inputs`:186, `predict`:416)
and ``aas-boltz-runner/task/run_npz.py`` (model load :191, trainer.predict :217). TensorLens recording
is intentionally **not** started here — we want structures, not activations.

Heavy imports (torch-lightning, nutz_and_boltz) are deferred into the functions so the rest of the
package (projection math, readout) imports and unit-tests without a Boltz install.
"""

from __future__ import annotations

import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
# The scale-up / pruning path uses the *conf* checkpoint; override via ExperimentConfig if needed.
MODEL_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters (verbatim from run_npz.py:55)."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class Processed:
    manifest: object
    targets_dir: Path
    msa_dir: Path


def _ensure_tensorlens_base() -> None:
    """The fork's forward() calls tensorlens.set_recording_directory_from_context(), which needs a
    base working dir even though we don't record (recording is deactivated). Point it at a throwaway
    dir so the call succeeds. Idempotent."""
    import os

    d = os.environ.get("TENSORLENS_WORKING_DIR") or str(Path.cwd() / ".tensorlens")
    os.environ["TENSORLENS_WORKING_DIR"] = d
    Path(d).mkdir(parents=True, exist_ok=True)
    try:
        from tensorlens.tensorlens import set_base_working_directory

        set_base_working_directory(d)
    except Exception:  # noqa: BLE001  # env var alone also satisfies get_base_working_directory
        pass


def download_cache(cache: Path, model_url: str = MODEL_URL) -> Path:
    """Ensure ccd.pkl + the checkpoint exist in ``cache``; return the checkpoint path."""
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310
    ckpt = cache / Path(model_url).name
    if not ckpt.exists():
        urllib.request.urlretrieve(model_url, str(ckpt))  # noqa: S310
    return ckpt


def process_protein(
    input_path: Path,
    work_dir: Path,
    ccd_path: Path,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    override: bool = False,
) -> Processed:
    """Parse + MSA-process one .yaml/.fasta into a Boltz processed dir; return handles.

    ``input_path`` must carry an MSA (or ``use_msa_server=True``). Mirrors main.predict:455-482.
    """
    from nutz_and_boltz.data.types import Manifest
    from nutz_and_boltz.main import check_inputs, process_inputs

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    data = check_inputs(Path(input_path), work_dir, override)
    if not data:
        # already processed → load existing manifest
        processed_dir = work_dir / "processed"
    else:
        process_inputs(
            data=data,
            out_dir=work_dir,
            ccd_path=ccd_path,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
        )
        processed_dir = work_dir / "processed"

    return Processed(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )


def make_datamodule(processed: Processed, num_workers: int = 2):
    """Build the inference datamodule (main.predict:485)."""
    from nutz_and_boltz.data.module.inference import BoltzInferenceDataModule

    return BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )


def load_boltz_model(
    checkpoint: Path,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
):
    """Load Boltz1 in eval mode on CPU (Trainer moves it to GPU). Mirrors run_npz.py:191."""
    import torch

    from nutz_and_boltz.model.model import Boltz1

    _ensure_tensorlens_base()

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale
    common = dict(
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
    )

    # torch >= 2.6 defaults torch.load(weights_only=True), which rejects the Boltz checkpoint
    # (it pickles an omegaconf.DictConfig). The checkpoint is the official boltz-community file, so
    # force weights_only=False for the duration of the load. Lightning doesn't expose weights_only
    # through load_from_checkpoint, so we patch torch.load directly and restore it after.
    _orig_load = torch.load

    def _load_full(*a, **k):
        k["weights_only"] = False
        return _orig_load(*a, **k)

    torch.load = _load_full
    try:
        try:
            model = Boltz1.load_from_checkpoint(checkpoint, ema=False, **common)  # scale-up path
        except TypeError:
            model = Boltz1.load_from_checkpoint(checkpoint, **common)  # main.py path
    finally:
        torch.load = _orig_load

    model.eval()
    return model


def predict_to_cif(
    model,
    datamodule,
    targets_dir: Path,
    predictions_dir: Path,
    accelerator: str = "gpu",
    devices: int = 1,
    output_format: str = "mmcif",
) -> Path:
    """Run one Boltz forward with hooks already installed; return the written mmCIF path.

    A fresh Trainer + BoltzWriter per call keeps outputs isolated per condition. The writer lays down
    ``predictions_dir/{record_id}/{record_id}_model_0.cif`` (writer.py:133-143).
    """
    from nutz_and_boltz.data.write.writer import BoltzWriter
    from pytorch_lightning import Trainer

    predictions_dir = Path(predictions_dir)
    writer = BoltzWriter(
        data_dir=str(targets_dir),
        output_dir=str(predictions_dir),
        output_format=output_format,
    )
    trainer = Trainer(
        default_root_dir=str(predictions_dir),
        strategy="auto",
        callbacks=[writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )
    trainer.predict(model, datamodule=datamodule, return_predictions=False)

    ext = "cif" if output_format == "mmcif" else output_format
    hits = sorted(predictions_dir.glob(f"*/*_model_0.{ext}"))
    if not hits:
        raise FileNotFoundError(f"No structure written under {predictions_dir}")
    return hits[0]
