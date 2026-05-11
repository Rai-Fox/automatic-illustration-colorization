from __future__ import annotations

import contextlib
import inspect
import os
import random
import time
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from illustration_colorizer.models.base import (
    ColorizationModel,
    ColorizationRequest,
    ColorizationResult,
    ModelBackendUnavailableError,
)
from illustration_colorizer.models.local_assets import ensure_hf_snapshot_dir
from illustration_colorizer.models.runtime import (
    isolated_vendor_imports,
    prepended_sys_path,
    project_path,
    request_option,
    request_seed,
    require_loaded,
    result,
    rgb_uint8,
)


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class CobraModel(ColorizationModel):
    _RATIOS = [
        (800, 800),
        (768, 896),
        (704, 928),
        (672, 960),
        (640, 1024),
        (608, 1056),
        (576, 1088),
        (576, 1184),
        (896, 768),
        (928, 704),
        (960, 672),
        (1024, 640),
        (1056, 608),
        (1088, 576),
        (1184, 576),
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._torch: Any | None = None
        self._transforms: Any | None = None
        self._image_processor: Any | None = None
        self._image_encoder: Any | None = None
        self._line_model: Any | None = None
        self._pipeline: Any | None = None
        self._multi_res_net: Any | None = None
        self._style: str | None = None
        self._snapshot_path: Path | None = None
        self._repo_path: Path | None = None

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def supports_multiple_references(self) -> bool:
        return True

    @property
    def supports_cpu(self) -> bool:
        return False

    def _filter_model_config(
        self,
        model_class: type[Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        parameters = inspect.signature(model_class.__init__).parameters
        accepted = {
            name for name in parameters if name not in {"self", "args", "kwargs"}
        }
        return {key: value for key, value in config.items() if key in accepted}

    def _patch_windows_safetensors_loader(
        self,
        *,
        model_loading_utils: Any,
        modeling_utils: Any,
    ) -> None:
        if os.name != "nt":
            return
        if getattr(model_loading_utils, "_cobra_windows_patch_applied", False):
            return

        from safetensors import safe_open

        original_load_state_dict = model_loading_utils.load_state_dict

        def _windows_safe_load_state_dict(
            checkpoint_file: str | os.PathLike[str],
            variant: str | None = None,
        ) -> Any:
            checkpoint_path = Path(checkpoint_file)
            if checkpoint_path.suffix == ".safetensors":
                with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
                    return OrderedDict((key, f.get_tensor(key)) for key in f.keys())
            return original_load_state_dict(checkpoint_file, variant=variant)

        model_loading_utils.load_state_dict = _windows_safe_load_state_dict
        modeling_utils.load_state_dict = _windows_safe_load_state_dict
        model_loading_utils._cobra_windows_patch_applied = True

    def load(self) -> None:
        if self._pipeline is not None:
            return

        try:
            import torch
            import transformers
            from torchvision import transforms
        except ImportError as exc:
            raise ModelBackendUnavailableError(
                f"Cobra dependencies are unavailable: {exc}"
            ) from exc

        transformers_version = str(getattr(transformers, "__version__", "unknown"))
        try:
            major_version = int(transformers_version.split(".", maxsplit=1)[0])
        except ValueError:
            major_version = -1
        if major_version >= 5:
            raise ModelBackendUnavailableError(
                "Cobra requires transformers<5 because its vendored diffusers "
                f"pipeline is incompatible with transformers {transformers_version}. "
                "Run `uv sync --group benchmark --group model-cobra` after "
                "updating dependencies."
            )

        if (
            not torch.cuda.is_available()
            or str(self.config.get("device", "cuda")) != "cuda"
        ):
            raise ModelBackendUnavailableError("Cobra is CUDA-only in this wrapper.")

        repo_path = project_path(self.config, "repo_path")
        snapshot_path = project_path(self.config, "snapshot_path")
        assert repo_path is not None and snapshot_path is not None
        self._repo_path = repo_path
        self._snapshot_path = snapshot_path
        self._torch = torch
        self._transforms = transforms

        try:
            with prepended_sys_path(repo_path / "diffusers" / "src"):
                with isolated_vendor_imports(repo_path):
                    from cobra_utils.utils import (
                        MultiHiddenResNetModel,
                        get_pixart_config,
                        init_causal_dit,
                        process_image,
                        process_image_Q_varres,
                        process_image_ref_varres,
                        res_skip,
                    )
                    from diffusers import (
                        CausalSparseDiTControlModel,
                        CausalSparseDiTModel,
                        CobraPixArtAlphaPipeline,
                        PixArtTransformer2DModel,
                    )
                    from diffusers.models import model_loading_utils, modeling_utils
                    from peft import LoraConfig
                    from transformers import (
                        CLIPImageProcessor,
                        CLIPVisionModelWithProjection,
                    )
        except ImportError as exc:
            raise ModelBackendUnavailableError(
                f"Cobra backend is unavailable: {exc}"
            ) from exc

        self._cobra_utils = {
            "process_image": process_image,
            "process_image_Q_varres": process_image_Q_varres,
            "process_image_ref_varres": process_image_ref_varres,
        }
        self._patch_windows_safetensors_loader(
            model_loading_utils=model_loading_utils,
            modeling_utils=modeling_utils,
        )
        weight_dtype = torch.float16

        line_model = res_skip()
        line_model.load_state_dict(torch.load(snapshot_path / "LE" / "erika.pth"))
        line_model.eval().cuda()
        self._line_model = line_model

        self._image_processor = CLIPImageProcessor()
        self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            snapshot_path / "image_encoder"
        ).to("cuda")

        block_out_channels = [128, 128, 256, 512, 512]
        multi_res_net = MultiHiddenResNetModel(
            block_out_channels, len(block_out_channels)
        )
        self._multi_res_net = multi_res_net

        pixart_repo_id = str(
            self.config.get("pixart_model", "PixArt-alpha/PixArt-XL-2-1024-MS")
        )
        pixart_local_dir = ensure_hf_snapshot_dir(
            project_root=Path(self.config["project_root"]),
            raw_path=str(self.config["pixart_local_dir"]),
            repo_id=pixart_repo_id,
            allow_download=bool(self.config.get("allow_download", False)),
        )
        transformer = PixArtTransformer2DModel.from_pretrained(
            str(pixart_local_dir),
            subfolder="transformer",
            revision=None,
            variant=None,
        )
        pixart_config = get_pixart_config()
        filtered_pixart_config = self._filter_model_config(
            CausalSparseDiTModel,
            pixart_config,
        )
        causal_dit = CausalSparseDiTModel(**filtered_pixart_config)
        causal_dit = init_causal_dit(causal_dit, transformer)
        control_config = self._filter_model_config(
            CausalSparseDiTControlModel,
            {**pixart_config, "cond_chanels": 9},
        )
        controlnet = CausalSparseDiTControlModel(**control_config)
        del transformer

        lora_rank = int(self.config.get("lora_rank", 128))
        causal_dit.add_adapter(
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=[
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "proj_in",
                    "proj_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "proj",
                    "linear",
                    "linear_1",
                    "linear_2",
                ],
            )
        )
        pipeline = CobraPixArtAlphaPipeline.from_pretrained(
            str(pixart_local_dir),
            transformer=causal_dit,
            controlnet=controlnet,
            safety_checker=None,
            revision=None,
            variant=None,
            torch_dtype=weight_dtype,
        ).to("cuda")
        self._pipeline = pipeline
        self._change_style(str(self.config.get("style", "line + shadow")))
        self._enable_vae_memory_savers(pipeline)

    def unload(self) -> None:
        self._pipeline = None
        self._multi_res_net = None
        self._line_model = None
        self._image_encoder = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _best_resolution(self, image: Image.Image) -> tuple[int, int]:
        input_rate = image.size[0] / image.size[1]
        return min(
            self._RATIOS, key=lambda ratio: abs(input_rate - ratio[0] / ratio[1])
        )

    def _round_to_multiple(
        self,
        value: float,
        *,
        multiple: int = 32,
        minimum: int = 64,
    ) -> int:
        return max(minimum, int(round(value / multiple)) * multiple)

    def _target_resolution(
        self,
        image: Image.Image,
        *,
        max_side: int | None,
    ) -> tuple[int, int]:
        width, height = self._best_resolution(image)
        if max_side is None or max_side <= 0:
            return width, height

        side = max(width, height)
        if side <= max_side:
            return width, height

        scale = max_side / side
        return (
            self._round_to_multiple(width * scale),
            self._round_to_multiple(height * scale),
        )

    def _scaled_size(
        self,
        width: int,
        height: int,
        *,
        scale: float,
    ) -> tuple[int, int]:
        bounded_scale = max(0.5, scale)
        return (
            self._round_to_multiple(width * bounded_scale, multiple=8),
            self._round_to_multiple(height * bounded_scale, multiple=8),
        )

    def _enable_vae_memory_savers(self, pipeline: Any) -> None:
        if _as_bool(self.config.get("vae_slicing", True)):
            enable_slicing = getattr(pipeline, "enable_vae_slicing", None)
            if callable(enable_slicing):
                enable_slicing()
            elif hasattr(pipeline, "vae"):
                vae_enable_slicing = getattr(pipeline.vae, "enable_slicing", None)
                if callable(vae_enable_slicing):
                    vae_enable_slicing()

        if _as_bool(self.config.get("vae_tiling", True)):
            enable_tiling = getattr(pipeline, "enable_vae_tiling", None)
            if callable(enable_tiling):
                enable_tiling()
            elif hasattr(pipeline, "vae"):
                vae_enable_tiling = getattr(pipeline.vae, "enable_tiling", None)
                if callable(vae_enable_tiling):
                    vae_enable_tiling()

    def _change_style(self, style: str) -> None:
        torch = require_loaded(self._torch, self.model_id)
        pipeline = require_loaded(self._pipeline, self.model_id)
        multi_res_net = require_loaded(self._multi_res_net, self.model_id)
        snapshot_path = require_loaded(self._snapshot_path, self.model_id)
        if style == "line":
            prefix = "line"
        elif style == "line + shadow":
            prefix = "shadow"
        else:
            raise ValueError(f"Invalid Cobra style: {style}")

        weight_dtype = torch.float16
        multi_res_net.load_state_dict(
            torch.load(
                snapshot_path / f"{prefix}_GSRP" / "MultiResNetModel.bin",
                map_location="cpu",
            ),
            strict=True,
        )
        multi_res_net.to("cuda", dtype=weight_dtype)
        pipeline.transformer.load_state_dict(
            torch.load(
                snapshot_path / f"{prefix}_ckpt" / "transformer_lora_pos.bin",
                map_location="cpu",
            ),
            strict=False,
        )
        pipeline.controlnet.load_state_dict(
            torch.load(
                snapshot_path / f"{prefix}_ckpt" / "controlnet.bin", map_location="cpu"
            ),
            strict=True,
        )
        pipeline.transformer.to("cuda", dtype=weight_dtype)
        pipeline.controlnet.to("cuda", dtype=weight_dtype)
        self._style = style

    def _extract_lines(self, image: Image.Image) -> Image.Image:
        import cv2

        torch = require_loaded(self._torch, self.model_id)
        line_model = require_loaded(self._line_model, self.model_id)
        src = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        rows = int(np.ceil(src.shape[0] / 16)) * 16
        cols = int(np.ceil(src.shape[1] / 16)) * 16
        patch = np.ones((1, 1, rows, cols), dtype="float32")
        patch[0, 0, : src.shape[0], : src.shape[1]] = src
        tensor = torch.from_numpy(patch).cuda()
        with torch.no_grad():
            lines = line_model(tensor)
        output = lines.cpu().numpy()[0, 0, : src.shape[0], : src.shape[1]]
        return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).convert("RGB")

    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        references = self.require_references(request)
        torch = require_loaded(self._torch, self.model_id)
        transforms = require_loaded(self._transforms, self.model_id)
        pipeline = require_loaded(self._pipeline, self.model_id)
        repo_path = require_loaded(self._repo_path, self.model_id)
        image_processor = require_loaded(self._image_processor, self.model_id)
        image_encoder = require_loaded(self._image_encoder, self.model_id)
        multi_res_net = require_loaded(self._multi_res_net, self.model_id)
        utils = require_loaded(getattr(self, "_cobra_utils", None), self.model_id)

        style = str(
            request_option(request, "style", self.config.get("style", "line + shadow"))
        )
        if style != self._style:
            self._change_style(style)

        seed = request_seed(request, int(self.config.get("seed", 1)))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        source = Image.fromarray(rgb_uint8(request.input_image)).convert("RGB")
        max_side_value = request_option(
            request,
            "max_side",
            self.config.get("max_side", 512),
        )
        max_side = int(max_side_value) if max_side_value is not None else None
        high_res_scale = float(
            request_option(
                request,
                "high_res_scale",
                self.config.get("high_res_scale", 1.0),
            )
        )
        tar_width, tar_height = self._target_resolution(source, max_side=max_side)
        query_origin = source.resize((tar_width, tar_height))
        query_bw = (
            self._extract_lines(query_origin)
            .resize((tar_width, tar_height))
            .convert("RGB")
        )
        hint_mask = Image.new("RGB", (tar_width, tar_height), "black")
        hint_color = query_bw
        query_vae = query_bw.resize(
            self._scaled_size(tar_width, tar_height, scale=high_res_scale)
        )
        reference_pils = [
            Image.fromarray(rgb_uint8(ref)).convert("RGB") for ref in references
        ]

        process_image = utils["process_image"]
        process_image_q = utils["process_image_Q_varres"]
        process_image_ref = utils["process_image_ref_varres"]

        top_k = int(request_option(request, "top_k", self.config.get("top_k", 20)))
        steps = int(
            request_option(
                request,
                "num_inference_steps",
                self.config.get("num_inference_steps", 10),
            )
        )

        start_time = time.perf_counter()
        reference_images = [
            process_image(ref_image, tar_width, tar_height)
            for ref_image in reference_pils
        ]
        query_patches = process_image_q(query_origin, tar_width, tar_height)
        reference_patches = []
        for reference_image in reference_images:
            reference_patches += process_image_ref(
                reference_image, tar_width, tar_height
            )

        with torch.no_grad():
            query_clip = image_processor(
                images=query_patches, return_tensors="pt"
            ).pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
            query_embeddings = image_encoder(query_clip).image_embeds
            ref_clip = image_processor(
                images=[image.convert("RGB") for image in reference_patches],
                return_tensors="pt",
            ).pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
            reference_embeddings = image_encoder(ref_clip).image_embeds
            similarities = torch.nn.functional.cosine_similarity(
                query_embeddings.unsqueeze(1),
                reference_embeddings.unsqueeze(0),
                dim=-1,
            )
            sorted_indices = torch.argsort(
                similarities, descending=True, dim=1
            ).tolist()
            available_ref_patches = [[], [], [], []]
            for i, sorted_list in enumerate(sorted_indices):
                for index in sorted_list[:top_k]:
                    available_ref_patches[i].append(
                        reference_patches[index]
                        .resize((tar_width // 2, tar_height // 2))
                        .convert("RGB")
                    )

            generator = torch.Generator(device="cuda").manual_seed(seed)
            with _working_directory(repo_path):
                colorized = pipeline(
                    cond_input=query_bw,
                    cond_refs=available_ref_patches,
                    hint_mask=hint_mask.resize(
                        (tar_width // 8, tar_height // 8)
                    ).convert("RGB"),
                    hint_color=hint_color.convert("RGB"),
                    num_inference_steps=steps,
                    generator=generator,
                )[0][0]

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            up_img = colorized.resize(query_vae.size)
            test_low_color = (
                transform(up_img).unsqueeze(0).to("cuda", dtype=torch.float16)
            )
            query_vae_tensor = (
                transform(query_vae).unsqueeze(0).to("cuda", dtype=torch.float16)
            )
            h_color, hidden_color = pipeline.vae._encode(
                test_low_color, return_dict=False, hidden_flag=True
            )
            _, hidden_bw = pipeline.vae._encode(
                query_vae_tensor, return_dict=False, hidden_flag=True
            )
            hidden_double = [
                torch.cat((hidden_color[idx], hidden_bw[idx]), dim=1)
                for idx in range(len(hidden_color))
            ]
            hidden = multi_res_net(hidden_double)
            output = pipeline.vae._decode(
                h_color.sample(), return_dict=False, hidden_list=hidden
            )[0]
            output = output.clamp(-1, 1)
            high_res = Image.fromarray(
                ((output[0] * 0.5 + 0.5).permute(1, 2, 0).detach().cpu().numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            ).convert("RGB")

        torch.cuda.empty_cache()
        return result(
            image=np.asarray(high_res.resize(source.size)),
            model_id=self.model_id,
            start_time=start_time,
            metadata={
                "style": style,
                "seed": seed,
                "top_k": top_k,
                "num_inference_steps": steps,
                "target_width": tar_width,
                "target_height": tar_height,
                "max_side": max_side,
                "high_res_scale": high_res_scale,
            },
        )
