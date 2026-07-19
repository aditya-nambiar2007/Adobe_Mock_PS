# Adobe Mock PS Backend

Prompt-based image editing via natural language. Upload an image, describe the edit, and the backend orchestrates Gemini for planning + local AI models (SAM, InstructPix2Pix) for execution.

```bash
# One-shot example
curl -X POST http://localhost:8000/upload -F "file=@photo.jpg"
# → {"filename":"upload_abc123.png",...}

curl -X POST http://localhost:8000/edit \
  -H "Content-Type: application/json" \
  -d '{"prompt":"remove the person and make it cyberpunk","image":"upload_abc123.png"}'
# → {"job_id":"...","status":"completed","steps":[...],"final_image":"<base64>"}
```

---

## Quick Start

```bash
# 1. Environment
cd backend
python3 -m venv venv && source venv/bin/activate

# 2. Dependencies (CPU-only)
pip install -r requirements.txt

# 3. Dependencies (PyTorch & Torchvision) — install AFTER requirements.txt
# Pick the command corresponding to your system/CUDA version:
# CUDA 12.4 (Recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.1 (Highly stable fallback):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8 (For older CUDA setups):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU Only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Default (Pip system default):
pip install torch torchvision

# 4. Configure
cp .env.example .env
# Edit .env → set GEMINI_API_KEY="your-key"

# 5. Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | `python3 --version` |
| **pip** | `python3 -m pip --version` |
| **Gemini API key** | Get one free at [aistudio.google.com](https://aistudio.google.com/apikey) |
| **NVIDIA GPU** (optional) | For faster model inference; CPU works but is slower |
| **CUDA 12.4+** (if GPU) | `nvidia-smi` to check driver version |

### GPU Setup (RTX 2050 / 4 GB VRAM tested)

The 4 GB VRAM on the RTX 2050 is tight but sufficient with these settings:

- **SAM** runs on **CPU** (saves VRAM for diffusion)
- **InstructPix2Pix** / **Stable Diffusion Inpainting** loads in **float16** on GPU
- Models are **unloaded** between pipeline steps
- **SAM mask is passed to the diffusion pipeline** for guided inpainting — the mask from the `segment` step is consumed by the subsequent `remove` or `replace_background` step to constrain regeneration to the relevant area instead of a blind full-image img2img pass

```bash
# Install CUDA 12.4 PyTorch (compatible with driver 550.xx)
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# → CUDA: True
```

---

## Full API Reference

### `GET /health`

Server status and capability check.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "cuda_available": true,
  "device": "cuda",
  "gemini_configured": true,
  "uptime_seconds": 42.5
}
```

---

### `POST /upload`

Upload an image file (multipart/form-data). Supported formats: JPEG, PNG, WebP, BMP. Max size: 10 MB (configurable).

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@photo.jpg"
```

```json
{
  "filename": "upload_a1b2c3d4e5f6.png",
  "original_name": "photo.jpg",
  "size_bytes": 284720,
  "width": 1920,
  "height": 1080,
  "content_type": "image/jpeg"
}
```

Save the `filename` — you'll use it in the `/edit` request.

---

### `POST /edit`

Execute an editing prompt on a previously uploaded image.

```bash
curl -X POST http://localhost:8000/edit \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "remove the person on the left and make the scene cyberpunk at night with neon lights",
    "image": "upload_a1b2c3d4e5f6.png"
  }'
```

**How it works:**

1. Gemini decomposes the prompt into a structured plan (e.g. `segment → remove → change_style`)
2. The pipeline executes each operation sequentially, passing a **context dict** between steps
3. The `segment` step stores the SAM-generated mask in the context; `remove` / `replace_background` consume it to constrain diffusion to the relevant area (masked inpainting, not blind full-image img2img)
4. Each step produces an intermediate image (returned as base64)
5. The final image is the output of the last step

**Response:**

```json
{
  "job_id": "f6e5d4c3b2a1",
  "status": "completed",
  "steps": [
    {
      "operation": "segment",
      "image": "iVBORw0KGgoAAAANSUhEUgAAAA...",
      "duration_ms": 9405.12,
      "details": { "mask": "temp/mask_abc.png" }
    },
    {
      "operation": "remove",
      "image": "iVBORw0KGgoAAAANSUhEUgAAAA...",
      "duration_ms": 14423.87,
      "details": null
    },
    {
      "operation": "change_style",
      "image": "iVBORw0KGgoAAAANSUhEUgAAAA...",
      "duration_ms": 13586.45,
      "details": null
    }
  ],
  "final_image": "iVBORw0KGgoAAAANSUhEUgAAAA...",
  "total_duration_ms": 37685.44,
  "execution_log": [
    {
      "operation": "segment",
      "target": "person",
      "model": "sam-vit-base",
      "parameters": { "target": "person" },
      "status": "success",
      "reason": "",
      "duration": 9405.12
    },
    {
      "operation": "remove",
      "target": "",
      "model": "sd-inpaint",
      "parameters": {},
      "status": "success",
      "reason": "",
      "duration": 14423.87
    }
  ],
  "explanation": {
    "plain_english": "The person was identified and removed from the image. The background was preserved. The overall colors were then adjusted to a cyberpunk style with neon tones.",
    "technical_summary": "- Person segmented using SAM.\n- Object removed via inpainting.\n- Style transferred via InstructPix2Pix.",
    "changes": [
      {
        "operation": "segment",
        "target": "person",
        "description": "The person was identified and separated.",
        "technical": "Segmented using SAM (ViT-B)."
      },
      {
        "operation": "remove",
        "target": "",
        "description": "The person was removed from the image.",
        "technical": "Object removed via inpainting."
      }
    ]
  },
  "error": null
}
```

**Decoding base64 images (JavaScript):**

```javascript
const response = await fetch('http://localhost:8000/edit', { /* ... */ });
const data = await response.json();

// Each image is a base64-encoded PNG
data.steps.forEach(step => {
  const img = document.createElement('img');
  img.src = `data:image/png;base64,${step.image}`;
  document.body.appendChild(img);
});

// Or final image
const finalImg = document.querySelector('#result');
finalImg.src = `data:image/png;base64,${data.final_image}`;
```

**Decoding base64 images (Python):**

```python
import base64
from PIL import Image
import io

# Save a step image to disk
img_data = base64.b64decode(response["steps"][0]["image"])
img = Image.open(io.BytesIO(img_data))
img.save("step_0_segment.png")
```

**If the request takes too long**, the HTTP connection may time out. You can:

- Increase the timeout: `curl --max-time 300 ...`
- Or use the async polling approach below

---

### `GET /result/{job_id}`

Retrieve a completed or failed job result. Use this for long-running edits.

```bash
JOB_ID=$(curl -s -X POST http://localhost:8000/edit \
  -H "Content-Type: application/json" \
  -d '{"prompt":"make it cyberpunk","image":"upload_abc.png"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

# Poll until complete
sleep 20
curl http://localhost:8000/result/$JOB_ID
```

Same response schema as `POST /edit`.

---

### `GET /history/{job_id}`

Job metadata (status, prompt, input image, progress).

```bash
curl http://localhost:8000/history/$JOB_ID
```

```json
{
  "job_id": "f6e5d4c3b2a1",
  "status": "completed",
  "created_at": "2026-07-07T18:25:50+00:00",
  "progress": {
    "prompt": "remove the person and make it cyberpunk",
    "input_image": "upload_a1b2c3d4e5f6.png",
    "steps_count": 3,
    "error": null
  }
}
```

---

## Editing Prompt Guide

The Gemini planner maps natural language to structured operations. Here's how prompts are interpreted:

| Prompt | Generated Plan |
|---|---|
| "remove the person" | `[{"operation":"segment","target":"person"}, {"operation":"remove"}]` |
| "make it cyberpunk" | `[{"operation":"change_style","instruction":"make the scene cyberpunk"}]` |
| "change the background to a beach" | `[{"operation":"replace_background","instruction":"change the background to a beach"}]` |
| "upscale the image" | `[{"operation":"upscale"}]` |
| "remove the car and make it rainy night" | `[{"operation":"segment","target":"car"}, {"operation":"remove"}, {"operation":"change_style","instruction":"make it rainy night"}]` |

**Tips for good results:**

- Be specific about what to segment / remove / change
- Combine operations naturally (Gemini splits them automatically)
- For style changes, describe the target style clearly
- Image size affects inference time; 512×512 is optimal

---

## Supported Operations

| Operation | Model | Description |
|---|---|---|---|
| `segment` | SAM (ViT-Base) | Segments an object by target name; generates a mask (stored in pipeline context) |
| `remove` | InstructPix2Pix / **SD Inpainting** | Removes the main subject. Uses **masked inpainting** when SAM mask is available from a prior `segment` step; falls back to InstructPix2Pix img2img |
| `replace_background` | InstructPix2Pix / **SD Inpainting** | Replaces image background. **Inverts** the SAM mask to inpaint only the background area when a mask is available |
| `remove_background` | rembg | Makes the background transparent. Portraits use `u2net_human_seg` for finer hair and clothing edges; other images use `isnet-general-use`. |
| `change_style` / `style_transfer` | InstructPix2Pix | Applies artistic style transformation |
| `upscale` | PIL Bicubic (4×) | Upscales the image (ESRGAN when available) |

---

## Post-Edit Explanation System

Every completed edit returns an `execution_log` and `explanation` alongside the edited image. This ensures users always know exactly what the AI did and why.

### Execution Log

The `ExecutionLog` records every pipeline step with these fields:

| Field | Type | Description |
|---|---|---|
| `operation` | string | Operation name (`segment`, `remove`, `replace_background`, etc.) |
| `target` | string | Target object (e.g. `"person"`, `"background"`) |
| `model` | string | AI model used (`sam-vit-base`, `sd-inpaint`, `instruct-pix2pix`, etc.) |
| `parameters` | object | Parameters passed to the operation |
| `status` | string | `"success"`, `"skipped"`, or `"failed"` |
| `reason` | string | Why an operation was skipped or failed |
| `duration` | float | Execution time in milliseconds |

Log entries are appended by the pipeline executor at each step. This is the **single source of truth** — explanations are always derived from the log, never from the original prompt.

### Explanation Service

Located at `app/services/explanation/`. Architecture:

```
ExplanationService
├── generate(prompt, execution_log, metadata) → ExplanationResult
│
├── [provider] ExplanationProvider (ABC)
│   ├── GeminiExplanationProvider    # LLM-backed (default when GEMINI_API_KEY is set)
│   └── LocalExplanationProvider     # Deterministic fallback (no LLM needed)
│
└── prompts.py                       # All LLM prompt templates (isolated)
```

**ExplanationResult** contains:

| Field | Description |
|---|---|
| `plain_english` | 2-3 sentence plain-English explanation of what actually happened |
| `technical_summary` | Bullet-point summary for advanced users |
| `changes[]` | Per-operation descriptions with `operation`, `target`, `description`, `technical` |

### Provider Selection

The `get_explanation_service()` dependency in `dependencies.py` selects the provider:

1. If `GEMINI_API_KEY` is configured → uses `GeminiExplanationProvider`
2. Otherwise → uses `LocalExplanationProvider` (deterministic, always works)

A local LLM can be swapped in later by implementing the `ExplanationProvider` ABC.

### Design Rules

- **Log is truth** — The explanation generator receives the execution log and must describe only what the log contains
- **No inference** — Never infer operations that don't exist in the log
- **Status-aware** — Successful operations are described; skipped/failed operations are mentioned accurately (never falsely described as successful)
- **Deterministic fallback** — Without an LLM, the local provider produces accurate explanations directly from the log
- **UI-friendly** — The `plain_english` field is designed to be shown directly to end users

### Example

Input execution log:

```
segment → person → success
replace_background → beach → success
upscale → skipped (image already high resolution)
```

Local provider produces:
> "The person was separated from the original background and placed onto a beach scene. Image upscaling was not performed."

Technical summary:
> ```
> - Person segmented using SAM.
> - Background replaced via inpainting.
> - upscale skipped.
> ```

### Unit Tests

```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

Tests verify:
- Skipped operations are never described as successful
- Failed operations are never described as successful
- Changes list accurately reflects the execution log
- Empty/all-skipped/all-failed logs produce correct output

---

## Configuration

All settings via `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | `""` | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini model name |
| `DEVICE` | `auto` | Override: `cuda`, `cpu`, `mps` |
| `OUTPUT_DIRECTORY` | `outputs` | Edited images directory |
| `UPLOAD_DIRECTORY` | `uploads` | Uploaded images directory |
| `TEMP_DIRECTORY` | `temp` | Temporary files directory |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max upload file size |
| `REQUEST_TIMEOUT_SECONDS` | `300` | Request timeout |
| `MODEL_CACHE_TIMEOUT_MINUTES` | `30` | How long to keep models in GPU |
| `DIFFUSION_LORA_WEIGHTS` | `""` | Path to LoRA adapter weights (`.safetensors` or directory); auto-applied on diffusion load |
| `DIFFUSION_LORA_ADAPTER_NAME` | `default` | Adapter name for the loaded LoRA weights |
| `LOG_LEVEL` | `INFO` | Logging: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CORS_ORIGINS` | `*` | CORS allowed origins (comma-separated) |

---

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐
│  Client  │ → │  POST /edit  │ → │    Gemini   │ → │  Pipeline Executor│
│ (curl/UI)│    │              │    │   Planner   │    │     (context)     │
└──────────┘    └──────────────┘    └─────────────┘    └────────┬─────────┘
                  ↓                      ↑                       │
               Response                  Plan               ExecutionLog
                  ↓                      │                       │
            ┌──────────┐                 └───────────────────────┘
            │Explanation│                                        │
            │  Service  │ ← Execution Log (single source of truth)
            │ (LLM/Local)│                                        │
            └─────┬─────┘                                        │
                  ↓                                              │
            ExplanationResult                              Steps (images)
            (plain_english,                               ┌─────────────┐
             technical_summary,     ← ← ← ← ← ← ← ← ← ← ←│SAM (CPU)    │──→ mask ─┐
             changes[])                                    └─────────────┘          │
                  ↓                                        ┌──────────────────────┘
            ┌──────────┐                                   ▼
            │  Client  │                              ┌──────────────────────────┐
            │ Response │                              │SD Inpaint (GPU, float16) │
            │  (JSON)  │                              │ (mask-guided inpainting) │
            └──────────┘                              └──────────────────────────┘
                                                      ┌──────────────────────────┐
                                                      │InstructPix2Pix (GPU)     │
                                                      │ (style transfer, fallback)│
                                                      └──────────────────────────┘
                                                      ┌────────┐
                                                      │ ESRGAN │
                                                      └────────┘
```

**Key design decisions:**

- **Model Manager singleton** — Only one heavy model in GPU at a time. Models are loaded lazily and unloaded before loading the next. This keeps VRAM usage under control.
- **Context-passing between steps** — The pipeline carries a `context` dict. `segment` stores the SAM mask; `remove` / `replace_background` consume it to switch from blind img2img to mask-guided inpainting (`StableDiffusionInpaintPipeline`). Background replacement inverts the mask (inpaint background, keep foreground).
- **LoRA adapter support** — Diffusion models automatically load LoRA adapter weights when `DIFFUSION_LORA_WEIGHTS` is set in `.env`. Adapters are applied via `diffusers.load_lora_weights()` after the base model loads.
- **Modular services** — Each model (SAM, diffusion, ESRGAN) has its own service class. Adding a new model means creating a new service and registering a loader.
- **Planner validation** — Gemini's output is validated against a strict schema before execution. Invalid operations are caught early.

### Request Flow

1. **Upload** → Image is validated, saved to `uploads/`, metadata returned
2. **Edit** → Prompt + image filename sent to `/edit`
3. **Gemini** → Prompt is sent to Gemini API with a system prompt asking for JSON output
4. **Plan** → Gemini's JSON response is parsed and validated by the planner
5. **Pipeline** → Each operation is executed in sequence:
   - Current model is loaded (previous model is unloaded)
   - Operation runs on the current image
   - Result image is saved and base64-encoded
   - Image is passed to the next operation
6. **Response** → All step images + final image returned as JSON

---

## Folder Structure

```
backend/
├── app/
│   ├── main.py                    # FastAPI app, CORS, lifespan, error handlers
│   ├── config.py                  # Pydantic Settings from .env
│   ├── dependencies.py            # FastAPI dependency injection
│   │
│   ├── api/
│   │   ├── health.py              # GET /health
│   │   ├── upload.py              # POST /upload
│   │   └── edit.py                # POST /edit, GET /result/{id}, GET /history/{id}
│   │
│   ├── schemas/
│   │   ├── requests.py            # EditRequest, UploadResponse
│   │   └── responses.py           # EditResponse, StepInfo, HealthResponse, etc.
│   │
│   ├── services/
│   │   ├── gemini_service.py      # google-genai SDK wrapper, GeminiError exception
│   │   ├── planner.py             # Validates Gemini JSON → structured plan
│   │   ├── model_manager.py       # Singleton: lazy load, cache, GPU management
│   │   ├── diffusion_service.py   # InstructPix2Pix / SD Inpaint wrappers + LoRA adapter loader
│   │   ├── sam_service.py         # SAM segmentation (ViT-Base, runs on CPU)
│   │   ├── esrgan_service.py      # ESRGAN upscaling (with PIL fallback)
│   │   ├── execution_log.py       # ExecutionLog dataclass (single source of truth for all pipeline ops)
│   │   ├── pipeline.py            # Operation handler + pipeline executor (context-passing between steps)
│   │   └── explanation/           # Post-edit explanation service
│   │       ├── __init__.py        # Exports: ExplanationService, ExplanationProvider, ExplanationResult
│   │       ├── interfaces.py      # ABC + data classes (ChangeDescription, ExplanationResult)
│   │       ├── service.py         # ExplanationService with fallback
│   │       ├── prompts.py         # Isolated LLM prompt templates
│   │       ├── gemini_provider.py # LLM-backed explanation generation
│   │       └── local_provider.py  # Deterministic fallback (no LLM)
│   │
│   └── utils/
│       ├── image_utils.py         # load/save/convert/base64 helpers
│       ├── gpu.py                 # CUDA detection, memory management (no-torch fallback)
│       └── logger.py              # Structured logging with timestamps
│
├── tests/                          # Pytest test suite
│   ├── test_execution_log.py       # ExecutionLog filtering & serialization
│   └── test_explanation.py         # LocalExplanationProvider reliability
├── uploads/                       # Uploaded images
├── outputs/                       # Edited images (organized by job_id)
├── temp/                          # Temporary masks and intermediates
├── requirements.txt
├── .env.example
└── README.md
```

---

## Error Handling

| HTTP | Code | When |
|---|---|---|
| 400 | `INVALID_IMAGE_TYPE` | Uploaded file is not JPEG/PNG/WebP/BMP |
| 400 | `INVALID_IMAGE` | Uploaded file is corrupted or not an image |
| 404 | `IMAGE_NOT_FOUND` | Referenced upload `filename` doesn't exist |
| 404 | `JOB_NOT_FOUND` | Job ID not found |
| 413 | `FILE_TOO_LARGE` | Upload exceeds `MAX_UPLOAD_SIZE_MB` |
| 429 | `GEMINI_QUOTA_EXCEEDED` | Gemini API free tier quota exhausted |
| 503 | `GEMINI_NOT_CONFIGURED` | No `GEMINI_API_KEY` in `.env` |
| 500 | `INTERNAL_ERROR` | Unexpected server error (check logs) |

All error responses follow this shape:

```json
{
  "error_code": "GEMINI_QUOTA_EXCEEDED",
  "detail": "Gemini API quota exceeded for 'gemini-3.1-flash-lite'. Wait for reset or use a different key.",
  "suggestion": "Check your Gemini API key and billing status"
}
```

---

## Frontend Integration Guide

### From a web app (JavaScript)

```javascript
async function editImage(file, prompt) {
  // 1. Upload
  const formData = new FormData();
  formData.append('file', file);
  const uploadRes = await fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: formData,
  });
  const { filename } = await uploadRes.json();

  // 2. Edit
  const editRes = await fetch('http://localhost:8000/edit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, image: filename }),
  });
  const data = await editRes.json();

  // 3. Display results
  const container = document.getElementById('steps');
  data.steps.forEach((step, i) => {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${step.image}`;
    img.alt = `Step ${i + 1}: ${step.operation}`;
    container.appendChild(img);
  });

  return data;
}
```

### From Python

```python
import requests
import base64
from PIL import Image
import io

BASE_URL = "http://localhost:8000"

# Upload
with open("photo.jpg", "rb") as f:
    upload = requests.post(f"{BASE_URL}/upload", files={"file": f})
filename = upload.json()["filename"]

# Edit
edit = requests.post(f"{BASE_URL}/edit", json={
    "prompt": "remove the person and make it cyberpunk",
    "image": filename,
})
data = edit.json()

# Save all step images
for i, step in enumerate(data["steps"]):
    img_data = base64.b64decode(step["image"])
    img = Image.open(io.BytesIO(img_data))
    img.save(f"step_{i}_{step['operation']}.png")

print(f"Done in {data['total_duration_ms']:.0f}ms")
```

---

## Model Details

| Model | ID | Size | VRAM | Runs On |
|---|---|---|---|---|---|
| SAM ViT-Base | `facebook/sam-vit-base` | 358 MB | — (CPU) | CPU |
| InstructPix2Pix | `timbrooks/instruct-pix2pix` | 2.9 GB | ~3.2 GB | GPU (float16) |
| Stable Diffusion v1.5 | `runwayml/stable-diffusion-v1-5` | 4.3 GB | ~4.5 GB | GPU (float16) |

Models are downloaded from HuggingFace Hub on first use and cached in `~/.cache/huggingface/hub/`.

### Model Pre-Installation (Optional)

To avoid network delays or request timeouts during the first run, you can pre-install and download all backend models beforehand:

#### Option 1: Via CLI Script
Activate your virtual environment and run the downloader:
```bash
python -m app.utils.download_models
```

#### Option 2: Via API Endpoint
While the server is running, trigger model download programmatically:
- **Trigger Download (asynchronous)**: `POST http://localhost:8000/models/install`
- **Check Status / Cache Status**: `GET http://localhost:8000/models/status`

### LoRA Adapters

Drop a LoRA `.safetensors` file anywhere and point to it in `.env`:

```env
DIFFUSION_LORA_WEIGHTS="/path/to/remove-bg-lora.safetensors"
DIFFUSION_LORA_ADAPTER_NAME="remove_bg"
```

The adapter is auto-applied every time the diffusion model loads. Call `unload_lora()` from code to remove it per-session. Multiple adapters can be loaded by calling `load_lora()` multiple times with different names; use `set_adapters()` from diffusers to blend them.

---

## Running Tests

```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

The test suite covers:

| Test file | What it tests |
|---|---|
| `tests/test_execution_log.py` | `ExecutionLog` append, filtering (success/skipped/failed), serialization |
| `tests/test_explanation.py` | `LocalExplanationProvider` — skipped/failed ops never falsely described, changes match log, edge cases |

---

## Extending the Backend

### Add a new operation

```python
# In app/services/pipeline.py

class OperationHandler:
    def handle_new_effect(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ):
        # Read artifacts from context (e.g. SAM mask from prior segment step)
        mask = context.get("mask") if context else None
        # Your implementation
        return result_image, {}

# Register it
self._operation_map["new_effect"] = self.handler.handle_new_effect
```

### Add a new model

1. Create `app/services/new_model_service.py` with a class that has `load_model()` and processing methods
2. Add a `load_new_model()` method to `ModelManager` in `model_manager.py`
3. Register the model type in `ModelType` enum
4. Use it from a pipeline handler

### Swap a model

Change the model ID in the service class constant, or set `MODEL_PATHS` in `.env`:

```env
MODEL_PATHS='{"sam": "my-org/my-sam","diffusion": "my-org/my-diffusion"}'
```

---

## Running in Production

```bash
# With gunicorn for process management
pip install gunicorn
gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 --timeout 300
```

Consider adding:
- **Redis/Celery** for background job processing
- **PostgreSQL** for persistent job storage
- **S3/GCS** for image storage instead of local filesystem
- **Rate limiting** via `slowapi`
- **Prometheus metrics** via `starlette-exporter`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA not available` | Install CUDA-compatible PyTorch: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` (or fallback to `cu121` / `cu118` index if `cu124` fails) |
| `CUDA out of memory` | SAM runs on CPU by default. If diffusion OOMs, set `DEVICE=cpu` in `.env` |
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install torch torchvision` |
| `FileNotFoundError: LoRA weights not found` | Check `DIFFUSION_LORA_WEIGHTS` path in `.env` |
| `Gemini API quota exceeded` | Wait for daily reset or use a different API key |
| `Image not found` | Upload the image first via `POST /upload`, use the returned `filename` |
| First portrait background removal is slow | `u2net_human_seg` is downloaded by rembg once (about 170 MB); later requests use the cached model. |
| Server won't start | Check `uvicorn` log for errors. Common: port in use (`fuser -k 8000/tcp`), missing `.env` |

---

## Tech Stack

- **Python 3.11+** with modern typing
- **FastAPI** for REST API
- **google-genai** for Gemini API integration
- **PyTorch** for model inference
- **HuggingFace Transformers** (SAM)
- **HuggingFace Diffusers** (InstructPix2Pix)
- **Pillow / OpenCV** for image processing
- **Pydantic v2** for data validation
- **httpx** for async HTTP
