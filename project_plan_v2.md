# CS-503 Project Plan — Auxiliary Evidence Routing for Egocentric VQA

## Team
- **Christina** (394672)
- **Clémentine** (347046)
- **Eren** (389941)
- **Ivan** (394235)

## Submitted Proposal Summary

We study whether **targeted auxiliary evidence** improves egocentric VQA on top of a frozen VLM's native video input. Using HD-EPIC, we compare video-only baseline against video + auxiliary evidence tools (CLIP retrieval, motion keyframes, hand/object evidence, crop/OCR refinement). If different question categories benefit from different tools, we build oracle and predicted **tool routers**.

**Two-stage design:**
- **Stage 1 — Ablation:** native video only vs. native video + higher FPS vs. native video + each tool
- **Stage 2 — Routing (conditional on Stage 1):** oracle routing by category, predicted routing by question text

**VLM:** Qwen3-VL-8B-Instruct (frozen)
**Dataset:** HD-EPIC VQA subset (≤10 min clips, ~3–5K questions)
**Metric:** Multiple-choice accuracy, per-category breakdown

---

## Timeline & Task Ownership

**Budget:** 4 people × 4 weeks × 8h/week = 128 person-hours

### Week 1 — Infrastructure & Pipeline (32h)

| Person | Task | Hours |
|--------|------|-------|
| **Vanya (Ivan)** | Project scaffold, data loading, VLM inference pipeline, evaluation harness | 8h |
| **Eren** | HD-EPIC dataset access: register CodaLab, download VQA questions, identify video subset, extract clips | 8h |
| **Christina** | Tool 1: CLIP/SigLIP similarity retrieval implementation | 8h |
| **Clémentine** | Tool 2: Motion/scene-change keyframe selection implementation | 8h |

### Week 2 — Tools & Baseline Runs (32h)

| Person | Task | Hours |
|--------|------|-------|
| **Vanya** | Run baseline experiments (video-only, video + higher FPS), debug pipeline issues | 8h |
| **Eren** | Tool 3: Hand/object-centric evidence (hand detector or interaction-heavy moment selection) | 8h |
| **Christina** | Tool 4: Crop/local-detail refinement (high-res crops around small objects / text regions) | 8h |
| **Clémentine** | Help run tool ablations, start per-category accuracy tracking | 8h |

### Week 3 — Stage 1 Ablation + Stage 2 Routing (32h)

| Person | Task | Hours |
|--------|------|-------|
| **Vanya** | Run all tool ablation experiments at scale, log-prob extraction | 8h |
| **Eren** | Oracle routing implementation + analysis | 8h |
| **Christina** | Predicted routing (lightweight classifier on question text embeddings) | 8h |
| **Clémentine** | Agreement analysis, per-category/per-prototype breakdowns | 8h |

### Week 4 — Analysis, Write-up, Presentation (32h)

| Person | Task | Hours |
|--------|------|-------|
| **Vanya** | Ablation experiments (vary K, frame budget), help with write-up | 8h |
| **Eren** | Project webpage (GitHub Pages) | 8h |
| **Christina** | Final presentation slides (4 min) | 8h |
| **Clémentine** | Qualitative analysis, figures, write-up, individual contributions | 8h |

---

## Deadlines

| Date | Deliverable |
|------|------------|
| ~~Apr 13~~ | ~~Project proposal~~ ✅ submitted |
| **May 04** | Progress report (2 pages) + Midterm slides |
| **May 05/07** | Midterm presentation (3 min) |
| **May 25** | Final presentation slides |
| **May 26/28** | Final presentation (4 min) |
| **May 29** | Project webpage + code zip |

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Auxiliary evidence doesn't help | That's a valid finding — "native VLM already captures relevant cues" is a meaningful negative result per proposal |
| HD-EPIC download too large | Start with 1-2 kitchens, filter to short clips (<30s). Even 500 questions gives per-category signal |
| Qwen3-VL too slow | Use Qwen3-VL-2B as fast iteration model, 8B for final numbers |
| Tool doesn't differentiate from uniform extra frames | The uniform-higher-FPS baseline is specifically designed to test this |

---

## Vanya's Task: Pipeline Setup — Claude Code Instructions

See the next section for the exact prompt to give to Claude Code.

---

# Claude Code Instructions: Project Pipeline Setup

Copy everything below this line and give it to Claude Code as a single prompt.

---

```
## Task

Set up the complete project scaffold for our CS-503 Visual Intelligence course project. This is an egocentric video QA project that tests whether auxiliary evidence tools improve a frozen VLM (Qwen3-VL) on the HD-EPIC benchmark.

## Requirements

### 1. Project structure

Create a Python project using `uv` for dependency management. The project should be called `ego-evidence-routing`. Use the following structure:

```
ego-evidence-routing/
├── pyproject.toml              # uv project config
├── README.md                   # setup instructions for the team
├── configs/
│   └── default.yaml            # all hyperparams in one place
├── src/
│   └── eer/                    # "ego evidence routing" package
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── hdepic.py       # HD-EPIC VQA dataset loader
│       │   └── frames.py       # frame extraction from video clips
│       ├── vlm/
│       │   ├── __init__.py
│       │   └── qwen.py         # Qwen3-VL inference wrapper
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py         # abstract Tool interface
│       │   ├── uniform.py      # uniform frame sampling (baseline)
│       │   ├── clip_retrieval.py   # CLIP/SigLIP text-image retrieval
│       │   ├── motion.py       # motion/scene-change keyframes
│       │   ├── hand.py         # hand/object interaction evidence
│       │   └── crop.py         # local-detail crop refinement
│       ├── routing/
│       │   ├── __init__.py
│       │   ├── oracle.py       # oracle router (best tool per category)
│       │   └── predicted.py    # predicted router (classifier on question text)
│       ├── eval/
│       │   ├── __init__.py
│       │   └── metrics.py      # accuracy, per-category, per-prototype
│       └── utils/
│           ├── __init__.py
│           └── logging.py      # structured logging setup
├── scripts/
│   ├── run_baseline.py         # run video-only baseline
│   ├── run_ablation.py         # run all tools, one at a time
│   ├── run_routing.py          # run oracle + predicted routing
│   └── analyze_results.py      # produce tables and plots
└── tests/
    ├── test_data.py            # test dataset loading
    ├── test_vlm.py             # test VLM inference on a single example
    └── test_tools.py           # test each tool produces valid frame indices
```

### 2. Dependencies (pyproject.toml)

Use `uv` and specify these dependencies:

```toml
[project]
name = "ego-evidence-routing"
version = "0.1.0"
description = "Auxiliary evidence routing for egocentric video QA (CS-503)"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "transformers>=4.57",
    "accelerate",
    "qwen-vl-utils[decord]>=0.0.14",
    "open-clip-torch",
    "decord",
    "Pillow",
    "pandas",
    "numpy",
    "pyyaml",
    "tqdm",
    "scikit-learn",
    "matplotlib",
    "seaborn",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "ipython"]
```

### 3. Config file (configs/default.yaml)

```yaml
model:
  name: "Qwen/Qwen3-VL-8B-Instruct"
  dtype: "bfloat16"
  device_map: "auto"

data:
  vqa_questions_path: "data/hdepic/vqa_questions.json"
  video_clips_dir: "data/hdepic/clips/"
  candidate_fps: 1  # extract candidate frames at this FPS
  max_clip_duration_s: 600  # only use clips <= 10 minutes

tools:
  frame_budget: 8  # number of auxiliary frames each tool selects
  clip_model: "ViT-SO400M-14-SigLIP"
  clip_pretrained: "webli"

eval:
  results_dir: "results/"
  val_split_ratio: 0.3  # for routing train/test split
```

### 4. Key implementations to write in full

#### 4a. `src/eer/data/hdepic.py`
- Define a `HDEPICDataset` class that loads the VQA questions JSON/CSV
- Each item should be a dict with: `question_id`, `video_id`, `start_time`, `end_time`, `question`, `choices` (list of 5 strings), `correct_answer` (letter A-E), `category` (one of 7), `prototype` (one of 30)
- Provide `filter_by_duration(max_seconds)` method
- Provide `filter_by_categories(categories: list[str])` method
- Provide `split(val_ratio)` → (train, val) method
- Use dataclasses for the question items
- For now, use a placeholder JSON structure since we don't have the exact HD-EPIC format yet — make it easy to adapt once Eren provides the real data. Include a `from_csv` and `from_json` classmethod.

#### 4b. `src/eer/data/frames.py`
- `extract_candidate_frames(video_path: str | Path, fps: float = 1.0) -> list[Frame]` using decord
- `Frame` is a dataclass with `index: int`, `timestamp_s: float`, `image: PIL.Image.Image`
- Cache extracted frames to disk as JPGs with a manifest CSV, and load from cache if available
- Handle gracefully if video file doesn't exist (log warning, return empty list)

#### 4c. `src/eer/vlm/qwen.py`
- `QwenVLM` class that wraps Qwen3-VL inference
- `__init__(self, model_name, dtype, device_map)` — loads model + processor once
- Key method: `answer_vqa(self, video_path: str | None, auxiliary_frames: list[PIL.Image.Image] | None, question: str, choices: list[str]) -> VLMResult`
- `VLMResult` is a dataclass: `predicted_letter: str`, `log_probs: dict[str, float]` (log-prob for each choice letter A-E), `raw_output: str`
- The method should:
  1. Build a Qwen3-VL message with the video (if provided) AND the auxiliary frames (if provided) as additional images
  2. Append the multiple-choice prompt: format the question + 5 lettered choices + "Answer with the letter only."
  3. Run generation with `max_new_tokens=16`
  4. Extract the predicted letter from the output
  5. Extract log-probabilities of the answer tokens (A/B/C/D/E) from the generation logits
- Support three modes: (a) video only, (b) video + auxiliary frames, (c) video at higher native FPS (no auxiliary frames, just pass more frames to the model)
- For mixed video+image input in Qwen3-VL, build the message content list with a `{"type": "video", "video": path}` entry followed by multiple `{"type": "image", "image": pil_image}` entries

#### 4d. `src/eer/tools/base.py`
- Abstract base class `EvidenceTool`:
  ```python
  from abc import ABC, abstractmethod

  class EvidenceTool(ABC):
      """Base class for auxiliary evidence selection tools."""

      @abstractmethod
      def name(self) -> str: ...

      @abstractmethod
      def select(self, candidate_frames: list[Frame], question: str, budget: int = 8) -> list[Frame]:
          """Select `budget` frames from candidates. Return selected frames."""
          ...
  ```

#### 4e. `src/eer/tools/uniform.py`
- Implement `UniformTool(EvidenceTool)`: evenly-spaced K frames from candidates

#### 4f. `src/eer/tools/clip_retrieval.py`
- Implement `CLIPRetrievalTool(EvidenceTool)`:
  - Load SigLIP model in `__init__` via open_clip
  - `select()`: embed all candidate frame images + question text, return top-K by cosine similarity
  - Sort returned frames by timestamp (temporal order)

#### 4g. `src/eer/tools/motion.py`
- Implement `MotionTool(EvidenceTool)`:
  - Compute pixel-wise L1 difference between consecutive frames
  - Score each frame by max(diff_with_prev, diff_with_next)
  - Return top-K highest-motion frames, sorted by timestamp

#### 4h. `src/eer/eval/metrics.py`
- `compute_accuracy(results: pd.DataFrame) -> dict`: overall accuracy
- `compute_per_category_accuracy(results: pd.DataFrame) -> pd.DataFrame`: accuracy grouped by category
- `compute_per_prototype_accuracy(results: pd.DataFrame) -> pd.DataFrame`: accuracy grouped by prototype
- `compute_oracle_routing(results: pd.DataFrame) -> dict`: for each question, pick the tool that got it right → compute upper-bound accuracy
- `compute_agreement(results: pd.DataFrame) -> pd.DataFrame`: for each question, count how many tools agree, compute accuracy per agreement level
- Results DataFrame expected columns: `question_id, category, prototype, tool, predicted, correct, log_prob, is_correct`

#### 4i. `scripts/run_baseline.py`
- CLI script using argparse
- Loads config, dataset, VLM
- Runs video-only baseline on the filtered dataset
- Saves results to CSV in `results/baseline_{timestamp}.csv`

#### 4j. `scripts/run_ablation.py`
- CLI script
- For each question × each tool: select auxiliary frames → run VLM with video + aux frames → record result
- Support `--tools` flag to run specific tools (e.g., `--tools uniform clip motion`)
- Support `--limit N` to run on first N questions only (for debugging)
- Save results to `results/ablation_{timestamp}.csv`

### 5. Code quality requirements

- Use type hints everywhere
- Use `logging` module (not print statements) via the `src/eer/utils/logging.py` setup
- Use `pathlib.Path` not string paths
- Use dataclasses for data containers
- Keep functions small and single-purpose
- Add docstrings to all public classes and methods
- Make sure `ruff check` passes (use ruff defaults)
- Write the 3 test files with basic sanity checks (they can use mock data)

### 6. README.md

Write a README with:
- Project description (2 sentences)
- Setup instructions using uv:
  ```bash
  # Clone and setup
  git clone <repo-url>
  cd ego-evidence-routing
  uv sync
  uv sync --extra dev  # for development tools

  # Download data (Eren will fill this in)
  # ...

  # Run tests
  uv run pytest tests/

  # Run baseline
  uv run python scripts/run_baseline.py --config configs/default.yaml --limit 10

  # Run ablation
  uv run python scripts/run_ablation.py --config configs/default.yaml --tools uniform clip --limit 10
  ```
- Brief description of the project structure
- How to add a new tool (implement EvidenceTool, register in run_ablation.py)

### 7. What NOT to implement yet

- `hand.py` and `crop.py` tools — leave as stubs with `raise NotImplementedError`. Christina and Eren will fill these in during Week 2.
- `routing/oracle.py` and `routing/predicted.py` — leave as stubs. These come in Week 3.
- Don't worry about the exact HD-EPIC JSON format — make the data loader adaptable with classmethods.

### 8. Important Qwen3-VL details

- Qwen3-VL requires `transformers>=4.57` (currently install from git: `pip install git+https://github.com/huggingface/transformers`)
- In pyproject.toml, use the git dependency for transformers if v4.57 is not yet on PyPI
- The model class is `Qwen3VLForConditionalGeneration` (NOT `Qwen2_5_VLForConditionalGeneration`)
- The processor is `AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")`
- For video input: `{"type": "video", "video": "/path/to/video.mp4"}`
- For image input: `{"type": "image", "image": pil_image_object}`
- For mixed input (our key use case): put video first, then images, then text in the content list
- Use `processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")`
- For log-prob extraction: use `model.generate(..., return_dict_in_generate=True, output_scores=True)` and extract logits for answer tokens
- qwen-vl-utils version must be >=0.0.14 for Qwen3-VL compatibility. Use `image_patch_size=16` (not 14 which is for Qwen2.5-VL)

Now please generate all the files. Make sure the code is complete and runnable (except for the intentional stubs). Prioritize correctness and clarity over cleverness.
```
