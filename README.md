# ego-evidence-routing

Auxiliary evidence routing for egocentric video question answering (CS-503 Visual Intelligence).
We test whether specialised frame-selection tools (CLIP retrieval, motion keyframes, hand-interaction detectors) improve a frozen Qwen3-VL-8B model on the HD-EPIC benchmark compared to a video-only baseline.

## Setup

```bash
# Clone and install
git clone <repo-url>
cd ego-evidence-routing
uv sync
uv sync --extra dev   # adds pytest, ruff, ipython

# Download data (Eren will fill this in)
# Place files under data/hdepic/vqa_questions.json and data/hdepic/clips/

# Run tests (no GPU needed)
uv run pytest tests/

# Run video-only baseline
uv run python scripts/run_baseline.py --config configs/default.yaml --limit 10

# Run ablation (all tools)
uv run python scripts/run_ablation.py --config configs/default.yaml --limit 10

# Run ablation (specific tools)
uv run python scripts/run_ablation.py --config configs/default.yaml \
    --tools uniform clip --limit 10

# Analyse results
uv run python scripts/analyze_results.py \
    --ablation-csv results/ablation_<timestamp>.csv \
    --output-dir results/analysis/
```

## Project structure

```
configs/          Hyperparameters (model, data paths, tool settings)
src/eer/
  data/           HD-EPIC dataset loader + decord frame extractor
  vlm/            Qwen3-VL inference wrapper
  tools/          Evidence-selection tools (uniform, clip, motion, …)
  routing/        Oracle and predicted routers (Week 3)
  eval/           Accuracy metrics and aggregation helpers
  utils/          Logging setup
scripts/          CLI entry points (baseline, ablation, routing, analysis)
tests/            Unit tests (no GPU or real data required)
```

## Adding a new tool

1. Create `src/eer/tools/my_tool.py` and implement `EvidenceTool`:

```python
from eer.tools.base import EvidenceTool
from eer.data.frames import Frame

class MyTool(EvidenceTool):
    @property
    def name(self) -> str:
        return "my_tool"

    def select(self, candidate_frames: list[Frame], question: str, budget: int = 8) -> list[Frame]:
        # ... your selection logic ...
        return selected_frames
```

2. Register it in `scripts/run_ablation.py`:

```python
from eer.tools.my_tool import MyTool

_TOOL_REGISTRY: dict[str, type[EvidenceTool]] = {
    ...
    "my_tool": MyTool,
}
```

3. Run: `uv run python scripts/run_ablation.py --tools my_tool --limit 10`

