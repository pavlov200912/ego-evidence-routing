# ego-evidence-routing

Auxiliary evidence routing for long egocentric video question answering.

This project tests whether targeted external evidence helps a frozen video-native VLM answer HD-EPIC multiple-choice questions. The model can receive either the native video alone, tool-selected evidence alone, or native video augmented with tool-selected evidence. The main experiment compares these conditions across question categories and evidence budgets.

## Current Experiment

The main runner is:

```bash
scripts/run_hdepic_ablation.py
```

It evaluates HD-EPIC questions in two modes:

```text
replace   selected evidence only
augment   native video + selected evidence
```

The Slurm wrapper runs both modes back to back:

```bash
scripts/sbatch_hdepic_ablation.sh
```

## Setup

Use a Python environment with Python 3.11/3.12 and a PyTorch build compatible with the cluster CUDA driver. On the Izar V100 nodes, CUDA 12.1 wheels are the safer choice than CUDA 13 wheels.

```bash
cd /path/to/ego-evidence-routing

# Example only: use the environment agreed by the team/cluster.
source /path/to/env/bin/activate

export PYTHONPATH=src
export SCRATCH_ROOT=/scratch/izar/<username>
export HF_HOME=$SCRATCH_ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_XET_CACHE=$HF_HOME/xet
export TMPDIR=$SCRATCH_ROOT/tmp
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE" "$TMPDIR"
```

`pyproject.toml` is the preferred dependency description. `requirements.txt` may contain environment-specific pins and should be treated with care on the cluster.

## Data Paths

The HD-EPIC runner expects:

```text
HD-EPIC videos:
  /scratch/izar/<username>/hd-epic/HD-EPIC/Videos

HD-EPIC VQA annotations:
  /scratch/izar/<username>/hd-epic/hd-epic-annotations/vqa-benchmark
```

Check `configs/hdepic.yaml` before running:

```yaml
data:
  vqa_benchmark_dir: "/scratch/izar/<username>/hd-epic/hd-epic-annotations/vqa-benchmark"
  video_clips_dir: "/scratch/izar/<username>/hd-epic/HD-EPIC/Videos"
```

## Evidence Tools

The HD-EPIC runner currently supports:

```text
qwen_native           native video only baseline
uniform               uniformly sampled auxiliary frames
clip                  CLIP/SigLIP question-image retrieval
motion_then_clip      motion over-selection followed by CLIP refinement
ocr_crop              OCR-guided high-resolution text/detail crops
hand                  hand-object interaction evidence
object_tracking       GroundingDINO object-centric frame selection
uniform+clip          union of temporal coverage and semantic retrieval
answer_guided_oracle  answer-choice time-window oracle/control
```

All non-native tools use the same evidence budget from `--budget`. If a tool returns too many evidence items, the runner deduplicates and caps it to that shared budget. The native baseline `qwen_native` is skipped in replacement mode because it would be identical to the augmentation-mode native-video baseline.

## Running HD-EPIC Ablations

Small nutrition pilot with four lighter conditions:

```bash
sbatch scripts/sbatch_hdepic_ablation.sh \
  --category nutrition_video_nutrition_estimation \
  --budget 8 \
  --tools qwen_native,uniform,clip,ocr_crop
```

Same category with all default tools:

```bash
sbatch scripts/sbatch_hdepic_ablation.sh \
  --category nutrition_video_nutrition_estimation \
  --budget 8
```

Run another budget:

```bash
sbatch scripts/sbatch_hdepic_ablation.sh \
  --category nutrition_video_nutrition_estimation \
  --budget 16 \
  --tools qwen_native,uniform,clip,ocr_crop
```

Run directly without Slurm for a tiny sanity check:

```bash
PYTHONPATH=src python scripts/run_hdepic_ablation.py \
  --config configs/hdepic.yaml \
  --category recipe_step_localization \
  --limit 2 \
  --budget 8 \
  --tools qwen_native,uniform,clip,ocr_crop \
  --augment
```

## HD-EPIC Categories

The progress report focuses on these groups:

```text
Recipes
Ingredients
Object Motion
Nutrition, partially
```

Useful category stems include:

```text
recipe_step_localization
recipe_rough_step_localization
recipe_prep_localization
recipe_recipe_recognition
ingredient_ingredient_adding_localization
ingredient_ingredient_recognition
ingredient_exact_ingredient_recognition
ingredient_ingredients_order
object_motion_object_movement_counting
object_motion_object_movement_itinerary
object_motion_stationary_object_localization
nutrition_video_nutrition_estimation
nutrition_nutrition_change
```

`nutrition_image_nutrition_estimation` is image-based and does not fit the current video runner.

## Results

HD-EPIC ablation results are saved by category and budget:

```text
results/hdepic/<category>/replace_k<budget>.csv
results/hdepic/<category>/augment_k<budget>.csv
```

Example:

```text
results/hdepic/nutrition_video_nutrition_estimation/replace_k8.csv
results/hdepic/nutrition_video_nutrition_estimation/augment_k8.csv
```

Each CSV row is one question-tool result:

```text
question_id
category
mode
tool
requested_budget
n_selected_raw
n_selected_final
predicted
correct
is_correct
```

`n_selected_raw` is how many evidence items the tool returned before capping. `n_selected_final` is how many evidence items were actually passed to Qwen.

Slurm logs are saved under:

```text
logs/hdepic_ablation_<jobid>.out
logs/hdepic_ablation_<jobid>.err
logs/hdepic_ablation_<jobid>_replace.log
logs/hdepic_ablation_<jobid>_augment.log
```

## EgoTextVQA Diagnostics

EgoTextVQA is used as a short-video diagnostic benchmark for OCR and crop tools, not as the main long-video experiment.

```bash
PYTHONPATH=src python scripts/run_egotextvqa_baseline.py \
  --config configs/egotextvqa_indoor.yaml \
  --tool ocr_crop \
  --limit 10 \
  --budget 8 \
  --fps 1.0 \
  --vlm-fps 1.0 \
  --save-collages
```

## Project Structure

```text
configs/       Experiment configs and data paths
scripts/       HD-EPIC/EgoTextVQA runners, Slurm wrappers, analysis helpers
src/eer/data/  Dataset loaders and frame extraction
src/eer/tools/ Evidence selection tools
src/eer/vlm/   Qwen3-VL wrapper and prompt construction
src/eer/eval/  Metrics
src/eer/routing/ Oracle and predicted routing utilities
tests/         Unit and integration tests
```

## Adding A Tool

Implement `EvidenceTool` in `src/eer/tools/`. Tools should return `Frame` objects in timestamp order and accept optional metadata via `**kwargs`:

```python
from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

class MyTool(EvidenceTool):
    @property
    def name(self) -> str:
        return "my_tool"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
        **kwargs,
    ) -> list[Frame]:
        return candidate_frames[:budget]
```

Register the tool in `scripts/run_hdepic_ablation.py` by adding it to `_TOOL_NAMES`, `_DEFAULT_TOOL_NAMES` if it should run by default, and `_build_tools()`.
