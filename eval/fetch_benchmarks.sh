#!/bin/bash
# Fetch the public benchmark data used by eval/ into ./bench (or $BENCH).  Pinned revisions, so scores are comparable.
#   Intern-Decision bundle (accuracy suites + calibration pilot + scorer)  InternLM/Intern-Decision @ 2f81580
#   JevBench scorer and public tiers                                        fstandhartinger/jevbench @ 7ce310c7
#   Typed Decisions                                                         LocalLLaMA/typed-decisions @ f7a2487e
# The Decision Index suite is not redistributable; build it with the kit (see docs/evaluation.md).
set -e
BENCH=${BENCH:-$(cd "$(dirname "$0")/.." && pwd)/bench}
mkdir -p "$BENCH" && cd "$BENCH"
[ -d intern-decision ] || git clone -q https://github.com/InternLM/Intern-Decision.git intern-decision
(cd intern-decision && git checkout -q 2f81580)
[ -d jevbench ] || git clone -q https://github.com/fstandhartinger/jevbench.git jevbench
(cd jevbench && git checkout -q 7ce310c7)
python -c "from huggingface_hub import snapshot_download; snapshot_download('LocalLLaMA/typed-decisions', repo_type='dataset', revision='f7a2487e', local_dir='typed_decisions', allow_patterns=['all/*'])"
ls "$BENCH"
