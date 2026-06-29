# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SkillRL memory subsystem. `skills_only_memory` / `skill_updater` carry the
# NPU embedding fallback (ported from the AReaL adaptation); the rest are the
# verl-agent originals (SimpleMemory / SearchMemory / RetrievalMemory) used by
# EnvironmentManagerBase subclasses.
from .memory import SimpleMemory, SearchMemory
from .skills_only_memory import SkillsOnlyMemory
from .skill_updater import SkillUpdater
from .base import BaseMemory

# RetrievalMemory (legacy faiss-indexed memory) is optional — it requires
# faiss-cpu, which the skills-only path does not need. Import lazily so the
# package loads without faiss installed.
try:
    from .retrieval_memory import RetrievalMemory
except ImportError:
    RetrievalMemory = None  # type: ignore[assignment]
