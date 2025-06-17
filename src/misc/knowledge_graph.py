from __future__ import annotations

"""Scaffold for a tag-and-alias knowledge-graph that supports
    * stable UUID identifiers
    * alias management with hit-counts
    * DAG edges with typed relationships (e.g., prerequisite, feature)
    * versioned, read-only snapshots
    * plug-in points for large-language-model (LLM) helpers

The intent is to keep the predictable CRUD plumbing in hand-written
code and leave the fuzzy judgment calls—like discovering new tags or
relationships—to an external LLM accessed through *stub* functions that
can later be wired to OpenAI, Anthropic, etc.
"""

import json
import time
import uuid
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# ↓↓↓                    Data-model primitives                       ↓↓↓
# ---------------------------------------------------------------------------

TagId = uuid.UUID
EdgeType = str  # e.g. "prerequisite", "feature", "subcomponent"


@dataclass
class Tag:
    """A single skill/concept node in the DAG."""

    id: TagId
    name: str  # canonical display name (mutable by alias promotion)
    description: str = ""

    # Alias → hit-count, kept *ordered* by descending hits so alias[0] is
    # the preferred surface name.  The key at alias[0] == name  by design.
    aliases: "OrderedDict[str, int]" = field(default_factory=lambda: OrderedDict())

    # Immediate edges; graph keeps the global picture.
    parents: Set[TagId] = field(default_factory=set)  # things *this* depends on
    children: Set[TagId] = field(default_factory=set)  # things that depend on *this*

    # Lightweight metadata for curriculum planners / schedulers.
    created_ts: float = field(default_factory=time.time)
    schema_version: int = 1

    def add_alias(self, alias: str):
        alias_lc = alias.strip().lower()
        self.aliases.setdefault(alias_lc, 0)

    def record_hit(self, alias: str):
        alias_lc = alias.strip().lower()
        self.aliases[alias_lc] = self.aliases.get(alias_lc, 0) + 1
        # Promote most-popular alias to canonical name
        self.aliases = OrderedDict(sorted(self.aliases.items(), key=lambda kv: -kv[1]))
        most_popular, _ = next(iter(self.aliases.items()))
        self.name = most_popular  # surfacing only; ID remains stable


@dataclass
class Edge:
    """Typed, weighted, *directed* edge in the knowledge DAG."""

    parent: TagId
    child: TagId
    etype: EdgeType = "prerequisite"
    weight: float = 1.0  # can encode strength / frequency
    created_ts: float = field(default_factory=time.time)
    schema_version: int = 1


# ---------------------------------------------------------------------------
# ↓↓↓                    Graph container + API                      ↓↓↓
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """In-memory DAG with alias resolution and versioned snapshots."""

    def __init__(self):
        self._tags: Dict[TagId, Tag] = {}
        self._alias_index: Dict[str, TagId] = {}  # alias(str) -> tag_id
        self._children: Dict[TagId, Set[TagId]] = defaultdict(set)
        self._parents: Dict[TagId, Set[TagId]] = defaultdict(set)

        # simple monotonically increasing snapshot marker
        self._snapshot_counter: int = 0

    # ---------------------  Tag CRUD  ---------------------

    def add_tag(self, name: str, description: str = "") -> TagId:
        alias_norm = name.strip().lower()
        if alias_norm in self._alias_index:
            return self._alias_index[alias_norm]

        tag_id = uuid.uuid4()
        tag = Tag(id=tag_id, name=alias_norm, description=description)
        tag.add_alias(alias_norm)

        self._tags[tag_id] = tag
        self._alias_index[alias_norm] = tag_id
        return tag_id

    def add_alias(self, tag_id: TagId, alias: str):
        alias_norm = alias.strip().lower()
        if alias_norm in self._alias_index:
            raise ValueError(f"Alias '{alias}' already bound to tag {self._alias_index[alias_norm]}")
        tag = self._tags[tag_id]
        tag.add_alias(alias_norm)
        self._alias_index[alias_norm] = tag_id

    # ---------------------  Edge CRUD  ---------------------

    def add_edge(self, parent: TagId, child: TagId, etype: EdgeType = "prerequisite", weight: float = 1.0):
        if parent == child:
            raise ValueError("Self-loops not allowed in DAG")
        self._children[parent].add(child)
        self._parents[child].add(parent)
        # we’re not storing Edge objects separately for now; weights go in adjacency list

    # ---------------------  Queries  ---------------------

    def resolve(self, alias_or_name: str) -> Optional[TagId]:
        return self._alias_index.get(alias_or_name.strip().lower())

    def parents(self, tag_id: TagId) -> Set[TagId]:
        return self._parents.get(tag_id, set())

    def children(self, tag_id: TagId) -> Set[TagId]:
        return self._children.get(tag_id, set())

    def downstream_closure(self, tag_ids: Set[TagId]) -> Set[TagId]:
        """All descendants reachable from *tag_ids* (inclusive)."""
        seen: Set[TagId] = set()
        stack = list(tag_ids)
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            stack.extend(self._children.get(tid, ()))
        return seen

    # ---------------------  Snapshots  ---------------------

    def snapshot(self, path: Path) -> str:
        """Write a JSONLines snapshot; returns version hash (simple counter)."""
        version = f"v{self._snapshot_counter}"
        self._snapshot_counter += 1
        out = {"version": version,
               "tags": [self._tag_to_json(t) for t in self._tags.values()],
               "edges": [(str(p), str(c)) for p, cs in self._children.items() for c in cs]}
        path.write_text(json.dumps(out, indent=2))
        return version

    @staticmethod
    def _tag_to_json(tag: Tag) -> Mapping:
        return {
            "id": str(tag.id),
            "name": tag.name,
            "description": tag.description,
            "aliases": list(tag.aliases.items()),
            "parents": [str(p) for p in tag.parents],
            "children": [str(c) for c in tag.children],
            "created_ts": tag.created_ts,
        }

    # ---------------------  Pretty print helpers  ---------------------

    def describe(self, tag_id: TagId, depth: int = 1):
        tag = self._tags[tag_id]
        print(f"{tag.name}  (id={tag_id})")
        if depth >= 1:
            print("  parents:")
            for p in tag.parents:
                print("   •", self._tags[p].name)
            print("  children:")
            for c in tag.children:
                print("   •", self._tags[c].name)

# ---------------------------------------------------------------------------
# ↓↓↓          STUBS: LLM-dependent fuzzy operations                   ↓↓↓
# ---------------------------------------------------------------------------

LLMClient = object  # placeholder for whichever API wrapper you prefer


def call_llm(client: LLMClient, prompt: str, **kwargs) -> str:
    """Central place to invoke the external language model.
    Replace with an OpenAI/Anthropic call.  Keeps the rest of the codebase
    agnostic.  In unit tests you can monkey-patch this to return canned values.
    """
    raise NotImplementedError("hook up your favourite provider here")


def llm_tag_example(client: LLMClient, example_text: str, graph: KnowledgeGraph) -> List[str]:
    """Ask the LLM for a set of tag *aliases* that fit *example_text*."""
    prompt = (
        "You are a classifier mapping examples to *existing* skill tags.\n"
        "Here is the text:\n" + example_text + "\n"
        "Return a comma-separated list of the 5–10 most relevant existing tags."
    )
    response = call_llm(client, prompt)
    return [alias.strip().lower() for alias in response.split(",") if alias.strip()]


def llm_suggest_new_tags(client: LLMClient, example_text: str, graph: KnowledgeGraph) -> List[Tuple[str, str]]:
    """Let the LLM propose truly *new* tags (alias + short description)."""
    prompt = (
        "You may invent *new* skill tags if needed.\n"
        "Provide up to 3 new tags in JSON: [{'name': str, 'description': str}]\n\n"
        "Example:\n[{'name': 'tail-recursion', 'description': '...'}]\n\n"
        "Input example:\n" + example_text
    )
    try:
        raw = call_llm(client, prompt)
        return json.loads(raw)
    except Exception:
        return []


def llm_infer_prereqs(client: LLMClient, tag_name: str, graph: KnowledgeGraph) -> List[str]:
    """Given a tag, ask LLM to guess immediate prerequisites by *alias*."""
    prompt = (
        f"Tag: {tag_name}\n"
        "List 3–5 immediate prerequisites (aliases) in a comma-separated line."
    )
    response = call_llm(client, prompt)
    return [a.strip().lower() for a in response.split(",") if a.strip()]


# ---------------------------------------------------------------------------
# ↓↓↓                    Demo / sandbox usage                        ↓↓↓
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kg = KnowledgeGraph()

    # Hand-add a couple of seeds
    tid_math = kg.add_tag("basic-arithmetic", "addition, subtraction, ...")
    tid_frac = kg.add_tag("fractions", "ratio of two integers")
    kg.add_edge(parent=tid_math, child=tid_frac)

    kg.describe(tid_frac)

    # (LLM calls remain NotImplemented; patch them out in tests.)
