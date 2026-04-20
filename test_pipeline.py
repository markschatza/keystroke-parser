#!/usr/bin/env python3
"""Standalone test of the iterative reasoner pipeline."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.home() / "keystroke-parser"))

# Fresh import - read directly from disk
import importlib.util
spec = importlib.util.spec_from_file_location(
    "iterative_reasoner_fresh",
    Path.home() / "keystroke-parser" / "iterative_reasoner.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

IterativeReasoner = module.IterativeReasoner
load_api_key = module.load_api_key

from burst import load_bursts_from_json

api_key = load_api_key()
print(f"API key: {'loaded' if api_key else 'MISSING'}")

reasoner = IterativeReasoner(api_key=api_key, model="MiniMax-M2")
data_dir = Path.home() / "keystroke-parser" / "data"
bursts = load_bursts_from_json(data_dir / "difficult_day.json")
print(f"Loaded {len(bursts)} bursts")

sorted_bursts = sorted(bursts, key=lambda b: b.timestamp)
burst_refs = reasoner._build_burst_refs(sorted_bursts)
print(f"Built {len(burst_refs)} burst refs")

# Test Pass 1
gr = reasoner._pass1_grouping(burst_refs, sorted_bursts)
print(f"Pass 1 produced {len(gr['sessions'])} sessions")

if len(gr['sessions']) == 51:
    raw_sessions = gr['sessions']
    # Test Pass 2
    labeled, tokens = reasoner._pass2_labeling(raw_sessions, sorted_bursts, burst_refs)
    print(f"Pass 2 produced {len(labeled)} labeled sessions, tokens={tokens}")
    
    # Test Pass 3
    summary, patterns, ptokens = reasoner._pass3_synthesis(labeled, sorted_bursts)
    print(f"Pass 3 summary: {summary[:100]}...")
    print(f"Pass 3 patterns: {patterns[:100] if patterns else 'none'}...")
    
    # Full pipeline
    result = reasoner.reason_day(bursts)
    print(f"\n=== FULL PIPELINE ===")
    print(f"Sessions: {len(result['sessions'])}")
    print(f"API calls: {result['total_api_calls']}")
    high_conf = [s for s in result["sessions"] if s.confidence == "high"]
    print(f"High confidence: {len(high_conf)}")
    for s in high_conf:
        print(f"  [{s.start_time}] {s.app_name} | {s.topic}")
        print(f"    {s.narrative[:100]}")
else:
    print("ERROR: Pass 1 did not produce 51 sessions!")
    # Debug: check what _call returns
    print(f"DEBUG: num_chunks in gr: {gr.get('num_chunks')}")
