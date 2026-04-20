#!/usr/bin/env python3
"""Test the generalized prompt on all challenge datasets."""
import json
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load real credentials from auth.json (not the masked .env)
auth_path = Path.home() / '.hermes' / 'auth.json'
api_key = ''
base_url = 'https://api.minimax.io'
if auth_path.exists():
    with open(auth_path) as f:
        auth = json.load(f)
    creds = auth.get('credential_pool', {}).get('minimax', [])
    if creds:
        api_key = creds[0].get('access_token', '')
        base_url = creds[0].get('base_url', base_url)

# Fallback to .env if auth.json key doesn't work
from pathlib import Path as P2
env_path = P2.home() / '.hermes' / '.env'
if env_path.exists():
    raw = env_path.read_bytes().decode('utf-8', errors='replace')
    for line in raw.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if 'MINIMAX_API_KEY' in line and '=' in line:
            key = line.split('=', 1)[1].strip()
            key = ''.join(c for c in key if c.isprintable()).strip()
            if key and not key.startswith('***'):
                api_key = key
                break

print(f"API key: {'loaded' if api_key else 'MISSING'} ({api_key[:10]}...)")

# Load the reasoner
import importlib.util
spec = importlib.util.spec_from_file_location(
    "iterative_reasoner_fresh",
    Path(__file__).parent / "iterative_reasoner.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
IterativeReasoner = module.IterativeReasoner
from burst import Burst  # noqa: E402

reasoner = IterativeReasoner(api_key=api_key, model="MiniMax-M2")

# Ground truth session counts (from manual analysis)
GROUND_TRUTH = {
    "challenge_long_haul":       14,   # 14 sessions in long_haul
    "challenge_meeting_day":      9,   # 9 sessions in meeting_day
    "challenge_parallel_tornado": 9,   # 9 sessions in parallel_tornado
    "challenge_rabbit_hole":      8,   # 8 sessions in rabbit_hole
}

# Ground truth session boundaries (for manual inspection)
# Format: (start_burst, end_burst, description)
GROUND_TRUTH_DETAIL = {
    "challenge_long_haul": [
        (0, 2, "paymentService write 08:00-08:10"),
        (3, 3, "types/payment.ts write 08:30"),
        (4, 4, "paymentService write 09:00"),
        (5, 5, "Terminal npm install 09:30"),
        (6, 8, "paymentService debugging 10:30-10:50"),
        (9, 9, "Chrome Stripe docs read 11:00"),
        (10, 10, "paymentService debugging 11:20"),
        (11, 11, "paymentService debugging 12:00"),
        (12, 12, "paymentService debugging 12:30"),
        (13, 13, "Terminal tests 13:30"),
        (14, 15, "paymentService debugging 13:35-13:50"),
        (16, 16, "Terminal curl test 14:10"),
        (17, 17, "paymentService write 15:30"),
        (18, 19, "PR #445 commit/review 15:45-16:00"),
    ],
    "challenge_meeting_day": [
        (0, 2, "user-prefs.ts write 08:00-08:30"),
        (3, 5, "standup meeting 09:00-09:10"),
        (6, 8, "sprint planning 10:30-11:30"),
        (9, 11, "Slack sync 13:00-13:30"),
        (12, 12, "PR #447 review read 14:30"),
        (13, 13, "avatarService write 14:45"),
        (14, 14, "PR #447 review read 15:00"),
        (15, 17, "user-prefs.ts write 16:00-16:40"),
    ],
    "challenge_parallel_tornado": [
        (0, 1, "api-gateway + auth/middleware write 09:00-09:02"),
        (2, 2, "Slack #backend coord 09:04"),
        (3, 4, "api-gateway + auth/middleware write 09:06-09:08"),
        (5, 5, "PR #441 review read 09:10"),
        (6, 6, "api-gateway write 09:12"),
        (7, 7, "Slack #backend coord 09:14"),
        (8, 9, "auth/middleware write + Terminal git 09:16-09:18"),
        (10, 10, "README write 09:20"),
        (11, 11, "PR #441 review read 09:22"),
        (12, 12, "api-gateway write 09:24"),
    ],
    "challenge_rabbit_hole": [
        (0, 4, "paymentService webhook debug 14:00-14:15"),
        (5, 5, "paymentService webhook debug 14:20"),
        (6, 8, "userService write 14:30-14:37"),
        (9, 9, "paymentService webhook debug 14:45"),
        (10, 10, "Chrome Stripe docs debug 14:50"),
        (11, 11, "paymentService webhook debug 14:55"),
        (12, 12, "Terminal test 15:00"),
        (13, 13, "paymentService webhook debug 15:10"),
    ],
}

def load_bursts(name):
    """Load challenge JSON and convert to Burst objects."""
    with open(Path(__file__).parent / "data" / f"{name}.json") as f:
        raw = json.load(f)
    
    bursts = []
    for r in raw:
        bursts.append(Burst(
            timestamp=r["timestamp"],
            window_id=r.get("window_id", r["window_title"]),
            window_title=r["window_title"],
            app_name=r["app_name"],
            app_path=r.get("app_path", r["app_name"]),
            chars=r["chars"],
            char_count=len(r["chars"]),
            source=r.get("source", "test"),
            focus_active=r.get("focus_active", True),
        ))
    return bursts

def count_sessions(sessions):
    """Count valid sessions (filter out orphans)."""
    return len([s for s in sessions if len(s.burst_ids) > 1 or (len(s.burst_ids) == 1 and s.confidence != "low")])

def compare_sessions(name, result_sessions, detail=False):
    """Compare produced sessions against ground truth."""
    gt = GROUND_TRUTH_DETAIL[name]
    gt_count = len(gt)
    produced = sorted(result_sessions, key=lambda s: s.start_time)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Ground truth: {gt_count} sessions")
    print(f"Produced:     {len(produced)} sessions")
    print(f"Match:        {'✓' if len(produced) == gt_count else '✗ MISMATCH'}")
    print(f"-{'-'*60}")
    
    for i, sess in enumerate(produced):
        burst_range = f"[{sess.burst_ids[0]}-{sess.burst_ids[-1]}]" if sess.burst_ids else "[?] "
        print(f"  {burst_range} {sess.start_time}-{sess.end_time} | {sess.app_name[:12]:12s} | {sess.topic[:40]:40s}")
        print(f"            | {sess.narrative[:80] if sess.narrative else '(no narrative)'}")
        print(f"            | conf={sess.confidence} type={sess.work_type}")
    
    if detail:
        print(f"\n--- Ground Truth ---")
        for start, end, desc in gt:
            print(f"  [{start}-{end}] {desc}")
    
    return len(produced)

def main():
    results = {}
    for name in sorted(GROUND_TRUTH.keys()):
        print(f"\n>>> Processing {name}...")
        bursts = load_bursts(name)
        print(f"    {len(bursts)} bursts")
        
        try:
            result = reasoner.reason_day(bursts)
            n = compare_sessions(name, result["sessions"], detail=False)
            results[name] = {"produced": n, "expected": GROUND_TRUTH[name], "match": n == GROUND_TRUTH[name]}
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {"error": str(e)}
        
        time.sleep(1)  # Rate limit
    
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name}: ERROR — {r['error'][:60]}")
        else:
            status = "✓" if r["match"] else "✗"
            print(f"  {name}: {r['produced']}/{r['expected']} sessions {status}")
    
    total_match = sum(1 for r in results.values() if "error" not in r and r["match"])
    print(f"\n  Total: {total_match}/{len(results)} challenges match")

if __name__ == "__main__":
    main()
