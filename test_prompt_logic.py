#!/usr/bin/env python3
"""
Deterministic rule-based evaluator for the sessionization logic.
Mimics what the LLM prompt should do, so we can verify the logic without an API call.
"""
import json
from datetime import datetime
from pathlib import Path

# Ground truth session counts (from manual analysis)
GROUND_TRUTH = {
    "challenge_long_haul":       14,
    "challenge_meeting_day":      8,
    "challenge_parallel_tornado": 10,
    "challenge_rabbit_hole":      8,
}

# Ground truth session boundaries (burst index ranges)
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

def infer_task(burst, prev_burst=None, next_burst=None):
    """Infer the semantic task from a burst's content and window.
    
    Uses surrounding context to resolve ambiguity — mimics how the LLM
    sees the full picture rather than just isolated burst data.
    """
    chars = burst['chars'].lower()
    window = burst['window_title'].lower()
    app = burst['app_name'].lower()
    
    # === Meeting patterns ===
    meeting_windows = {'standup', 'meet.google.com', 'zoom.us', 'huddle', 'slack huddle'}
    if any(x in window for x in meeting_windows):
        if 'standup' in window:
            return 'standup_meeting'
        if 'zoom' in window or 'meet.google' in window:
            return 'sprint_planning'
        if 'huddle' in window or ('slack' in app and 'huddle' in window):
            return 'slack_sync'
    
    # === Terminal: check chars content, not just path ===
    if app == 'terminal':
        if 'test' in chars or 'npm test' in chars:
            return 'testing'
        if 'curl' in chars:
            return 'testing'
        if 'git commit' in chars or 'git add' in chars:
            return 'git_commit'
        if 'npm install' in chars:
            return 'npm_install'
        return 'terminal_task'
    
    # === File-based task identification ===
    # Extract file or feature area from window
    parts = window.replace('\\', '/').split('/')
    filename = parts[-1] if parts else window
    
    # Strip common path prefixes
    for prefix in ['services/', 'feature/', 'api-gateway/', 'auth/']:
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
    
    # Extract feature name without extension
    if '.' in filename:
        feature = filename.rsplit('.', 1)[0]
    else:
        feature = filename
    
    # === Context-aware disambiguation ===
    # If a user-prefs window mentions PR #447 but user is TYPING in it,
    # it's user-prefs work, not PR review. Only reading PR comments = PR review.
    if feature == 'user-prefs' or feature == 'user-prefs.ts':
        # If the chars are actual code (not "reviewing"), it's prefs work
        if any(kw in chars for kw in ['export', 'async', 'function', 'await', 'const', 'db.query', 'UPDATE']):
            return 'user_prefs_feature'
    
    # === PR review: reading comments, not writing code ===
    if 'pr #' in window or 'github.com/pr' in window:
        # If the content being typed looks like code review (not just code)
        if any(kw in chars for kw in ['lgtm', 'looks good', 'left detailed', 'review', 'comment', 'addressed']):
            return 'pr_review'
        # If it's opening/viewing, it's reading
        if 'opening' in chars.lower() or chars.strip() == '':
            return 'pr_review'
    
    # === Reading docs (part of debugging/investigation) ===
    # These are usually Chrome with docs in the URL or window
    if any(x in window for x in ['docs', 'documentation', 'stripe.com', 'dashboard.stripe']):
        return 'reading_docs'
    
    return feature

def feature_area(window):
    """Extract the feature area from a window title.
    
    e.g. 'services/paymentService.ts' → 'payment'
        'api-gateway/server.ts' → 'api-gateway'
        'auth/middleware.ts' → 'auth'  (cross-file with api-gateway = same feature)
    """
    parts = window.replace('\\', '/').split('/')
    filename = parts[-1] if parts else window
    # Strip extension
    if '.' in filename:
        filename = filename.rsplit('.', 1)[0]
    
    # Map files to their feature area
    feature_map = {
        'paymentService': 'payment',
        'payment': 'payment',
        'userService': 'user',
        'user': 'user',
        'user-prefs': 'user-prefs',
        'user-prefs.ts': 'user-prefs',
        'server': 'api-gateway',   # api-gateway/server.ts = auth feature
        'middleware': 'auth',       # auth/middleware.ts = auth feature
        'api-gateway': 'api-gateway',
        'auth': 'auth',
    }
    
    return feature_map.get(filename, filename)


def same_task(b1, b2, gap_min):
    """
    Determine if two bursts belong to the SAME task.
    Returns True if they should be merged into one session.
    
    Same task means:
    - Same feature area (e.g., paymentService.ts and types/payment.ts = payment)
    - Or: same abstract debugging/investigation goal even across different files/apps
    """
    w1 = b1['window_title'].lower()
    w2 = b2['window_title'].lower()
    a1 = b1['app_name'].lower()
    a2 = b2['app_name'].lower()
    c1 = b1['chars'].lower()
    c2 = b2['chars'].lower()
    
    f1 = feature_area(b1['window_title'])
    f2 = feature_area(b2['window_title'])
    
    # Same feature area → same task
    if f1 == f2 and f1 not in ('unknown', 'readme', 'terminal_task'):
        return True
    
    # Reading docs about a feature while debugging that feature = same task
    # e.g., Chrome Stripe dashboard while debugging paymentService webhook
    debug_features = {'payment', 'user', 'auth', 'api-gateway'}
    if f1 in debug_features and f2 in debug_features:
        # Reading docs for a feature you're debugging = same task
        if ('stripe.com' in w2 or 'dashboard.stripe' in w2 or 'docs' in w2) and gap_min < 10:
            return True
        if ('stripe.com' in w1 or 'dashboard.stripe' in w1 or 'docs' in w1) and gap_min < 10:
            return True
    
    # PR review: reading PR comments + writing code for that PR = same task
    # e.g., PR #441 in Chrome + api-gateway code changes in VS Code
    if 'pr #' in w1 or 'github.com/pr' in w1:
        if 'pr #' in w2 or 'github.com/pr' in w2:
            return True
    if 'pr #' in w2 or 'github.com/pr' in w2:
        if 'pr #' in w1 or 'github.com/pr' in w1:
            return True
    
    return False


def is_quick_coordination(burst, prev_burst, gap_min):
    """Is this burst a quick coordination message during active development?"""
    if gap_min >= 5:
        return False
    app = burst['app_name'].lower()
    window = burst['window_title'].lower()
    # Short messages in Slack/chat during active dev = coordination
    if app == 'slack' and ('#backend' in window or 'huddle' in window):
        if len(burst['chars']) < 200:
            return True
    return False


def evaluate_prompt_logic(bursts):
    """
    Apply the generalized prompt's logic rules to produce sessions.
    Returns list of (start_idx, end_idx) session ranges.
    
    Rules from the new prompt:
    1. Gap > 30 min → NEW session (hard boundary)
    2. Different task → NEW session
    3. Same task → SAME session (even if type varies, even across files/apps)
    4. Interrupt + return → SPLIT at interrupt (resumed task = new session)
    5. Quick coordination (< 5 min, Slack) during dev → merge into surrounding
    """
    if not bursts:
        return []
    
    sessions = []
    sess_start = 0
    
    for i in range(1, len(bursts)):
        prev_ts = datetime.fromisoformat(bursts[i-1]['timestamp'])
        curr_ts = datetime.fromisoformat(bursts[i]['timestamp'])
        gap_min = (curr_ts - prev_ts).total_seconds() / 60
        
        # Rule 1: Gap > 30 min → always split
        if gap_min >= 30:
            sessions.append((sess_start, i - 1))
            sess_start = i
            continue
        
        # Rule 5: Quick coordination → merge into surrounding session
        if is_quick_coordination(bursts[i], bursts[i-1], gap_min):
            # Don't split, just continue
            continue
        
        # Rule 2/3: Same vs different task
        if same_task(bursts[i-1], bursts[i], gap_min):
            # Same task → continue
            continue
        else:
            # Different task → split
            sessions.append((sess_start, i - 1))
            sess_start = i
    
    sessions.append((sess_start, len(bursts) - 1))
    return sessions

def load_bursts(name):
    with open(Path(__file__).parent / "data" / f"{name}.json") as f:
        return json.load(f)

def main():
    print("=" * 70)
    print("  PROMPT LOGIC EVALUATOR — deterministic rule-based simulation")
    print("=" * 70)
    print()
    print("Rules being tested:")
    print("  1. Gap > 30 min → split")
    print("  2. Different inferred task → split")
    print("  3. Same task → merge (work type variations don't split)")
    print("  4. Interrupt (different task) + return → split at interrupt")
    print("  5. Quick coordination (Slack < 5 min) during dev → merge")
    print()
    
    task_inference_log = {}
    
    for name in sorted(GROUND_TRUTH.keys()):
        bursts = load_bursts(name)
        
        # Show task inference for debugging
        tasks = [infer_task(b) for b in bursts]
        
        produced = evaluate_prompt_logic(bursts)
        expected = GROUND_TRUTH_DETAIL[name]
        
        match = len(produced) == len(expected)
        
        print(f"\n{'='*70}")
        print(f"  {name} — {len(bursts)} bursts")
        print(f"{'='*70}")
        print(f"Ground truth: {len(expected)} sessions")
        print(f"Produced:     {len(produced)} sessions")
        print(f"Match:        {'✓' if match else '✗ MISMATCH'}")
        
        print(f"\n--- Task inference ---")
        for i, (b, t) in enumerate(zip(bursts, tasks)):
            gap = ''
            if i > 0:
                t1 = datetime.fromisoformat(bursts[i-1]['timestamp'])
                t2 = datetime.fromisoformat(b['timestamp'])
                gap_min = int((t2 - t1).total_seconds() / 60)
                if gap_min >= 5:
                    gap = f' [+{gap_min}min]'
            print(f"  [{i:2d}] {b['timestamp'][11:16]} {b['app_name'][:10]:10s} {b['window_title'][:25]:25s} → {t}{gap}")
        
        print(f"\n--- Produced sessions ---")
        for idx, (s, e) in enumerate(produced):
            print(f"  sess {idx+1}: bursts [{s}-{e}] {bursts[s]['timestamp'][11:16]}-{bursts[e]['timestamp'][11:16]} task={tasks[s]}")
        
        print(f"\n--- Ground truth sessions ---")
        for s, e, desc in expected:
            print(f"  [{s}-{e}] {desc}")
        
        if not match:
            print(f"\n  ⚠ MISMATCH: produced {len(produced)} vs expected {len(expected)}")
    
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    
    all_match = True
    for name in sorted(GROUND_TRUTH.keys()):
        bursts = load_bursts(name)
        produced = evaluate_prompt_logic(bursts)
        expected_count = GROUND_TRUTH[name]
        match = len(produced) == expected_count
        status = "✓" if match else "✗"
        print(f"  {name}: {len(produced)}/{expected_count} sessions {status}")
        if not match:
            all_match = False
    print()
    if all_match:
        print("  All challenges match!")
    else:
        print("  ⚠ Some mismatches — prompt logic needs adjustment")

if __name__ == "__main__":
    main()
