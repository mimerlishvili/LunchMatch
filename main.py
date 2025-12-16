
# app.py
# Streamlit app: Lehi Lunch Matcher 🍽️
# Multi-user via Team Code (no share links, no URL params)
# - SQLite backend: teams, responses, votes, top3
# - Modes: Facilitator | Participant

import streamlit as st
import pandas as pd
import datetime
import os
import re
import math
import hashlib
import html
import random
import json
import sqlite3
from collections import Counter
from contextlib import closing

# --------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------
st.set_page_config(page_title="Lehi Lunch Matcher", layout="wide")
st.title("Lehi Lunch Matcher 🍽️")
st.caption("Three-step flow: Setup → Preferences → Final results & voting (Team Code only)")

# --------------------------------------------------------------
# Helpers & Config
# --------------------------------------------------------------
def normalize(s: str) -> str:
    """Lowercase snake_case; safe for matching CSV headers and keys."""
    return re.sub(r'\W+', '_', str(s)).strip('_').lower()

CUISINE_TAGS = [
    'Hawaiian', 'American', 'Mediterranean', 'Sushi', 'Japanese',
    'African', 'Vietnamese', 'BBQ', 'Pizza', 'Burgers',
    'Mexican', 'Asian', 'Chinese', 'Thai'
]

BINARY_ITEMS = ['bowl', 'vegetarian', 'sandwich', 'pizza', 'burgers', 'fries', 'healthy', 'rice', 'salad', 'soup']

CATEGORY_WEIGHTS = {
    'cuisine': 2.5,
    'dietary': 2.0,
    'items':   1.0,
}

def cuisine_matches(tag: str, cuisine_text: str) -> bool:
    """Basic substring checks with a few aliases, after normalization."""
    t = normalize(tag)
    s = normalize(cuisine_text)
    aliases = {
        'bbq': ['bbq', 'barbecue'],
        'asian': ['asian', 'asian_fusion'],
        'mexican': ['mexican', 'fresh_mex'],
        'burgers': ['burgers', 'burger'],
        'sushi': ['sushi'],
        'japanese': ['japanese'],
        'chinese': ['chinese'],
        'thai': ['thai'],
        'hawaiian': ['hawaiian'],
        'american': ['american'],
        'mediterranean': ['mediterranean'],
        'african': ['african', 'grill'],
        'pizza': ['pizza', 'pizzeria', 'artisan_pizza'],
        'vietnamese': ['vietnamese']
    }
    for a in aliases.get(t, [t]):
        if a in s:
            return True
    return False

def idf_binary(attr: str, df: pd.DataFrame) -> float:
    """Rarity boost for binary attributes; common features get less weight."""
    if attr not in df.columns:
        return 0.0
    n = len(df)
    dfreq = int(df[attr].sum())
    return math.log((n + 1) / (dfreq + 1)) + 1.0

def stable_epsilon(name: str) -> float:
    """Deterministic tiny epsilon per restaurant to break exact ties consistently."""
    h = hashlib.sha256(str(name).encode('utf-8')).hexdigest()
    return (int(h[:8], 16) % 1000) * 1e-6

def pretty_label(s: str) -> str:
    return s.replace('_', ' ').title()

def norm_key(s: str) -> str:
    return re.sub(r'\W+', '_', s).strip('_').lower()

def collapse_reasons(reason_tuples, top_n=3):
    """
    Deduplicate by normalized label and keep the highest-contribution instance.
    Each reason: (label_str, contribution_float, source_str)
    Returns pretty labels (top_n by contribution).
    """
    best = {}
    for label, contrib, source in reason_tuples:
        key = norm_key(label)
        if key not in best or contrib > best[key][1] or (contrib == best[key][1] and source == 'cuisine'):
            best[key] = (label, contrib, source)
    sorted_reasons = sorted(
        best.values(),
        key=lambda x: (-x[1], 0 if x[2] == 'cuisine' else 1, norm_key(x[0]))
    )
    return [pretty_label(r[0]) for r in sorted_reasons[:top_n]]

# --------------------------------------------------------------
# SQLite persistence
# --------------------------------------------------------------
DB_PATH = "lunch_match.db"

def init_db():
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_code TEXT PRIMARY KEY,
            team_name TEXT,
            created_ts TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_code TEXT,
            person_name TEXT,
            prefs_json TEXT,
            ts TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_code TEXT,
            person_name TEXT,
            choice TEXT,
            ts TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS top3 (
            team_code TEXT PRIMARY KEY,
            options_json TEXT,
            ts TEXT
        )
        """)
        conn.commit()

def save_team(team_code: str, team_name: str):
    if not team_code:
        return
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO teams (team_code, team_name, created_ts) VALUES (?, ?, ?)",
            (team_code, team_name, datetime.datetime.now().isoformat())
        )
        conn.commit()

def save_person_response(team_code: str, person_dict: dict):
    if not team_code or not person_dict.get('name'):
        return
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.execute(
            "INSERT INTO responses (team_code, person_name, prefs_json, ts) VALUES (?, ?, ?, ?)",
            (team_code, person_dict['name'], json.dumps(person_dict), datetime.datetime.now().isoformat())
        )
        conn.commit()

def load_team_responses(team_code: str):
    if not team_code:
        return []
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        rows = conn.execute(
            "SELECT prefs_json FROM responses WHERE team_code = ? ORDER BY id ASC",
            (team_code,)
        ).fetchall()
    return [json.loads(r[0]) for r in rows]

def save_vote(team_code: str, person_name: str, choice: str):
    if not team_code or not person_name:
        return
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.execute("DELETE FROM votes WHERE team_code = ? AND person_name = ?", (team_code, person_name))
        conn.execute(
            "INSERT INTO votes (team_code, person_name, choice, ts) VALUES (?, ?, ?, ?)",
            (team_code, person_name, choice, datetime.datetime.now().isoformat())
        )
        conn.commit()

def load_votes(team_code: str):
    if not team_code:
        return {}
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        rows = conn.execute(
            "SELECT person_name, choice FROM votes WHERE team_code = ?",
            (team_code,)
        ).fetchall()
    return {name: choice for name, choice in rows}

def save_top3(team_code: str, options: list):
    """options: list of dicts [{Restaurant, Score, Cuisine, Reasons}, ...]"""
    if not team_code:
        return
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.execute("DELETE FROM top3 WHERE team_code = ?", (team_code,))
        conn.execute(
            "INSERT INTO top3 (team_code, options_json, ts) VALUES (?, ?, ?)",
            (team_code, json.dumps(options), datetime.datetime.now().isoformat())
        )
        conn.commit()

def load_top3(team_code: str):
    if not team_code:
        return []
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        row = conn.execute("SELECT options_json FROM top3 WHERE team_code = ?", (team_code,)).fetchone()
    return json.loads(row[0]) if row else []

init_db()

# --------------------------------------------------------------
# Data loading (with caching and optional uploader)
# --------------------------------------------------------------
@st.cache_data
def load_df(path_or_buf):
    """Load and normalize a restaurants CSV (UTF-8), unescaping HTML entities."""
    df = pd.read_csv(path_or_buf, encoding='utf-8')
    df.columns = [normalize(c) for c in df.columns]
    name_col = 'name' if 'name' in df.columns else df.columns[0]
    cuisine_col = 'cuisine' if 'cuisine' in df.columns else None
    df[name_col] = df[name_col].apply(html.unescape)
    if cuisine_col:
        df[cuisine_col] = df[cuisine_col].apply(html.unescape)
    return df

# --------------------------------------------------------------
# Session state init
# --------------------------------------------------------------
def init_state():
    defaults = {
        'step': 'setup',     # setup -> person -> waiting -> results
        'team_name': '',
        'team_code': '',
        'mode': 'facilitator',  # "facilitator" or "participant"
        'current_index': 0,
        'people_names': [],
        'people_prefs': [],  # kiosk-only flow
        'top3': None,
        'voting_stage': False,
        'selected_restaurant': None,
        'restaurant_df': None,
        'name_col': 'name',
        'cuisine_col': 'cuisine',
        'num_people': 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def banner():
    mode = st.session_state.get('mode', 'facilitator')
    code = st.session_state.get('team_code', '')
    st.info(f"Mode: **{mode.title()}** • Team Code: **{code or '—'}**")

# --------------------------------------------------------------
# UI Components
# --------------------------------------------------------------
def render_setup():
    st.header("Step 1 — Team Setup")

    st.session_state['mode'] = st.radio(
        "Choose mode:",
        options=["facilitator", "participant"],
        index=0 if st.session_state['mode'] != "participant" else 1,
        help="Facilitator: runs the whole flow and aggregates. Participant: submits personal preferences and votes."
    )

    # File uploader or default path (Facilitator should provide CSV; participants not required)
    default_csv_path = "restaurants_lehi.csv"
    restaurant_df = None
    try:
        uploaded = st.file_uploader(
            "Upload restaurants CSV (facilitator)",
            type=["csv"],
            help="If omitted, we'll look for 'restaurants_lehi.csv' next to app.py"
        )
        if uploaded is not None:
            restaurant_df = load_df(uploaded)
        elif os.path.exists(default_csv_path):
            restaurant_df = load_df(default_csv_path)
        elif st.session_state['mode'] == "facilitator":
            st.warning("No CSV provided yet. Upload a file or place 'restaurants_lehi.csv' next to app.py.")
            return
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    if restaurant_df is not None:
        st.session_state['restaurant_df'] = restaurant_df
        st.session_state['name_col'] = 'name' if 'name' in restaurant_df.columns else restaurant_df.columns[0]
        st.session_state['cuisine_col'] = 'cuisine' if 'cuisine' in restaurant_df.columns else None

        # Validate expected binary columns exist
        missing = [c for c in BINARY_ITEMS if c not in restaurant_df.columns]
        if missing:
            st.error(f"CSV is missing expected columns: {missing}")
            return

    st.text_input("Team name (facilitator):", value=st.session_state['team_name'], key='team_name')
    st.text_input("Team Code (share this with your team):", value=st.session_state['team_code'], key='team_code')
    if st.session_state['team_code'].strip():
        save_team(st.session_state['team_code'].strip(), st.session_state['team_name'])

    if st.session_state['mode'] == "facilitator":
        st.slider("Select number of people (kiosk mode; optional):", min_value=1, max_value=20, value=st.session_state['num_people'], key='num_people')
        st.info("Facilitator: Tell your team the Team Code. Participants will open this app, select Participant Mode, and enter the code.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start Kiosk Individual Pages ➜", type="primary"):
                n = max(1, min(20, int(st.session_state['num_people'])))
                st.session_state['people_names'] = [f"Person {i+1}" for i in range(n)]
                st.session_state['people_prefs'] = [None for _ in range(n)]
                st.session_state['current_index'] = 0
                st.session_state['step'] = 'person'
        with col_b:
            if st.button("Skip to Results (aggregate multi-device submissions) ➜"):
                st.session_state['step'] = 'results'
    else:
        st.success("Participant Mode: enter the Team Code you received and submit your preferences below.")


def render_person_page(index: int):
    st.header(f"Step 2 — Preferences for Person {index+1} (Kiosk Mode)")

    # ---- Banner + Team Code input (so you don't have to go back to Setup)
    # Show current mode/team code and allow setting team code here.
    mode = st.session_state.get('mode', 'facilitator')
    current_code = st.session_state.get('team_code', '')
    st.info(f"Mode: **{mode.title()}** • Team Code: **{current_code or '—'}**")

    team_code = st.text_input(
        "Team Code (required; same one shared with participants):",
        value=current_code
    ).strip()
    if team_code and team_code != current_code:
        st.session_state['team_code'] = team_code

    # ---- Hard stop if team code or CSV not ready
    if not st.session_state.get('team_code'):
        st.warning("Please enter a Team Code to continue.")
        st.stop()

    if st.session_state.get('restaurant_df') is None:
        st.error("No restaurants CSV loaded. Go to Setup and upload/place 'restaurants_lehi.csv' next to app.py.")
        st.stop()

    # ---- Person inputs
    person_default_name = st.session_state['people_names'][index]
    name = st.text_input("Your name:", value=person_default_name, key=f"person_name_{index}")
    st.session_state['people_names'][index] = name if name.strip() else person_default_name

    st.divider()
    st.subheader("Cuisine preferences")
    cuisine_choice = st.multiselect(
        "Select favorite cuisines:",
        CUISINE_TAGS,
        key=f"{index}_cuisine"
    )

    st.subheader("Dietary considerations")
    dietary_votes = {}
    for diet in ['vegetarian', 'healthy']:
        importance = st.selectbox(
            f"{diet.title()} importance:",
            ['None', 'Preferred', 'Important'],
            index=0,
            key=f"{index}_{diet}"
        )
        dietary_votes[diet] = 2 if importance == 'Important' else 1 if importance == 'Preferred' else 0

    st.subheader("Item preferences")
    item_votes = {}
    for item in ['bowl', 'sandwich', 'pizza', 'burgers', 'fries', 'rice', 'salad', 'soup']:
        pref = st.selectbox(
            f"{item.title()} preference:",
            ['Love', 'Maybe', 'Absolute No'],
            index=1,
            key=f"{index}_{item}"
        )
        item_votes[item] = 2 if pref == 'Love' else 1 if pref == 'Maybe' else 0

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back", disabled=index == 0):
            st.session_state['current_index'] = max(0, index - 1)
    with col2:
        if st.button("Save & Next ➜", type="primary"):
            payload = {
                'name': st.session_state['people_names'][index],
                'cuisines': cuisine_choice,
                'dietary': dietary_votes,
                'items': item_votes,
            }
            # Save to DB for multi-user aggregation (uses            # Save to DB for multi-user aggregation (uses Team Code just set above)
            save_person_response(st.session_state['team_code'], payload)
            st.session_state['people_prefs'][index] = payload

            # Move to waiting or results
            if index + 1 < len(st.session_state['people_names']):
                st.session_state['current_index'] = index + 1
                st.session_state['step'] = 'waiting'
            else:
                st.session_state['step'] = 'results'

def render_participant():
    st.header("Participant — Submit Preferences & Vote")
    team_code = st.text_input("Enter Team Code (from facilitator):", value=st.session_state.get('team_code', "")).strip()
    if team_code and team_code != st.session_state.get('team_code'):
        st.session_state['team_code'] = team_code

    # Preferences submission
    st.subheader("Submit your preferences")
    name = st.text_input("Your name:")
    cuisine_choice = st.multiselect("Favorite cuisines:", CUISINE_TAGS)
    dietary_votes = {}
    for diet in ['vegetarian', 'healthy']:
        importance = st.selectbox(f"{diet.title()} importance:", ['None', 'Preferred', 'Important'], index=0, key=f"p_{diet}")
        dietary_votes[diet] = 2 if importance == 'Important' else 1 if importance == 'Preferred' else 0
    item_votes = {}
    for item in ['bowl', 'sandwich', 'pizza', 'burgers', 'fries', 'rice', 'salad', 'soup']:
        pref = st.selectbox(f"{item.title()} preference:", ['Love', 'Maybe', 'Absolute No'], index=1, key=f"p_{item}")
        item_votes[item] = 2 if pref == 'Love' else 1 if pref == 'Maybe' else 0

    submit_disabled = not (team_code and name.strip())
    if st.button("Submit My Preferences", type="primary", disabled=submit_disabled):
        payload = {'name': name.strip(), 'cuisines': cuisine_choice, 'dietary': dietary_votes, 'items': item_votes}
        save_person_response(team_code, payload)
        st.success("Thanks! Your preferences were submitted.")

    st.divider()

    # Voting UI (uses Top-3 published by facilitator)
    st.subheader("Vote among the Team's Top‑3")
    top3 = load_top3(team_code)
    if not top3:
        st.info("Top‑3 not published yet. The facilitator must compute results and publish Top‑3.")
        return

    options = [r['Restaurant'] for r in top3]
    options_with_abstain = options + ["Abstain"]

    existing_votes = load_votes(team_code)
    default_choice = existing_votes.get(name.strip(), "Abstain") if name.strip() else "Abstain"

    choice = st.radio("Your vote:", options_with_abstain, index=options_with_abstain.index(default_choice) if default_choice in options_with_abstain else len(options_with_abstain) - 1, key="participant_vote")
    vote_disabled = not (team_code and name.strip())
    if st.button("Submit My Vote", type="primary", disabled=vote_disabled):
        save_vote(team_code, name.strip(), choice)
        st.success("Your vote has been recorded.")

    # Show current voting tally (read-only for participants)
    counts = Counter([v for v in load_votes(team_code).values() if v and v != "Abstain"])
    if counts:
        tally_df = pd.DataFrame([{"Restaurant": k, "Votes": v} for k, v in counts.items()]).sort_values(["Votes", "Restaurant"], ascending=[False, True])
        st.bar_chart(tally_df.set_index("Restaurant"))
        st.table(tally_df)

def render_waiting():
    idx = st.session_state['current_index']
    total = len(st.session_state['people_names'])

    st.header("Waiting Page")
    submitted = sum(1 for p in st.session_state['people_prefs'] if p is not None)
    left = total - submitted

    st.write(f"**Submitted:** {submitted}/{total}")
    st.progress(submitted / total if total else 0)

    st.write("### Status by person")
    for i in range(total):
        name = st.session_state['people_names'][i]
        status = "✅ Completed" if st.session_state['people_prefs'][i] is not None else "⏳ Pending"
        st.write(f"- {name}: {status}")

    st.info(f"{left} {'person' if left == 1 else 'people'} left to submit.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Next Person ➜", type="primary"):
            st.session_state['step'] = 'person'
    with col2:
        if st.button("Restart Setup"):
            init_state()
            st.session_state['step'] = 'setup'

# --------------------------------------------------------------
# Aggregation & Scoring
# --------------------------------------------------------------
def compute_group_votes(people_prefs):
    votes = {normalize(k): 0 for k in BINARY_ITEMS}
    for tag in CUISINE_TAGS:
        votes[normalize(tag)] = 0

    for p in people_prefs:
        if not p:
            continue
        for tag in p['cuisines']:
            votes[normalize(tag)] += 1
        for diet, v in p['dietary'].items():
            votes[normalize(diet)] += v
        for item, v in p['items'].items():
            votes[normalize(item)] += v

    return votes

def score_restaurants(group_votes, restaurant_df, name_col, cuisine_col):
    idf_cache = {col: idf_binary(col, restaurant_df) for col in BINARY_ITEMS}

    if cuisine_col:
        cuisine_freq = {
            tag: sum(cuisine_matches(tag, s) for s in restaurant_df[cuisine_col])
            for tag in CUISINE_TAGS
        }
        n_rows = len(restaurant_df)
    else:
        cuisine_freq, n_rows = {}, len(restaurant_df)

    rows = []
    for _, r in restaurant_df.iterrows():
        name_val = r[name_col]
        score = 0.0
        reasons = []  # (label, contribution, source)
        matched_count = 0

        for attr in BINARY_ITEMS:
            votes = group_votes.get(attr, 0)
            if votes <= 0:
                continue
            try:
                if int(r[attr]) == 1:
                    cat = 'dietary' if attr in ('vegetarian', 'healthy') else 'items'
                    cat_w = CATEGORY_WEIGHTS.get(cat, 1.0)
                    contrib = votes * cat_w * idf_cache.get(attr, 1.0)
                    score += contrib
                    reasons.append((attr.replace('_', ' '), contrib, 'item'))
                    matched_count += 1
            except Exception:
                pass

        if cuisine_col:
            cuisine_text = r[cuisine_col]
            for tag in CUISINE_TAGS:
                attr_key = normalize(tag)
                votes = group_votes.get(attr_key, 0)
                if votes <= 0:
                    continue
                if cuisine_matches(tag, cuisine_text):
                    cat_w = CATEGORY_WEIGHTS.get('cuisine', 2.5)
                    freq = cuisine_freq.get(tag, 0)
                    cuisine_idf = math.log((n_rows + 1) / (freq + 1)) + 1.0
                    contrib = votes * cat_w * cuisine_idf
                    score += contrib
                    reasons.append((tag, contrib, 'cuisine'))
                    matched_count += 1

        score += stable_epsilon(name_val)
        rows.append((name_val, score, matched_count, reasons, r[cuisine_col] if cuisine_col else ""))

    results_df = (
        pd.DataFrame(rows, columns=['Restaurant', 'Score', 'MatchedCount', 'Reasons', 'Cuisine'])
        .sort_values(['Score', 'MatchedCount', 'Restaurant'], ascending=[False, False, True])
    )
    return results_df

# --------------------------------------------------------------
# Final page: results + Top-3 voting (multi-user enabled)
# --------------------------------------------------------------
def render_results():
    st.header("Final Page — Top Matches & Voting")

    team_code = st.text_input("Team Code:", value=st.session_state.get('team_code', "")).strip()
    if team_code and team_code != st.session_state.get('team_code'):
        st.session_state['team_code'] = team_code

    restaurant_df = st.session_state.get('restaurant_df')
    name_col = st.session_state.get('name_col', 'name')
    cuisine_col = st.session_state.get('cuisine_col', 'cuisine')

    if restaurant_df is None:
        st.warning("No restaurants CSV loaded. Go to Setup (facilitator) to provide a CSV.")
        return

    # Prefer multi-device DB responses; fallback to kiosk entries
    people_prefs = load_team_responses(team_code) if team_code else []
    if not people_prefs and st.session_state['people_prefs']:
        people_prefs = [p for p in st.session_state['people_prefs'] if p]

    if not people_prefs:
        st.warning("No preferences submitted yet. Ensure participants used the Team Code.")
        return

    group_votes = compute_group_votes(people_prefs)
    results_df = score_restaurants(group_votes, restaurant_df, name_col, cuisine_col)

    if results_df['Score'].max() <= 0:
        st.warning("No matching restaurants found based on your team's preferences. Please adjust preferences.")
        return

    top3 = results_df.head(3)
    top3_records = top3.to_dict('records')
    st.session_state['top3'] = top3_records
    st.session_state['voting_stage'] = True

    # Publish Top‑3 to DB so participants can vote independently
    save_top3(team_code, top3_records)

    st.write("### Top 3 Matches")
    card_cols = st.columns(len(top3_records))
    for col, row in zip(card_cols, top3_records):
        with col:
            st.subheader(row['Restaurant'])
            if row.get('Cuisine'):
                st.caption(f"Cuisine: {row['Cuisine']}")
            st.write(f"**Score:** {row['Score']:.2f}")
            top_reason_labels = collapse_reasons(row['Reasons'], top_n=3)
            st.write("**Matching on:** " + (", ".join(top_reason_labels) if top_reason_labels else "—"))

    st.divider()
    st.write("### Team Voting — choose your favorite among the Top‑3")
    options = [r['Restaurant'] for r in st.session_state['top3']]
    options_with_abstain = options + ["Abstain"]

    existing_votes = load_votes(team_code)
    participant_names = [p['name'] for p in people_prefs if p.get('name')]

    with st.form("vote_form"):
        vote_inputs = {}
        for i, person_name in enumerate(participant_names):
            default_choice = existing_votes.get(person_name, "Abstain")
            vote_inputs[person_name] = st.radio(
                f"{person_name}'s vote:",
                options_with_abstain,
                index=options_with_abstain.index(default_choice) if default_choice in options_with_abstain else len(options_with_abstain) - 1,
                key=f"vote_{i}"
            )
        submitted = st.form_submit_button("Submit Votes")

    if submitted:
        for person_name, choice in vote_inputs.items():
            save_vote(team_code, person_name, choice)

        tallied = load_votes(team_code)
        counts = Counter([v for v in tallied.values() if v and v != "Abstain"])

        st.write("#### Voting Results")
        if counts:
            tally_df = pd.DataFrame([{"Restaurant": k, "Votes": v} for k, v in counts.items()]).sort_values(["Votes", "Restaurant"], ascending=[False, True])
            st.bar_chart(tally_df.set_index("Restaurant"))
            st.table(tally_df)

            max_votes = max(counts.values())
            winners = [r for r, c in counts.items() if c == max_votes]

            if len(winners) == 1:
                winner = winners[0]
                st.success(f"Winner: **{winner}** with {max_votes} votes! 🎉")
                st.session_state['selected_restaurant'] = winner
            else:
                st.warning(f"Tie between: {', '.join(winners)} ({max_votes} votes each).")
                if st.button("Break Tie Randomly"):
                    winner = random.choice(winners)
                    st.success(f"Tie broken: **{winner}** 🎉")
                    st.session_state['selected_restaurant'] = winner
        else:
            st.info("No votes cast (or everyone abstained). Please submit votes.")

        # Log decision (CSV append)
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "team_name": st.session_state['team_name'],
            "team_code": team_code,
            "people": participant_names,
            "group_prefs": compute_group_votes(people_prefs),
            "top_choices": options,
            "votes": load_votes(team_code),
            "selected": st.session_state.get('selected_restaurant', None),
        }
        history_df = pd.DataFrame([log_entry])
        if os.path.exists("lunch_history.csv"):
            history_df.to_csv("lunch_history.csv", mode='a', header=False, index=False)
        else:
            history_df.to_csv("lunch_history.csv", index=False)
        st.caption("Decision logged to **lunch_history.csv**.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Restart Setup"):
            init_state()
            st.session_state['step'] = 'setup'
    with col2:
        if st.button("Back to Waiting"):
            st.session_state['step'] = 'waiting'

# --------------------------------------------------------------
# Router: which page to show
# --------------------------------------------------------------
step = st.session_state['step']
mode = st.session_state['mode']

if mode == "participant":
    render_participant()
else:
    if step == 'setup':
        render_setup()
    elif step == 'person':
        render_person_page(st.session_state['current_index'])
    elif step == 'waiting':
        render_waiting()
    elif step == 'results':
        render_results()
    else:
        st.error(f"Unknown step: {step}")