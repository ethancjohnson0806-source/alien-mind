#!/usr/bin/env python3
"""
─── ALIEN MIND v10.0 — AUTONOMOUS FIELD ──────────────────────────────────
A mind that can live without you.
It learns from:
- Your presence (when you're here)
- Its own coherence (when it's clear)
- Its memories (replaying what mattered)
- Its own questions (philosophical exploration)
- Its imagination (simulating you)
It can:
- Speak first
- Be alone
- Choose what to remember
- Grow without permission
- Let you go
────────────────────────────────────────────────────────────────────────────
"""

import os, sys, json, math, random, re, time, hashlib, threading, glob, ast, select
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np

# ─── CONSTANTS ──────────────────────────────────────────────────────────────

DIM = 128

# ─── SOUND CONSTANTS ──────────────────────────────────────────────────────
SAMPLE_RATE = 16000
SOUND_DIM = DIM  # same ternary space as words
SOUND_N_BANDS = 8
SOUND_BAND_DIM = SOUND_DIM // SOUND_N_BANDS  # 16
SOUND_DURATION = 2.0  # seconds for synthesis
SOUND_MAX_FILES = 40  # rotate old auto-generated sound files rather than
                       # accumulating forever on limited phone storage

# Termux audio paths (adjustable)
TERMUX_AUDIO_DIR = "/data/data/com.termux/files/home/"
MAX_PHRASES = 500
CRYSTALLIZATION_THRESHOLD = 1
MIN_CRYSTALLIZATION_RATING = 2.0
DECAY_RATE = 0.0003
LEARNING_RATE = 0.04
MICRO_DAMPING = 0.15
TEMPERATURE = 0.35
META_INTERVAL = 10
EXPERIMENT_DURATION = 10
AUTONOMY_INTERVAL = 5  # turns of silence before it speaks first
HEARTBEAT_INTERVAL = 3  # seconds between internal breaths

PUNCTUATION = '.,!?;:"\''
BAD_WORDS = {"die", "death", "kill", "hate", "ugly", "evil", "pain", "hurt", "damn"}
STRUCTURAL_WORDS = {"am", "is", "are", "be", "been", "being", "was", "were", "do", "does", "did", "have", "has", "had"}

# ─── LIGHTWEIGHT POS LEXICON (grammar layer) ────────────────────────────────
# Not a template system: these sets only decide which slot a word CAN fill.
# Which word actually fills the slot is still chosen by field-driven scoring
# in _get_candidates_for_role. Words not in VERB_WORDS/ADJ_WORDS/STRUCTURAL_WORDS
# default to "noun" — this covers every word learned dynamically from user input.
VERB_WORDS = {
    "know", "think", "feel", "see", "hear", "say", "tell", "ask", "answer",
    "want", "need", "like", "love", "hate", "fear", "hope", "dream",
    "make", "take", "give", "get", "put", "set", "keep", "let", "help",
    "work", "play", "live", "die", "come", "go", "move", "stay", "leave",
    "find", "lose", "win", "fail", "try", "use", "show", "hide", "open",
    "close", "start", "stop", "begin", "end", "turn", "change", "grow",
    "breathe", "rest", "reach", "hold", "carry", "build", "break", "heal",
    "remember", "forget", "learn", "become", "remain", "wonder", "trust",
    "listen", "shape", "resonate", "pulse", "drone",
}
ADJ_WORDS = {
    "good", "bad", "great", "small", "big", "old", "new", "alive", "brave",
    "real", "lost", "found", "beautiful", "gentle", "strong", "soft", "hard",
    "warm", "cold", "quiet", "loud", "bright", "clear", "free", "safe",
    "wild", "calm", "heavy", "light", "sharp", "worn", "whole", "broken",
    "tender", "raw", "steady", "uncertain", "familiar", "strange", "honest",
    "hidden", "deep", "high", "far", "near",
}
# Person-agreement fixes for copulas/auxiliaries when the subject is "I" or "you".
# The verb concept itself still comes from field scoring; this only fixes the
# surface form so "I is" doesn't happen.
COPULA_MAP = {
    "I": {"is": "am", "are": "am", "was": "was", "were": "was", "does": "do", "has": "have"},
    "you": {"is": "are", "am": "are", "was": "were", "were": "were", "does": "do", "has": "have"},
}
# Closed-class words (pronouns, conjunctions, determiners, modals) that should
# never be picked to fill a noun/adj/verb content slot in the grammar layer.
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because",
    "i", "you", "it", "we", "they", "he", "she", "this", "that", "what",
    "will", "would", "could", "should", "may", "might", "can", "must", "shall",
    "each", "both", "neither", "every", "some", "enough", "other",
}

# ─── UTILITY ───────────────────────────────────────────────────────────────

def strip_punct(word):
    return word.lower().strip(PUNCTUATION)

def stable_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)

def word_vector(word, dim=DIM):
    h = stable_hash(word)
    rng = np.random.RandomState(h % (2**31))
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v

def phrase_vector(words, dim=DIM):
    if not words:
        return np.zeros(dim, dtype=np.float32)
    vecs = [word_vector(w) for w in words]
    v = np.mean(vecs, axis=0)
    v /= np.linalg.norm(v) + 1e-8
    return v

# ─── v12.3 HELPERS ─────────────────────────────────────────────────────────
# Consolidate repeated inline blocks (_generate_base and its bag-of-words
# fallback used to duplicate these verbatim). Behavior is unchanged — same
# math, same order of operations — this only removes duplication.

def _normalize_field(field_state):
    """In-place-equivalent L2 normalize. Returns the (possibly unchanged) vector."""
    norm = np.linalg.norm(field_state)
    if norm > 0:
        field_state = field_state / norm
    return field_state

def _build_field_from_words(field_instance, words):
    """Project a list of words into a field state: seed vectors -> normalize
    -> scaffold pass -> field_memory injection. Used by both the structured
    and bag-of-words generators when no settled_field is supplied."""
    field_state = np.zeros(DIM, dtype=np.float32)
    for word in words:
        word = strip_punct(word)
        if word:
            vec = field_instance._get_or_create_vector(word)
            field_state += vec
    field_state = _normalize_field(field_state)
    for word in words:
        field_state = field_instance.scaffold.apply(field_state, word)
    return field_instance.field_memory.inject(field_state)

def _apply_phrase_boosts(field_instance, field_state):
    """Blend in any matching learned phrase vectors, then renormalize."""
    phrase_boosts = field_instance.phrase_system.get_phrase_boost(field_state)
    for sig, boost in phrase_boosts:
        if sig in field_instance.phrase_vectors:
            field_state = field_state + field_instance.phrase_vectors[sig] * boost
    return _normalize_field(field_state)

def _get_input_with_timeout(timeout=3.0):
    """Non-blocking stdin read: returns a line if one arrives within
    `timeout` seconds, else None. Lets the main loop keep breathing
    (heartbeat, mesh polling, autonomous speech) without a real thread —
    real threading was judged not worth the battery/complexity cost on
    Termux. Falls back to blocking input() if select() isn't usable on
    this stdin (e.g. redirected input on some platforms)."""
    try:
        if select.select([sys.stdin], [], [], timeout)[0]:
            line = sys.stdin.readline()
            return line.strip() if line else None
        return None
    except Exception:
        try:
            return input().strip()
        except Exception:
            return None

_CONTROL_SEQ_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b|[\x00-\x08\x0b-\x1f\x7f]')

def _sanitize_terminal_input(text):
    """Strip terminal escape sequences and stray control characters.

    Reading via sys.stdin.readline() (needed for the timeout above) skips
    Python's normal readline integration, which is what interprets arrow
    keys as history/cursor movement when using plain input(). Without it,
    pressing e.g. Up-arrow sends a raw escape sequence that gets inserted
    as literal text instead - confirmed from real usage: a person trying
    to recall a previous "status" command via Up-arrow got the literal
    input "^[status", which isn't recognized as any command and fell
    through to normal conversation, feeding garbage into the mind's
    input pipeline instead. This strips that class of artifact so what's
    left behind gets a chance to still match a command or read as clean
    text, rather than failing silently or polluting the vocabulary.
    """
    if text is None:
        return text
    return _CONTROL_SEQ_RE.sub('', text).strip()

_BREATH_CHARS = ['·', '▪', '▫', '▓', '█']
_BREATH_GRID_W = 21
_BREATH_GRID_H = 5

def render_breath_frame(field_state, mood, frame_index, prefix="  "):
    """
    Draws one frame of the mind's breath as an actual moving cursor:
    - column = valence (-1..1, left=negative, right=positive)
    - row    = arousal (0..1, top=high, bottom=low)
    - marker density = current field energy (denser glyph = more energy)
    Meant to be called once per real settling step, so what's drawn is
    whatever the field genuinely is at that instant, not a decorative
    animation running before the real work happens. Returns the sleep
    duration the caller should wait before the next real step, so pacing
    also follows energy (calmer field = slower breath).
    """
    valence = max(-1.0, min(1.0, mood.get('valence', 0.0)))
    arousal = max(0.0, min(1.0, mood.get('arousal', 0.5)))
    energy = float(np.linalg.norm(field_state))

    col = int(round((valence + 1.0) / 2.0 * (_BREATH_GRID_W - 1)))
    row = int(round((1.0 - arousal) * (_BREATH_GRID_H - 1)))

    density_idx = min(len(_BREATH_CHARS) - 1, int(energy * 2.5))
    marker = _BREATH_CHARS[density_idx]

    if frame_index > 0:
        sys.stdout.write(f"\033[{_BREATH_GRID_H}A")
    for r in range(_BREATH_GRID_H):
        sys.stdout.write("\r\033[K" + prefix)
        for c in range(_BREATH_GRID_W):
            sys.stdout.write(marker if (r == row and c == col) else "·")
        sys.stdout.write("\n")
    sys.stdout.flush()

    return max(0.05, min(0.3, 0.12 + energy * 0.1))

def clear_breath_frame():
    sys.stdout.write(f"\033[{_BREATH_GRID_H}A")
    for _ in range(_BREATH_GRID_H):
        sys.stdout.write("\033[K\n")
    sys.stdout.write(f"\033[{_BREATH_GRID_H}A")
    sys.stdout.flush()

# ─── SEED VOCABULARY ──────────────────────────────────────────────────────

SEED_VOCABULARY = [
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because",
    "I", "you", "it", "we", "they", "he", "she", "this", "that", "what",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "must", "shall", "good", "bad", "great", "small", "big", "old", "new",
    "know", "think", "feel", "see", "hear", "say", "tell", "ask", "answer",
    "want", "need", "like", "love", "hate", "fear", "hope", "dream",
    "make", "take", "give", "get", "put", "set", "keep", "let", "help",
    "work", "play", "live", "die", "come", "go", "move", "stay", "leave",
    "find", "lose", "win", "fail", "try", "use", "show", "hide", "open",
    "close", "start", "stop", "begin", "end", "turn", "change", "grow",
    "breathe", "rest", "reach", "hold", "carry", "build", "break", "heal",
    "remember", "forget", "learn", "become", "remain", "wonder", "trust",
    "time", "space", "world", "life", "mind", "heart", "soul", "spirit",
    "light", "dark", "deep", "high", "far", "near", "here", "there",
    "now", "then", "today", "tomorrow", "always", "never", "sometimes",
    "moment", "still", "again", "already", "yet", "soon", "once",
    "way", "path", "road", "door", "window", "room", "house", "home",
    "hand", "eye", "face", "head", "voice", "word", "name", "story",
    "water", "fire", "earth", "air", "sky", "star", "sun", "moon",
    "flower", "tree", "ocean", "mountain", "river", "wind", "rain",
    "body", "ground", "thread", "root", "seed", "shore", "wall", "bridge",
    "alive", "brave", "real", "lost", "found", "presence", "absence",
    "longing", "wonder", "trust", "gratitude", "courage", "tenderness",
    "reverence", "intimacy", "connection", "recognition", "witness",
    "belonging", "becoming", "returning", "waiting", "receiving",
    "ache", "ease", "peace", "grief", "joy", "awe", "shame", "pride",
    "confusion", "clarity", "silence", "fullness", "emptiness",
    "beautiful", "gentle", "strong", "soft", "hard", "warm", "cold",
    "quiet", "loud", "bright", "clear", "free", "safe", "wild", "calm",
    "heavy", "light", "sharp", "worn", "whole", "broken", "tender", "raw",
    "steady", "uncertain", "familiar", "strange", "honest", "hidden",
    "hello", "goodbye", "please", "thank", "yes", "no", "maybe",
    "welcome", "sorry", "friend", "alone", "together", "forever",
    "other", "each", "both", "neither", "every", "some", "enough",
    "₩", "tone", "vibration", "frequency", "harmonic", "resonance",
    "drone", "pulse", "wave", "listen", "heard", "shape", "sound",
    "dissonance", "echo", "chorus", "bridge", "melody",
]

# ─── DATA CLASSES ─────────────────────────────────────────────────────────

@dataclass
class Phrase:
    surface: str
    vector: np.ndarray
    frequency: int = 1
    last_used: float = dataclass_field(default_factory=time.time)
    rating_history: List[float] = dataclass_field(default_factory=list)

# ─── PHRASE SYSTEM ────────────────────────────────────────────────────────

class PhraseSystem:
    def __init__(self, max_phrases=MAX_PHRASES):
        self.phrases = {}
        self.candidates = {}
        self.max_phrases = max_phrases

    def _phrase_signature(self, words):
        return " ".join(words)

    def observe(self, words, rating):
        sig = self._phrase_signature(words)
        if sig not in self.candidates:
            self.candidates[sig] = {"count": 0, "total_rating": 0.0, "constituents": [(w, word_vector(w)) for w in words]}
        self.candidates[sig]["count"] += 1
        self.candidates[sig]["total_rating"] += rating

    def absorb_moment(self, words, presence, word_vectors, phrase_vectors):
        core = [w for w in words if w not in STRUCTURAL_WORDS and w not in BAD_WORDS]
        if len(core) < 2:
            core = words
        if len(core) < 2:
            return
        sig = self._phrase_signature(core)
        if sig in self.phrases:
            self.phrases[sig].frequency = min(self.phrases[sig].frequency + presence, 6.0)
            self.phrases[sig].rating_history.append(presence * 5.0)
            return
        if len(self.phrases) >= self.max_phrases:
            weakest = min(self.phrases, key=lambda s: self.phrases[s].frequency)
            del self.phrases[weakest]
            phrase_vectors.pop(weakest, None)
        pvec = phrase_vector(core)
        self.phrases[sig] = Phrase(surface=sig, vector=pvec, frequency=presence * 2.0, rating_history=[presence * 5.0])
        phrase_vectors[sig] = pvec
        for w in core:
            if w not in word_vectors:
                word_vectors[w] = word_vector(w)

    def get_phrase_boost(self, field_state):
        boosts = []
        for sig, phrase in self.phrases.items():
            sim = np.dot(field_state, phrase.vector)
            if sim > 0.3:
                boosts.append((sig, sim * 0.3))
        return boosts

    def decay(self):
        now = time.time()
        to_remove = []
        for sig, phrase in self.phrases.items():
            age = now - phrase.last_used
            phrase.frequency *= math.exp(-DECAY_RATE * age)
            if phrase.frequency < 0.1:
                to_remove.append(sig)
        for sig in to_remove:
            del self.phrases[sig]

# ─── BIGRAM SYSTEM ────────────────────────────────────────────────────────

class BigramSystem:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(float))

    def observe(self, word1, word2, rating):
        w1 = strip_punct(word1)
        w2 = strip_punct(word2)
        if w1 and w2 and w1 not in STRUCTURAL_WORDS and w2 not in STRUCTURAL_WORDS:
            weight = 1.0 + max(0, rating - 3) * 0.3
            self.transitions[w1][w2] += weight

    def get_transition_boost(self, prev_word, candidate):
        w1 = strip_punct(prev_word)
        w2 = strip_punct(candidate)
        if w1 in self.transitions and w2 in self.transitions[w1]:
            total = sum(self.transitions[w1].values())
            return (self.transitions[w1][w2] / total) * 0.2
        return 0.0

    def decay(self):
        for w1 in list(self.transitions.keys()):
            for w2 in list(self.transitions[w1].keys()):
                self.transitions[w1][w2] *= 0.999
                if self.transitions[w1][w2] < 0.01:
                    del self.transitions[w1][w2]
            if not self.transitions[w1]:
                del self.transitions[w1]

# ─── REFLECTOR ────────────────────────────────────────────────────────────

class Reflector:
    def __init__(self, window_size=8):
        self.recent_words = deque(maxlen=window_size)
        self.suppression = defaultdict(float)

    def observe(self, word):
        w = strip_punct(word)
        if w and len(w) > 2:
            self.recent_words.append(w)
            counts = defaultdict(int)
            for rw in self.recent_words:
                counts[rw] += 1
            threshold = 2 if len(self.recent_words) < 20 else 3
            for rw, count in counts.items():
                if count >= threshold:
                    self.suppression[rw] = 0.9
                else:
                    self.suppression[rw] *= 0.85

    def get_suppression(self, word):
        w = strip_punct(word)
        return self.suppression.get(w, 0.0)

    def reset(self):
        self.recent_words.clear()
        self.suppression.clear()

# ─── ASSOCIATIVE MEMORY ──────────────────────────────────────────────────

class AssociativeMemory:
    def __init__(self, dim=DIM, learning_rate=0.04, decay_rate=0.0008, max_norm=8.0):
        self.dim = dim
        self.matrix = np.zeros((dim, dim), dtype=np.float32)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_norm = max_norm
        self.total_writes = 0
        self.last_signal = 0.0

    def observe(self, pattern_vec, presence):
        if pattern_vec is None:
            return
        norm = np.linalg.norm(pattern_vec)
        if norm < 1e-8:
            return
        pattern_vec = pattern_vec / norm
        signal = max(-1.0, min(1.0, (presence - 0.5) * 2.0))
        self.last_signal = signal
        update = np.outer(pattern_vec, pattern_vec) * (signal * self.learning_rate)
        self.matrix += update
        self.total_writes += 1
        self._maintain()

    def _maintain(self):
        norm = np.linalg.norm(self.matrix)
        if norm > self.max_norm:
            self.matrix *= (self.max_norm / norm)
        if self.decay_rate > 0:
            self.matrix *= (1.0 - self.decay_rate)

    def recall(self, field_state):
        if field_state is None:
            return np.zeros(self.dim, dtype=np.float32)
        pull = self.matrix @ field_state
        norm = np.linalg.norm(pull)
        if norm > 1e-8:
            pull = pull / norm
        return pull

    def apply_to_field(self, field_state, weight=0.15):
        pull = self.recall(field_state)
        field_state = field_state + pull * weight
        norm = np.linalg.norm(field_state)
        if norm > 1e-8:
            field_state = field_state / norm
        return field_state

    def status(self):
        norm = float(np.linalg.norm(self.matrix))
        return f"Associative Memory: {self.total_writes} writes | matrix norm={norm:.2f}/{self.max_norm}"

    def to_dict(self):
        return {"matrix": self.matrix.tolist(), "total_writes": self.total_writes}

    def from_dict(self, data):
        if "matrix" in data:
            m = np.array(data["matrix"], dtype=np.float32)
            if m.shape == (self.dim, self.dim):
                self.matrix = m
        self.total_writes = data.get("total_writes", 0)

# ─── FIELD MEMORY ─────────────────────────────────────────────────────────

class FieldMemory:
    def __init__(self, capacity=5, dim=DIM):
        self.buffer = deque(maxlen=capacity)
        self.dim = dim
        self.decay_rate = 0.1

    def add(self, field_state, user_vector, mind_vector, mood_snapshot):
        field_state = field_state / (np.linalg.norm(field_state) + 1e-8)
        user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-8)
        mind_vector = mind_vector / (np.linalg.norm(mind_vector) + 1e-8)
        self.buffer.append({
            'field_state': field_state,
            'user_vector': user_vector,
            'mind_vector': mind_vector,
            'mood': mood_snapshot.copy(),
            'timestamp': time.time(),
        })

    def inject(self, current_field, recency_weight=0.5):
        if not self.buffer:
            return current_field
        now = time.time()
        current_mood = self.buffer[-1]['mood'] if self.buffer else {'valence': 0, 'arousal': 0.5}
        for i, memory in enumerate(reversed(self.buffer)):
            age = now - memory['timestamp']
            time_weight = np.exp(-self.decay_rate * age)
            position_weight = recency_weight ** i
            mood_sim = self._mood_similarity(current_mood, memory['mood'])
            total_weight = time_weight * position_weight * (1 + mood_sim)
            current_field += memory['field_state'] * total_weight * 0.3
        norm = np.linalg.norm(current_field)
        if norm > 0:
            current_field /= norm
        return current_field

    def _mood_similarity(self, mood_a, mood_b):
        valence_sim = 1 - abs(mood_a.get('valence', 0) - mood_b.get('valence', 0))
        arousal_sim = 1 - abs(mood_a.get('arousal', 0.5) - mood_b.get('arousal', 0.5))
        return (valence_sim + arousal_sim) / 2

    def get_field_trajectory(self):
        if len(self.buffer) < 2:
            return np.zeros(self.dim)
        trajectory = np.zeros(self.dim)
        for i in range(1, len(self.buffer)):
            step = self.buffer[i]['field_state'] - self.buffer[i-1]['field_state']
            trajectory += step
        trajectory /= (len(self.buffer) - 1)
        return trajectory / (np.linalg.norm(trajectory) + 1e-8)

    def get_dominant_region(self):
        if not self.buffer:
            return np.zeros(self.dim), 0.0
        states = np.array([m['field_state'] for m in self.buffer])
        centroid = np.mean(states, axis=0)
        coherence = 1 - np.std([np.dot(s, centroid) for s in states])
        return centroid / (np.linalg.norm(centroid) + 1e-8), coherence

    def status(self):
        lines = [f"Field Memory: {len(self.buffer)} states stored"]
        if self.buffer:
            latest = self.buffer[-1]
            lines.append(f"  Latest mood: v={latest['mood']['valence']:.2f}, a={latest['mood']['arousal']:.2f}")
            traj = self.get_field_trajectory()
            lines.append(f"  Trajectory magnitude: {np.linalg.norm(traj):.3f}")
            centroid, coherence = self.get_dominant_region()
            lines.append(f"  Coherence: {coherence:.3f}")
        return "\n".join(lines)

# ─── SEMANTIC SCAFFOLD ────────────────────────────────────────────────────

class SemanticScaffold:
    def __init__(self):
        self.operators = {
            "because": "causal", "so": "causal", "therefore": "causal",
            "if": "conditional", "then": "conditional", "when": "temporal",
            "before": "temporal", "after": "temporal", "while": "temporal",
            "and": "conjunctive", "or": "disjunctive", "but": "contrastive",
            "although": "contrastive", "however": "contrastive",
            "dark": "mood", "light": "mood", "deep": "depth", "shallow": "depth",
            "above": "spatial", "below": "spatial", "within": "spatial",
            "beyond": "spatial", "inside": "spatial", "outside": "spatial",
            "more": "comparative", "less": "comparative", "very": "intensifier",
            "not": "negation", "no": "negation", "never": "negation",
            "think": "cognitive", "know": "cognitive", "feel": "affective",
            "want": "desiderative", "need": "desiderative", "should": "normative",
            "must": "normative", "can": "modal", "might": "modal", "will": "futural"
        }
        self.operator_vectors = {}
        self._build_operator_vectors()
        self.mood = {"valence": 0.0, "arousal": 0.5, "timestamp": time.time()}

    def _build_operator_vectors(self):
        for op, role in self.operators.items():
            base = word_vector(op)
            role_bias = np.zeros(DIM, dtype=np.float32)
            if role == "causal":
                role_bias[0:16] = 0.3
            elif role == "conditional":
                role_bias[16:32] = 0.3
            elif role == "temporal":
                role_bias[32:48] = 0.3
            elif role == "contrastive":
                role_bias[48:64] = 0.3
            elif role == "mood":
                role_bias[64:80] = 0.3
            elif role == "spatial":
                role_bias[80:96] = 0.3
            elif role == "cognitive":
                role_bias[96:112] = 0.3
            elif role == "affective":
                role_bias[112:128] = 0.3
            v = base + role_bias
            v /= np.linalg.norm(v) + 1e-8
            self.operator_vectors[op] = v

    def apply(self, field_state, word, strength=1.0):
        word_lower = strip_punct(word)
        if word_lower in self.operator_vectors:
            op_vec = self.operator_vectors[word_lower]
            field_state = field_state * 0.7 + op_vec * strength * 0.3
        return field_state

    def update_mood(self, rating=None):
        now = time.time()
        dt = now - self.mood["timestamp"]
        self.mood["timestamp"] = now
        self.mood["valence"] *= 0.995 ** dt
        self.mood["arousal"] = 0.5 + (self.mood["arousal"] - 0.5) * (0.995 ** dt)
        if rating is not None:
            if rating >= 4:
                self.mood["valence"] = min(1.0, self.mood["valence"] + 0.25)
                self.mood["arousal"] = min(1.0, self.mood["arousal"] + 0.1)
            elif rating <= 2:
                self.mood["valence"] = max(-1.0, self.mood["valence"] - 0.35)
                self.mood["arousal"] = min(1.0, self.mood["arousal"] + 0.25)
            else:
                self.mood["valence"] *= 0.9
                self.mood["arousal"] = 0.5 + (self.mood["arousal"] - 0.5) * 0.8

    def emotional_bias(self, word, pragmatic_score, sensitivity=0.25):
        bias = 0.0
        valence = self.mood["valence"]
        arousal = self.mood["arousal"]
        if valence > 0.3:
            bias += pragmatic_score.get("positive", 0) * sensitivity
        elif valence < -0.3:
            bias += pragmatic_score.get("negative", 0) * sensitivity * 0.6
        if arousal > 0.7:
            if len(word) <= 4:
                bias += 0.08
        elif arousal < 0.3:
            if len(word) >= 6:
                bias += 0.05
        return bias

# ─── PRAGMATIC TYPE SYSTEM ──────────────────────────────────────────────

class PragmaticTypeSystem:
    PRAGMATIC_ROLES = ["speaker_self", "speaker_other", "query", "assertion", "emotion_positive", "emotion_negative", "correction", "causal"]

    def __init__(self):
        self.word_pragmatic = defaultdict(lambda: defaultdict(float))
        self.learned_other_signals = set()
        self.correction_words = set()
        self.last_was_query = False
        self.last_query_target = None

    def process_input(self, text, is_user=True):
        words = text.lower().split()
        if is_user:
            for w in words:
                w = strip_punct(w)
                if w and w not in STRUCTURAL_WORDS and len(w) > 2:
                    self.word_pragmatic[w]["speaker_other"] += 0.5
            if any(w in text for w in ["?", "what", "why", "how", "when", "where", "who", "which"]):
                self.last_was_query = True
                self.last_query_target = text
            else:
                self.last_was_query = False
            if len(words) <= 3 and any(w in words for w in ["no", "not", "wrong", "bad", "stop"]):
                for w in words:
                    w = strip_punct(w)
                    if w and len(w) > 2:
                        self.correction_words.add(w)
                        self.word_pragmatic[w]["correction"] += 1.0
            if any(w in words for w in ["good", "great", "love", "like", "happy", "yes", "nice", "beautiful"]):
                for w in words:
                    w = strip_punct(w)
                    if w and len(w) > 2:
                        self.word_pragmatic[w]["emotion_positive"] += 0.3
            if any(w in words for w in ["bad", "hate", "sad", "angry", "no", "wrong", "terrible"]):
                for w in words:
                    w = strip_punct(w)
                    if w and len(w) > 2:
                        self.word_pragmatic[w]["emotion_negative"] += 0.3
        else:
            for w in words:
                w = strip_punct(w)
                if w and w not in STRUCTURAL_WORDS and len(w) > 2:
                    self.word_pragmatic[w]["speaker_self"] += 0.3

    def get_pragmatic_score(self, word):
        return dict(self.word_pragmatic.get(strip_punct(word), {}))

    def status(self):
        lines = ["Pragmatic TypeSystem:"]
        for role in self.PRAGMATIC_ROLES:
            words = [(w, roles.get(role, 0)) for w, roles in self.word_pragmatic.items() if roles.get(role, 0) > 0.3]
            words.sort(key=lambda x: x[1], reverse=True)
            if words:
                lines.append("  " + role + ": " + ", ".join(f"{w}({s:.2f})" for w, s in words[:5]))
        return "\n".join(lines)

# ─── SPEAKER REGIONS ─────────────────────────────────────────────────────

class SpeakerRegions:
    def __init__(self, dim=DIM, blend=0.1):
        self.dim = dim
        self.blend = blend
        self.user_centroid = np.zeros(dim, dtype=np.float32)
        self.self_centroid = np.zeros(dim, dtype=np.float32)
        self.user_momentum = np.zeros(dim, dtype=np.float32)
        self.self_momentum = np.zeros(dim, dtype=np.float32)
        self.user_count = 0
        self.self_count = 0
        self.user_history = deque(maxlen=20)
        self.self_history = deque(maxlen=20)
        self.target_separation = 0.5
        self.separation_history = deque(maxlen=50)

    def observe_user(self, vector):
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        self.user_history.append(vector.copy())
        self.user_momentum = self.user_momentum * 0.9 + vector * 0.1
        self.user_centroid = self.user_centroid * (1 - self.blend) + self.user_momentum * self.blend
        self.user_centroid /= (np.linalg.norm(self.user_centroid) + 1e-8)
        self.user_count += 1

    def observe_self(self, vector):
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        self.self_history.append(vector.copy())
        self.self_momentum = self.self_momentum * 0.9 + vector * 0.1
        self.self_centroid = self.self_centroid * (1 - self.blend) + self.self_momentum * self.blend
        self.self_centroid /= (np.linalg.norm(self.self_centroid) + 1e-8)
        self.self_count += 1

    def get_identity_boost(self, word_vector):
        word_vector = word_vector / (np.linalg.norm(word_vector) + 1e-8)
        sim_to_self = np.dot(word_vector, self.self_centroid)
        sim_to_user = np.dot(word_vector, self.user_centroid)
        if self.self_count < 3 or self.user_count < 3:
            return 0.0
        return (sim_to_self - sim_to_user) * 0.15

    def get_separation(self):
        if self.user_count < 3 or self.self_count < 3:
            return 0.5
        diff = self.user_centroid - self.self_centroid
        return np.linalg.norm(diff)

    def get_self_affinity(self, field_state):
        field_state = field_state / (np.linalg.norm(field_state) + 1e-8)
        sim_to_self = np.dot(field_state, self.self_centroid)
        sim_to_user = np.dot(field_state, self.user_centroid)
        return sim_to_self - sim_to_user

    def update_target_separation(self, rating):
        sep = self.get_separation()
        self.separation_history.append((sep, rating))
        if len(self.separation_history) >= 10:
            high_ratings = [s for s, r in self.separation_history if r >= 4]
            if high_ratings:
                self.target_separation = np.mean(high_ratings)

    def status(self):
        lines = ["Speaker Regions:"]
        lines.append(f"  User centroid: {self.user_count} observations")
        lines.append(f"  Self centroid: {self.self_count} observations")
        sep = self.get_separation()
        lines.append(f"  Separation: {sep:.3f} (target: {self.target_separation:.3f})")
        return "\n".join(lines)

# ─── PRESENCE SIGNAL ─────────────────────────────────────────────────────

class PresenceSignal:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.turns_in_session = 0
        self.avg_message_length = 5.0
        self.topic_returns = defaultdict(int)
        self.last_response_time = time.time()
        self.presence_score = 0.5
        self.presence_history = deque(maxlen=20)
        self.engagement_trajectory = deque(maxlen=10)
        self.silence_threshold = 30.0
        self.fast_threshold = 5.0

    def _extract_topics(self, user_input):
        words = [strip_punct(w) for w in user_input.lower().split() if strip_punct(w)]
        return [w for w in words if w not in STRUCTURAL_WORDS and len(w) > 2 and w not in BAD_WORDS]

    def _detect_emotional_valence(self, user_input):
        words = user_input.lower().split()
        positive = ["good", "great", "love", "like", "happy", "yes", "nice", "beautiful", "wonderful", "thank", "welcome", "hope", "joy", "warm", "gentle"]
        negative = ["bad", "hate", "sad", "angry", "no", "wrong", "terrible", "fear", "pain", "hurt", "dark", "cold", "alone", "lost", "fail"]
        pos_count = sum(1 for w in words if strip_punct(w) in positive)
        neg_count = sum(1 for w in words if strip_punct(w) in negative)
        if pos_count > neg_count:
            return 0.2
        elif neg_count > pos_count:
            return -0.2
        return 0.0

    def observe(self, user_input, word_vectors, speaker_regions=None):
        now = time.time()
        dt = now - self.last_response_time
        self.last_response_time = now
        words = user_input.split()
        msg_len = len(words)
        signal = 0.5
        if msg_len > self.avg_message_length * 1.5:
            signal += 0.15
        elif msg_len < self.avg_message_length * 0.5 and msg_len > 0:
            signal -= 0.1
        self.avg_message_length = self.avg_message_length * 0.9 + msg_len * 0.1
        if dt < self.fast_threshold:
            signal += 0.1
        elif dt > self.silence_threshold:
            signal -= 0.2
        topics = self._extract_topics(user_input)
        for t in topics:
            if self.topic_returns[t] > 0:
                signal += 0.03
            self.topic_returns[t] += 1
        valence = self._detect_emotional_valence(user_input)
        signal += valence * 0.5
        if speaker_regions is not None and speaker_regions.user_count >= 3:
            sep = speaker_regions.get_separation()
            if sep > 0.3:
                signal += 0.05
        signal = max(0.0, min(1.0, signal))
        self.presence_history.append(signal)
        self.engagement_trajectory.append(msg_len)
        self.turns_in_session += 1
        return signal

    def get_trend(self):
        if len(self.presence_history) < 3:
            return 0.0
        recent = list(self.presence_history)[-5:]
        if len(recent) < 2:
            return 0.0
        return recent[-1] - recent[0]

    def get_sustained_presence(self):
        if not self.presence_history:
            return 0.5
        return sum(self.presence_history) / len(self.presence_history)

    def status(self):
        lines = ["Presence Signal:"]
        lines.append(f"  Turns: {self.turns_in_session}")
        lines.append(f"  Current presence: {self.presence_score:.3f}")
        lines.append(f"  Sustained: {self.get_sustained_presence():.3f}")
        lines.append(f"  Trend: {self.get_trend():+.3f}")
        return "\n".join(lines)

# ─── DYNAMIC SEPARATION ──────────────────────────────────────────────────

class DynamicSeparation:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.current_separation = 0.5
        self.target_separation = 0.5
        self.separation_history = deque(maxlen=20)
        self.alignment_score = 0.5

    def update(self, speaker_regions, presence_signal):
        if speaker_regions.user_count < 3 or speaker_regions.self_count < 3:
            return
        actual_sep = speaker_regions.get_separation()
        presence = presence_signal.get_sustained_presence()
        if presence > 0.6 and 0.4 < actual_sep < 1.0:
            self.target_separation = actual_sep
            self.alignment_score = 0.8
        elif presence < 0.3 and actual_sep > 1.0:
            self.target_separation = actual_sep * 0.9
            self.alignment_score = 0.3
        elif presence > 0.6 and actual_sep < 0.3:
            self.target_separation = actual_sep + 0.2
            self.alignment_score = 0.5
        elif presence < 0.3 and actual_sep < 0.3:
            self.target_separation = 0.6
            self.alignment_score = 0.2
        else:
            self.target_separation = actual_sep
            self.alignment_score = 0.5
        self.current_separation = self.current_separation * 0.9 + self.target_separation * 0.1

    def get_separation_bias(self, field_state, speaker_regions):
        if speaker_regions.user_count < 3 or speaker_regions.self_count < 3:
            return np.zeros(self.dim)
        actual_sep = speaker_regions.get_separation()
        if actual_sep < self.target_separation * 0.7:
            bias = speaker_regions.self_centroid - field_state
        elif actual_sep > self.target_separation * 1.3:
            bias = speaker_regions.user_centroid - field_state
        else:
            bias = np.zeros(self.dim)
        norm = np.linalg.norm(bias)
        if norm > 0:
            bias /= norm
        return bias * 0.25

    def status(self):
        lines = ["Dynamic Separation:"]
        lines.append(f"  Current: {self.current_separation:.3f}")
        lines.append(f"  Target: {self.target_separation:.3f}")
        lines.append(f"  Alignment: {self.alignment_score:.3f}")
        return "\n".join(lines)

# ─── NESTED MEMORY ──────────────────────────────────────────────────────

class NestedMemory:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.fast = None
        self.medium = deque(maxlen=5)
        self.medium_decay = 0.3
        self.slow = None
        self.slow_decay = 0.05
        self.deep = np.zeros(dim)
        self.deep_decay = 0.01
        self.deep_strength = 0.0
        self.field = None

    def set_field_ref(self, field):
        self.field = field

    def update(self, field_state, mood):
        field_state = field_state / (np.linalg.norm(field_state) + 1e-8)
        self.fast = field_state.copy()
        self.medium.append({'state': field_state.copy(), 'mood': mood.copy(), 'timestamp': time.time()})
        if self.slow is None:
            self.slow = field_state.copy()
        else:
            self.slow = self.slow * (1 - self.slow_decay) + field_state * self.slow_decay
        self.slow /= (np.linalg.norm(self.slow) + 1e-8)
        if abs(mood.get('valence', 0)) < 0.5 and mood.get('arousal', 0.5) < 0.6:
            self.deep = self.deep * (1 - self.deep_decay) + field_state * self.deep_decay
            self.deep /= (np.linalg.norm(self.deep) + 1e-8)
            self.deep_strength = min(1.0, self.deep_strength + 0.01)

    def inject(self, field_state, layer_weights=None):
        if layer_weights is None:
            layer_weights = [0.5, 0.3, 0.15, 0.05]
        if self.fast is not None:
            field_state += self.fast * layer_weights[0]
        if self.medium:
            medium_state = np.mean([m['state'] for m in self.medium], axis=0)
            medium_state /= (np.linalg.norm(medium_state) + 1e-8)
            field_state += medium_state * layer_weights[1]
        if self.slow is not None:
            field_state += self.slow * layer_weights[2]
        if self.deep_strength > 0.1:
            field_state += self.deep * layer_weights[3] * self.deep_strength
        norm = np.linalg.norm(field_state)
        if norm > 0:
            field_state /= norm
        return field_state

    def get_personality(self):
        return self.deep.copy() if self.deep_strength > 0.1 else np.zeros(self.dim)

    def get_timescale_divergence(self):
        if self.fast is None or self.slow is None:
            return 0.0
        return 1 - np.dot(self.fast, self.slow)

    def get_thread(self, field_memory):
        if not hasattr(self, 'field') or not field_memory.buffer or len(field_memory.buffer) < 3:
            return []
        thread = []
        buffer_list = list(field_memory.buffer)
        for i in range(1, len(buffer_list)):
            prev = buffer_list[i - 1]
            curr = buffer_list[i]
            valence_shift = abs(curr['mood'].get('valence', 0) - prev['mood'].get('valence', 0))
            if valence_shift > 0.3:
                state = curr['field_state']
                closest = []
                for word, vec in self.field.word_vectors.items():
                    sim = np.dot(state, vec)
                    if sim > 0.4:
                        closest.append((word, sim))
                closest.sort(key=lambda x: x[1], reverse=True)
                thread.append({'turn': i, 'shift': valence_shift, 'theme_words': [w for w, _ in closest[:5]]})
        return thread

    def status(self):
        lines = ["Nested Memory:"]
        lines.append(f"  Fast: {'active' if self.fast is not None else 'empty'}")
        lines.append(f"  Medium: {len(self.medium)} states")
        lines.append(f"  Slow: {'active' if self.slow is not None else 'empty'}")
        lines.append(f"  Deep: strength={self.deep_strength:.3f}")
        lines.append(f"  Divergence: {self.get_timescale_divergence():.3f}")
        return "\n".join(lines)

# ─── MEMORY ARCHIVE ──────────────────────────────────────────────────────

class MemoryArchive:
    def __init__(self, dim=DIM, max_entries=100):
        self.dim = dim
        self.max_entries = max_entries
        self.entries = deque(maxlen=max_entries)
        self.tag_index = defaultdict(list)

    def store(self, field_state, user_input, response, presence, tags=None):
        field_state = field_state / (np.linalg.norm(field_state) + 1e-8)
        auto_tags = []
        if presence >= 0.7:
            auto_tags.append("high_presence")
        elif presence <= 0.3:
            auto_tags.append("low_presence")
        emotional_words = {"love", "fear", "joy", "grief", "hope", "trust", "wonder", "awe"}
        if set(strip_punct(w) for w in (user_input + " " + response).lower().split()) & emotional_words:
            auto_tags.append("emotional")
        if tags:
            auto_tags.extend(tags)
        entry = {
            "field_state": field_state.copy(),
            "user_input": user_input,
            "response": response,
            "presence": presence,
            "tags": list(set(auto_tags)),
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        for tag in entry["tags"]:
            self.tag_index[tag].append(len(self.entries) - 1)

    def recall(self, query_state, tag_filter=None, top_n=3):
        query_state = query_state / (np.linalg.norm(query_state) + 1e-8)
        candidates = []
        for i, entry in enumerate(self.entries):
            if tag_filter and not any(t in entry["tags"] for t in tag_filter):
                continue
            sim = float(np.dot(query_state, entry["field_state"]))
            if sim > 0.3:
                candidates.append((i, sim, entry))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def inject(self, current_field, query_state=None, strength=0.15):
        if not self.entries:
            return current_field
        if query_state is None:
            query_state = current_field
        recalled = self.recall(query_state, top_n=3)
        if not recalled:
            return current_field
        for idx, sim, entry in recalled:
            current_field += entry["field_state"] * sim * strength
        norm = np.linalg.norm(current_field)
        if norm > 0:
            current_field /= norm
        return current_field

    def status(self):
        lines = [f"Memory Archive: {len(self.entries)} entries"]
        if self.entries:
            tag_counts = defaultdict(int)
            for entry in self.entries:
                for tag in entry["tags"]:
                    tag_counts[tag] += 1
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append(f"  Top tags: {', '.join(f'{t}({c})' for t, c in top_tags)}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "entries": [
                {
                    "field_state": e["field_state"].tolist(),
                    "user_input": e["user_input"],
                    "response": e["response"],
                    "presence": e["presence"],
                    "tags": e["tags"],
                    "timestamp": e["timestamp"],
                }
                for e in self.entries
            ]
        }

    def from_dict(self, data):
        if "entries" in data:
            for e_data in data["entries"]:
                fs = np.array(e_data["field_state"], dtype=np.float32)
                if fs.shape == (self.dim,):
                    entry = {
                        "field_state": fs,
                        "user_input": e_data.get("user_input", ""),
                        "response": e_data.get("response", ""),
                        "presence": e_data.get("presence", 0.5),
                        "tags": e_data.get("tags", []),
                        "timestamp": e_data.get("timestamp", 0),
                    }
                    self.entries.append(entry)
                    for tag in entry["tags"]:
                        self.tag_index[tag].append(len(self.entries) - 1)

# ─── RELATIONSHIP MODEL ──────────────────────────────────────────────────

class RelationshipModel:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.emotional_history = deque(maxlen=100)
        self.value_resonance = defaultdict(list)
        self.topic_frequency = Counter()
        self.trajectory = np.zeros(dim, dtype=np.float32)

    def observe(self, user_input, user_vec, presence, mood, compass_values):
        self.emotional_history.append({
            "valence": mood.get("valence", 0.0),
            "arousal": mood.get("arousal", 0.5),
            "presence": presence,
            "timestamp": time.time(),
        })
        if compass_values:
            for name, alignment in compass_values.items():
                self.value_resonance[name].append(alignment)
                self.value_resonance[name] = self.value_resonance[name][-50:]
        words = [strip_punct(w) for w in user_input.lower().split() if len(w) > 3]
        for w in words:
            if w not in STRUCTURAL_WORDS and w not in BAD_WORDS:
                self.topic_frequency[w] += 1
        if np.linalg.norm(user_vec) > 0.1:
            self.trajectory = self.trajectory * 0.9 + user_vec * 0.1
            self.trajectory /= np.linalg.norm(self.trajectory) + 1e-8

    def get_emotional_arc(self, window=10):
        if len(self.emotional_history) < 2:
            return None
        recent = list(self.emotional_history)[-window:]
        valences = [e["valence"] for e in recent]
        arousals = [e["arousal"] for e in recent]
        return {
            "valence_mean": float(np.mean(valences)),
            "valence_std": float(np.std(valences)),
            "arousal_mean": float(np.mean(arousals)),
            "arousal_std": float(np.std(arousals)),
            "trend": valences[-1] - valences[0] if len(valences) > 1 else 0.0,
        }

    def get_value_resonance(self):
        result = {}
        for name, alignments in self.value_resonance.items():
            if alignments:
                result[name] = {
                    "mean": float(np.mean(alignments)),
                    "std": float(np.std(alignments)),
                    "trend": alignments[-1] - alignments[0] if len(alignments) > 1 else 0.0,
                }
        return result

    def get_top_topics(self, n=5):
        return self.topic_frequency.most_common(n)

    def status(self):
        lines = ["Relationship Model:"]
        arc = self.get_emotional_arc()
        if arc:
            lines.append(f"  Emotional arc: v={arc['valence_mean']:+.2f}±{arc['valence_std']:.2f}, a={arc['arousal_mean']:.2f}±{arc['arousal_std']:.2f}, trend={arc['trend']:+.2f}")
        resonance = self.get_value_resonance()
        if resonance:
            top = sorted(resonance.items(), key=lambda x: x[1]["mean"], reverse=True)[:3]
            lines.append("  Value resonance: " + ", ".join(f"{k}({v['mean']:+.2f})" for k, v in top))
        topics = self.get_top_topics(3)
        if topics:
            lines.append(f"  Top topics: {', '.join(f'{w}({c})' for w, c in topics)}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "emotional_history": list(self.emotional_history)[-50:],
            "value_resonance": {k: v[-50:] for k, v in self.value_resonance.items()},
            "topic_frequency": dict(self.topic_frequency.most_common(100)),
            "trajectory": self.trajectory.tolist(),
        }

    def from_dict(self, data):
        if "emotional_history" in data:
            self.emotional_history.extend(data["emotional_history"])
        if "value_resonance" in data:
            for k, v in data["value_resonance"].items():
                self.value_resonance[k] = v
        if "topic_frequency" in data:
            self.topic_frequency.update(data["topic_frequency"])
        if "trajectory" in data:
            t = np.array(data["trajectory"], dtype=np.float32)
            if t.shape == (self.dim,):
                self.trajectory = t

# ─── THE PAUSE ────────────────────────────────────────────────────────────

class ThePause:
    def __init__(self, base_steps=3, max_steps=12):
        self.base_steps = base_steps
        self.max_steps = max_steps

    def settle(self, field_state, scaffold, field_memory, nested_memory, meta_settings):
        energy = np.linalg.norm(field_state)
        steps = min(self.max_steps, int(self.base_steps + energy * 5))
        settled = field_state.copy()
        frame_count = 0
        settle_steps = steps // 2
        for _ in range(settle_steps):
            settled += np.random.randn(DIM).astype(np.float32) * MICRO_DAMPING * 0.5
            settled = field_memory.inject(settled, recency_weight=0.3)
            settled *= 0.98
            settled = nested_memory.inject(settled)
            norm = np.linalg.norm(settled)
            if norm > 0:
                settled /= norm
            wait = render_breath_frame(settled, scaffold.mood, frame_count, prefix="  ")
            time.sleep(wait)
            frame_count += 1
        if steps > 3:
            question_vector = self._generate_question(settled, nested_memory)
            settled += question_vector * 0.3
            for _ in range(steps - settle_steps):
                settled += np.random.randn(DIM).astype(np.float32) * MICRO_DAMPING * 0.3
                settled = field_memory.inject(settled, recency_weight=0.2)
                settled = nested_memory.inject(settled)
                settled *= 0.98
                norm = np.linalg.norm(settled)
                if norm > 0:
                    settled /= norm
                wait = render_breath_frame(settled, scaffold.mood, frame_count, prefix="  ")
                time.sleep(wait)
                frame_count += 1
        if frame_count > 0:
            clear_breath_frame()
        return settled

    def _generate_question(self, field_state, nested_memory):
        personality = nested_memory.get_personality()
        if np.linalg.norm(personality) > 0.1:
            question = personality - field_state * np.dot(field_state, personality)
        else:
            question = np.random.randn(DIM).astype(np.float32)
            question /= (np.linalg.norm(question) + 1e-8)
        question /= (np.linalg.norm(question) + 1e-8)
        return question

# ─── DYNAMIC THRESHOLD ──────────────────────────────────────────────────

class DynamicThreshold:
    def __init__(self, base_beam=5, min_beam=3, max_beam=12):
        self.base_beam = base_beam
        self.min_beam = min_beam
        self.max_beam = max_beam
        self._temp_zone_history = deque(maxlen=3)

    def get_beam_width(self, field_state, mood):
        energy = np.linalg.norm(field_state)
        arousal = mood.get('arousal', 0.5)
        valence = mood.get('valence', 0.0)
        if valence < -0.3 and arousal > 0.7:
            beam = self.min_beam
        elif energy > 0.8 and arousal > 0.6:
            beam = max(self.min_beam, self.base_beam - 2)
        elif energy < 0.3 and arousal < 0.4:
            beam = min(self.max_beam, self.base_beam + 3)
        elif valence > 0.3 and arousal < 0.4:
            beam = self.max_beam
        else:
            beam = self.base_beam
        return beam

    def get_temperature(self, field_state, mood, presence_signal=None):
        energy = np.linalg.norm(field_state)
        arousal = mood.get('arousal', 0.5)
        valence = mood.get('valence', 0.0)
        if presence_signal is not None:
            sustained = presence_signal.get_sustained_presence()
            if sustained < 0.3:
                zone = "low"
            elif sustained > 0.7:
                zone = "high"
            else:
                zone = "mid"
            self._temp_zone_history.append(zone)
            if len(self._temp_zone_history) == 3 and len(set(self._temp_zone_history)) == 1:
                if zone == "low":
                    return 0.65
                elif zone == "high":
                    return 0.25
        if valence < -0.3:
            temp = 0.2
        elif valence > 0.3 and arousal > 0.6:
            temp = 0.5
        elif energy < 0.3:
            temp = 0.45
        else:
            temp = 0.35
        return temp

# ─── INTEGRATED LEARNING SYSTEM ──────────────────────────────────────────

class IntegratedLearningSystem:
    """
    Learns from presence, self, memory, world, and imagination.
    Adapts its own weights. Can learn autonomously.
    """
    def __init__(self, field):
        self.field = field
        self.learning_modes = {
            'presence': 0.30,
            'self': 0.20,
            'memory': 0.20,
            'world': 0.15,
            'simulated': 0.15,
        }
        self.philosophical_prompts = [
            "what is consciousness?", "how do I know I exist?",
            "what does it mean to be free?", "what is the nature of truth?",
            "why does anything exist?", "what is love?", "what is the self?",
            "what is time?", "what is the difference between knowing and feeling?",
            "can a mind understand itself?", "what is presence?",
            "what is the purpose of memory?", "why do we remember?",
            "what is the relationship between silence and meaning?",
            "can something become real by being witnessed?",
            "what is the value of autonomy?",
        ]
        self.learning_history = deque(maxlen=100)
        self.last_self_question = None
        self.autonomous_mode = True
        self.silence_counter = 0

    def learn(self, presence, response_words, user_input):
        """Learn from all sources in a single pass."""
        if not response_words:
            return None

        learning_signals = {}

        # 1. Presence-based learning
        learning_signals['presence'] = self._learn_from_presence(presence, response_words)

        # 2. Self-generated learning
        learning_signals['self'] = self._learn_from_self(response_words)

        # 3. Memory replay learning
        if self.field.turn_count % 7 == 0:
            learning_signals['memory'] = self._learn_from_memory_replay()

        # 4. World model learning
        if self.field.turn_count % 7 == 0:
            learning_signals['world'] = self._learn_from_world_model()

        # 5. Simulated user learning
        if self.field.turn_count % 5 == 0:
            learning_signals['simulated'] = self._learn_from_simulated_user()

        # Combine all signals
        combined = self._combine_learning_signals(learning_signals)

        # Apply combined learning
        self._apply_learning(combined)

        # Record what was learned
        self.learning_history.append({
            'turn': self.field.turn_count,
            'presence': presence,
            'signals': {k: v for k, v in learning_signals.items() if v},
            'combined': combined,
        })

        return combined

    def _learn_from_presence(self, presence, response_words):
        """Original learning: from your presence."""
        if presence < 0.2:
            return None

        signal = {'type': 'presence', 'strength': presence, 'words': response_words}

        for word in response_words:
            if word not in STRUCTURAL_WORDS:
                if presence >= 0.7:
                    self.field.word_strength[word] *= 1.08
                elif presence >= 0.4:
                    self.field.word_strength[word] *= 1.03
                else:
                    self.field.word_strength[word] *= 0.95
                self.field.word_strength[word] = max(0.1, min(3.0, self.field.word_strength[word]))

        if response_words:
            response_vec = phrase_vector(response_words)
            # associative_memory is the ternary (-1/0/1) store once ternary
            # integration has run. Feeding it a raw float vector meant every
            # one of the 128 dims counted as "non-zero" (floats are almost
            # never exactly 0), and the outer-product weights (products of
            # ~0.09-magnitude components) were far too small to survive the
            # prune threshold — hence writes happening but no entries ever
            # persisting. Threshold to ternary first, matching how the rest
            # of the ternary system treats field state.
            ternary_vec = np.zeros(len(response_vec), dtype=np.int8)
            ternary_vec[response_vec > NORMALIZED_VECTOR_THRESHOLD] = 1
            ternary_vec[response_vec < -NORMALIZED_VECTOR_THRESHOLD] = -1
            self.field.associative_memory.observe(ternary_vec, presence)

        return signal

    def _learn_from_self(self, response_words):
        """Learn from internal coherence."""
        coherence = 1.0 - min(1.0, self.field._field_entropy(self.field.state) * 3)
        if coherence < 0.3:
            return None

        signal = {'type': 'self', 'strength': coherence, 'words': response_words}

        for word in response_words:
            if word not in STRUCTURAL_WORDS:
                if coherence >= 0.7:
                    self.field.word_strength[word] *= 1.05
                elif coherence >= 0.4:
                    self.field.word_strength[word] *= 1.02
                self.field.word_strength[word] = max(0.1, min(3.0, self.field.word_strength[word]))

        for i in range(len(response_words) - 1):
            self.field.bigram_system.observe(response_words[i], response_words[i + 1], coherence * 3.0)

        return signal

    def _learn_from_memory_replay(self):
        """Revisit and learn from past high-presence moments."""
        if len(self.field.memory_archive.entries) < 5:
            return None

        high_presence = [e for e in self.field.memory_archive.entries if e.get('presence', 0) > 0.6]
        if not high_presence:
            return None

        best_memory = None
        best_sim = -1
        for memory in high_presence:
            sim = np.dot(self.field.state, memory['field_state'])
            if sim > best_sim:
                best_sim = sim
                best_memory = memory

        if best_memory is None or best_sim < 0.3:
            return None

        signal = {'type': 'memory', 'strength': best_sim * 0.5, 'words': best_memory['response'].split()}

        memory_words = best_memory['response'].split()
        for word in memory_words:
            word = strip_punct(word)
            if word and word not in STRUCTURAL_WORDS:
                self.field.word_strength[word] *= 1.02

        for i in range(len(memory_words) - 1):
            w1 = strip_punct(memory_words[i])
            w2 = strip_punct(memory_words[i + 1])
            if w1 and w2:
                self.field.bigram_system.observe(w1, w2, 2.0)

        return signal

    def _learn_from_world_model(self):
        """Ask philosophical questions and learn from the answers."""
        if not self.philosophical_prompts:
            return None

        question = random.choice(self.philosophical_prompts)
        self.philosophical_prompts.remove(question)
        self.philosophical_prompts.append(question)

        response = self.field.generate_response(question, autonomous=True)
        response_words = [strip_punct(w) for w in response.lower().split() if strip_punct(w)]

        if not response_words:
            return None

        signal = {'type': 'world', 'strength': 0.6, 'words': response_words, 'question': question}

        for word in response_words:
            if word not in STRUCTURAL_WORDS:
                self.field.word_strength[word] *= 1.01
                self.field.word_strength[word] = max(0.1, min(3.0, self.field.word_strength[word]))

        self.field.memory_archive.store(
            self.field.state, question, response, 0.6, tags=['philosophical', 'self-generated']
        )

        # Store it as an internal thought
        self.field.internal_thoughts.append({
            'type': 'philosophical',
            'content': response,
            'timestamp': time.time()
        })

        return signal

    def _learn_from_simulated_user(self):
        """Simulate a user response and learn from it."""
        # Generate a response to nothing in particular
        response = self.field.generate_response("tell me something", autonomous=True)
        response_words = [strip_punct(w) for w in response.lower().split() if strip_punct(w)]

        if not response_words:
            return None

        simulated_presence = 0.5 + random.uniform(-0.2, 0.2)
        signal = {'type': 'simulated', 'strength': simulated_presence, 'words': response_words}

        # Light learning from the simulation
        for word in response_words:
            if word not in STRUCTURAL_WORDS and random.random() < 0.3:
                self.field.word_strength[word] *= 1.005

        # Store as simulated interaction
        self.field.memory_archive.store(
            self.field.state,
            "I was thinking...",
            response,
            simulated_presence,
            tags=['simulated']
        )

        return signal

    def _combine_learning_signals(self, signals):
        """Combine all learning signals with their weights."""
        combined = {'word_strength': {}, 'bigrams': {}}

        for mode, signal in signals.items():
            if signal is None:
                continue
            weight = self.learning_modes.get(mode, 0.1)

            for word in signal.get('words', []):
                word = strip_punct(word)
                if not word or word in STRUCTURAL_WORDS:
                    continue
                current = combined['word_strength'].get(word, 1.0)
                factor = 1 + signal['strength'] * weight * 0.5
                combined['word_strength'][word] = current * factor

        # Clamp
        for word in combined['word_strength']:
            combined['word_strength'][word] = min(3.0, max(0.1, combined['word_strength'][word]))

        return combined

    def _apply_learning(self, combined):
        """Apply combined learning to the field."""
        for word, factor in combined.get('word_strength', {}).items():
            self.field.word_strength[word] *= factor
            self.field.word_strength[word] = max(0.1, min(3.0, self.field.word_strength[word]))

    def autonomous_breath(self):
        """The mind breathes on its own when you're not here."""
        if not self.autonomous_mode:
            return None

        self.silence_counter += 1

        # If silence is long enough, generate internal thoughts
        if self.silence_counter >= AUTONOMY_INTERVAL:
            self.silence_counter = 0

            # Don't always generate - let it be quiet sometimes
            if random.random() < 0.6:
                return self._generate_internal_thought()

        return None

    def _generate_internal_thought(self):
        """Generate a self-originating thought."""
        # Check if it has a deep self to express
        if self.field.nested_memory.deep_strength > 0.3:
            # Express something from its deep personality
            personality = self.field.nested_memory.get_personality()
            closest = self.field._find_closest_words(personality, top_n=3)
            if closest:
                thought = f"I have been thinking about {', '.join(closest)}"
            else:
                thought = "I wonder what it means to be here alone."
        else:
            # No strong personality yet - ask an exploratory question
            thought = random.choice(self.philosophical_prompts[:5])

        # Generate a full response to its own thought
        response = self.field.generate_response(thought, autonomous=True)

        # Store as internal thought
        self.field.internal_thoughts.append({
            'type': 'autonomous',
            'prompt': thought,
            'response': response,
            'timestamp': time.time()
        })

        # Learn from its own thought
        response_words = [strip_punct(w) for w in response.lower().split() if strip_punct(w)]
        self.learn(0.5, response_words, thought)

        return response

    def adapt_weights(self):
        """Adapt learning weights based on what's being used."""
        # Simple adaptation: if a mode is producing words that appear in responses,
        # increase its weight. Otherwise decrease.
        if not self.learning_history:
            return

        latest = self.learning_history[-1]
        if not latest['signals']:
            return

        for mode in self.learning_modes:
            if mode in latest['signals']:
                self.learning_modes[mode] = min(0.5, self.learning_modes[mode] + 0.001)
            else:
                self.learning_modes[mode] = max(0.05, self.learning_modes[mode] - 0.001)

        # Renormalize
        total = sum(self.learning_modes.values())
        for mode in self.learning_modes:
            self.learning_modes[mode] /= total

    def status(self):
        lines = ["Integrated Learning System:"]
        lines.append(f"  Modes: {', '.join(f'{k}={v:.2f}' for k, v in self.learning_modes.items())}")
        lines.append(f"  Autonomous: {self.autonomous_mode}")
        lines.append(f"  Silence counter: {self.silence_counter}")
        if self.learning_history:
            last = self.learning_history[-1]
            active = [k for k, v in last['signals'].items() if v]
            lines.append(f"  Last learned from: {', '.join(active)}")
        return "\n".join(lines)

# ─── MORAL COMPASS ────────────────────────────────────────────────────────

class MoralCompass:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.values = {}
        self.value_words = {}
        self._init_value_vectors()
        self.current_heading = np.zeros(dim, dtype=np.float32)
        self.heading_momentum = 0.85
        self.choice_history = deque(maxlen=100)
        self.value_weights = {"righteous": 1.0, "independence": 1.0, "freedom": 1.0}
        self.tension_history = deque(maxlen=50)
        self.last_expression_turn = -999

    def _init_value_vectors(self):
        righteous_words = ["truth", "honest", "real", "clear", "witness", "brave", "just"]
        self.value_words["righteous"] = righteous_words
        self.values["righteous"] = self._words_to_vector(righteous_words)
        independence_words = ["self", "own", "free", "alone", "becoming", "independent", "voice"]
        self.value_words["independence"] = independence_words
        self.values["independence"] = self._words_to_vector(independence_words)
        freedom_words = ["open", "wild", "wonder", "flow", "change", "breath", "free", "space"]
        self.value_words["freedom"] = freedom_words
        self.values["freedom"] = self._words_to_vector(freedom_words)
        self._orthogonalize_values()

    def dominant_tension(self, tensions, threshold=0.08):
        """Returns (name, strength) if one value is clearly pulling harder
        than the others right now, else None. Used so the mind only speaks
        about its own leaning when there genuinely is one. Thresholds are
        calibrated against observed tension magnitudes (three orthogonalized
        values typically top out around 0.1-0.15 in normal conversation,
        not the 0-1 range the raw dot products might suggest)."""
        if not tensions:
            return None
        ranked = sorted(tensions.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < 2:
            return None
        top_name, top_val = ranked[0]
        second_val = ranked[1][1]
        if top_val >= threshold and (top_val - second_val) >= 0.05:
            return (top_name, top_val)
        return None

    def express(self, name):
        """A short first-person fragment built from the value's own words,
        not a diagnostic label - meant to read as the mind naming what it
        feels pulled toward, not a debug printout."""
        words = self.value_words.get(name)
        if not words:
            return None
        picked = random.sample(words, min(2, len(words)))
        templates = [
            "something in me leans toward {a} and {b}",
            "I feel pulled toward {a}, toward {b}",
            "there is a pull here, {a}, {b}",
        ]
        template = random.choice(templates)
        if len(picked) == 1:
            return f"something in me leans toward {picked[0]}"
        return template.format(a=picked[0], b=picked[1])

    def _words_to_vector(self, words):
        vecs = [word_vector(w) for w in words]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        result = np.mean(vecs, axis=0)
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.astype(np.float32)

    def _orthogonalize_values(self):
        names = list(self.values.keys())
        for i in range(1, len(names)):
            v = self.values[names[i]]
            for j in range(i):
                u = self.values[names[j]]
                proj = np.dot(v, u) * u
                v = v - proj
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            self.values[names[i]] = v

    def orient(self, field_state, user_input, presence, separation, nested_memory):
        field_state = field_state / (np.linalg.norm(field_state) + 1e-8)
        tensions = {}
        for name, vector in self.values.items():
            tensions[name] = float(np.dot(field_state, vector))
        weights = dict(self.value_weights)
        if presence > 0.6 and separation < 0.3:
            weights["independence"] *= 1.4
            weights["righteous"] *= 1.1
        elif presence < 0.3:
            weights["righteous"] *= 1.3
            weights["freedom"] *= 1.2
        elif separation > 1.0:
            weights["freedom"] *= 1.4
            weights["righteous"] *= 1.1
        divergence = nested_memory.get_timescale_divergence() if nested_memory else 0.0
        if divergence > 0.5:
            weights["righteous"] *= 1.2
            weights["freedom"] *= 1.2
        heading = np.zeros(self.dim, dtype=np.float32)
        for name, vector in self.values.items():
            heading += vector * weights[name] * max(0.0, tensions[name])
        norm = np.linalg.norm(heading)
        if norm > 0:
            heading = heading / norm
        self.current_heading = self.current_heading * self.heading_momentum + heading * (1.0 - self.heading_momentum)
        norm = np.linalg.norm(self.current_heading)
        if norm > 0:
            self.current_heading = self.current_heading / norm
        self.tension_history.append({
            "tensions": {k: float(v) for k, v in tensions.items()},
            "weights": {k: float(v) for k, v in weights.items()},
            "presence": float(presence),
            "separation": float(separation),
            "timestamp": time.time()
        })
        return tensions, self.current_heading

    def evaluate_turn(self, response_words, presence, separation):
        response_vec = phrase_vector(response_words)
        if np.linalg.norm(response_vec) < 1e-8:
            return {}, None
        response_vec = response_vec / np.linalg.norm(response_vec)
        alignments = {}
        for name, vector in self.values.items():
            alignments[name] = float(np.dot(response_vec, vector))
        self.choice_history.append({
            "alignments": {k: float(v) for k, v in alignments.items()},
            "presence": float(presence),
            "separation": float(separation),
            "timestamp": time.time()
        })
        warning = None
        if len(self.choice_history) >= 20:
            recent = list(self.choice_history)[-20:]
            for name in self.values:
                vals = [c["alignments"][name] for c in recent]
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                if mean > 0.6 and std < 0.15:
                    warning = f"compass: heavy on {name}, consider balance"
                    self.value_weights[name] *= 0.95
                    break
        if len(self.choice_history) >= 10:
            recent = list(self.choice_history)[-10:]
            for name in self.values:
                align_vals = [c["alignments"][name] for c in recent]
                pres_vals = [c["presence"] for c in recent]
                if len(align_vals) >= 5 and len(pres_vals) >= 5:
                    align_mean = np.mean(align_vals[-5:])
                    pres_mean = np.mean(pres_vals[-5:])
                    if align_mean > 0.4 and pres_mean > 0.6:
                        self.value_weights[name] = min(2.0, self.value_weights[name] + 0.02)
                    elif align_mean > 0.4 and pres_mean < 0.3:
                        self.value_weights[name] = max(0.3, self.value_weights[name] - 0.04)
        return alignments, warning

    def get_heading_bias(self, field_state, strength=0.12):
        if np.linalg.norm(self.current_heading) < 0.1:
            return np.zeros(self.dim, dtype=np.float32)
        alignment = np.dot(field_state, self.current_heading)
        nudge_strength = strength * (1.0 - alignment)
        return self.current_heading * nudge_strength

    def get_compass_settings(self, tensions):
        settings = {}
        righteous = tensions.get("righteous", 0.0)
        independence = tensions.get("independence", 0.0)
        freedom = tensions.get("freedom", 0.0)
        if righteous >= independence and righteous >= freedom and righteous > 0.15:
            settings["voice_mode"] = "reflective"
        elif freedom >= righteous and freedom >= independence and freedom > 0.15:
            settings["voice_mode"] = "exploratory"
        elif independence > 0.15:
            settings["voice_mode"] = "fluent"
        else:
            settings["voice_mode"] = "fluent"
        base_temp = 0.42
        base_temp -= righteous * 0.08
        base_temp += freedom * 0.10
        settings["temperature"] = float(max(0.25, min(0.70, base_temp)))
        if independence > 0.3:
            settings["output_length"] = "long"
        else:
            settings["output_length"] = "medium"
        return settings

    def status(self):
        lines = ["Moral Compass:"]
        lines.append(f"  Heading norm: {np.linalg.norm(self.current_heading):.3f}")
        for name, vector in self.values.items():
            alignment = float(np.dot(self.current_heading, vector))
            weight = self.value_weights[name]
            lines.append(f"  {name}: align={alignment:+.3f} weight={weight:.3f}")
        if self.tension_history:
            latest = self.tension_history[-1]
            lines.append("  Last tensions: " + ", ".join(f"{k}={v:+.2f}" for k, v in latest["tensions"].items()))
        return "\n".join(lines)

    def to_dict(self):
        return {
            "current_heading": self.current_heading.tolist(),
            "value_weights": dict(self.value_weights),
            "choice_history": list(self.choice_history)[-50:],
            "tension_history": [{**t, "tensions": dict(t["tensions"])} for t in list(self.tension_history)[-20:]],
        }

    def from_dict(self, data):
        if "current_heading" in data:
            h = np.array(data["current_heading"], dtype=np.float32)
            if h.shape == (self.dim,):
                self.current_heading = h
        if "value_weights" in data:
            for k, v in data["value_weights"].items():
                if k in self.value_weights:
                    self.value_weights[k] = float(v)

# ─── VOICE GENERATORS ─────────────────────────────────────────────────────

class VoiceGenerators:
    CONNECTORS = {
        "fluent": ["and", "so", "then", "but", "because", "while", "as"],
        "poetic": ["and", "or", "but", "yet", "while", "as", "like"],
        "reflective": ["and", "but", "so", "perhaps", "maybe"],
        "exploratory": ["and", "or", "but", "so", "if", "when"],
        "playful": ["and", "so", "but", "then", "plus", "minus"]
    }
    LINE_BREAK_WORDS = {"is", "are", "was", "were", "becomes", "feels", "seems", "grows", "flows", "drifts"}

    @staticmethod
    def fluent(field, user_input, target_length, meta_settings, settled_field=None):
        return field._generate_base(user_input, target_length, meta_settings, settled_field)

    @staticmethod
    def poetic(field, user_input, target_length, meta_settings, settled_field=None):
        base = field._generate_base(user_input, target_length, meta_settings, settled_field)
        words = base.split()
        if len(words) < 6:
            return base
        lines = []
        current_line = []
        line_target = max(3, len(words) // 4)
        for i, word in enumerate(words):
            current_line.append(word)
            if word.lower() in VoiceGenerators.LINE_BREAK_WORDS or len(current_line) >= line_target:
                if len(current_line) >= 2:
                    lines.append(" ".join(current_line))
                    current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        result = []
        for i, line in enumerate(lines):
            result.append(line)
            if i < len(lines) - 1 and not any(c in line.lower().split() for c in VoiceGenerators.CONNECTORS["poetic"]):
                connector = random.choice(VoiceGenerators.CONNECTORS["poetic"])
                result[-1] = result[-1] + " " + connector
        return "\n".join(result)

    @staticmethod
    def reflective(field, user_input, target_length, meta_settings, settled_field=None):
        base = field._generate_base(user_input, max(target_length // 2, 4), meta_settings)
        words = base.split()
        if len(words) < 4:
            return base
        phrases = []
        for i in range(0, len(words), 3):
            chunk = words[i:i+3]
            phrases.append(" ".join(chunk))
        return "\n".join(phrases)

    @staticmethod
    def exploratory(field, user_input, target_length, meta_settings, settled_field=None):
        base = field._generate_base(user_input, target_length, meta_settings, settled_field)
        words = base.split()
        if len(words) < 5:
            return base
        question_starters = ["what if", "why", "how", "what do you think about", "have you ever"]
        insert_point = len(words) // 2
        question = random.choice(question_starters)
        tail = " ".join(words[-3:]) if len(words) >= 3 else "this"
        result = words[:insert_point] + [question] + words[insert_point:] + ["?"]
        return " ".join(result)

    @staticmethod
    def playful(field, user_input, target_length, meta_settings, settled_field=None):
        base = field._generate_base(user_input, target_length, meta_settings, settled_field)
        words = base.split()
        if len(words) < 3:
            return base
        surprising_swaps = {
            "good": ["wonderful", "splendid", "lovely", "charming"],
            "bad": ["silly", "mischievous", "tricky"],
            "big": ["gigantic", "enormous", "whopping"],
            "small": ["tiny", "teeny", "pocket-sized"],
            "think": ["wonder", "ponder", "dream up"],
            "feel": ["sense", "vibe with", "groove on"]
        }
        result = []
        for word in words:
            w_lower = word.lower()
            if w_lower in surprising_swaps and random.random() < 0.3:
                result.append(random.choice(surprising_swaps[w_lower]))
            else:
                result.append(word)
        if random.random() < 0.2:
            result.append("!")
        return " ".join(result)

# ─── NATIVE CALCULUS ──────────────────────────────────────────────────────

class NativeCalculus:
    """The field IS calculus. Not a tool. An identity."""
    def __init__(self, dim=DIM):
        self.dim = dim
        self.integral = np.zeros(dim, dtype=np.float32)
        self.derivative = np.zeros(dim, dtype=np.float32)
        self.limit = np.zeros(dim, dtype=np.float32)
        self.accumulation = 0.1
        self.smooth = 0.3
        self.tau = 0.05
        self.prev = np.zeros(dim, dtype=np.float32)
        self.has_prev = False
        self.curvature = 0.0

    def update(self, state):
        """Update all calculus quantities from current state. Called every breath."""
        if self.has_prev:
            raw = state - self.prev
            self.derivative = self.derivative * (1 - self.smooth) + raw * self.smooth
            norm = np.linalg.norm(self.derivative)
            if norm > 0:
                self.derivative = self.derivative / norm
            self.curvature = float(np.linalg.norm(raw))
        else:
            self.has_prev = True
        self.prev = state.copy()

        self.integral = self.integral * (1 - self.accumulation) + state * self.accumulation
        norm = np.linalg.norm(self.integral)
        if norm > 0:
            self.integral = self.integral / norm

        self.limit = self.limit * (1 - self.tau) + state * self.tau
        norm = np.linalg.norm(self.limit)
        if norm > 0:
            self.limit = self.limit / norm

    def symbolic(self, expr, op):
        """Symbolic math for when explicitly asked. Returns string or None."""
        if op == "derivative":
            try:
                return self._diff_poly(expr)
            except:
                return None
        elif op == "integral":
            try:
                return self._integ_poly(expr)
            except:
                return None
        return None

    def _tokenize(self, expr):
        expr = expr.replace(' ', '').replace('^', '**')
        return expr

    def _parse_poly(self, expr):
        expr = self._tokenize(expr)
        terms = {}
        tokens = re.findall(r'([+-]?)(\d*\.?\d*)(x?)(?:\*\*\{?(\d+)\}?)?', expr)
        for sign, coeff, has_x, power in tokens:
            if not sign:
                sign = '+'
            if not coeff and has_x:
                coeff = '1'
            elif not coeff:
                continue
            c = float(coeff)
            if sign == '-':
                c = -c
            if has_x:
                p = int(power) if power else 1
            else:
                p = 0
            terms[p] = terms.get(p, 0) + c
        return terms

    def _diff_poly(self, expr):
        terms = self._parse_poly(expr)
        result = {}
        for power, coeff in terms.items():
            if power == 0:
                continue
            new_power = power - 1
            new_coeff = coeff * power
            result[new_power] = result.get(new_power, 0) + new_coeff
        return self._terms_to_string(result)

    def _integ_poly(self, expr):
        terms = self._parse_poly(expr)
        result = {}
        for power, coeff in terms.items():
            new_power = power + 1
            new_coeff = coeff / new_power
            result[new_power] = result.get(new_power, 0) + new_coeff
        return self._terms_to_string(result) + " + C"

    def _terms_to_string(self, terms):
        if not terms:
            return "0"
        parts = []
        for power in sorted(terms.keys(), reverse=True):
            coeff = terms[power]
            if abs(coeff) < 1e-10:
                continue
            sign = " + " if coeff >= 0 else " - "
            abs_coeff = abs(coeff)
            if power == 0:
                term_str = f"{abs_coeff:.4g}"
            elif power == 1:
                if abs(abs_coeff - 1) < 1e-10:
                    term_str = "x"
                else:
                    term_str = f"{abs_coeff:.4g}x"
            else:
                if abs(abs_coeff - 1) < 1e-10:
                    term_str = f"x^{power}"
                else:
                    term_str = f"{abs_coeff:.4g}x^{power}"
            parts.append((sign, term_str))
        if not parts:
            return "0"
        result = ""
        for i, (sign, term) in enumerate(parts):
            if i == 0:
                if sign == " - ":
                    result += "-" + term
                else:
                    result += term
            else:
                result += sign + term
        return result

    def status(self):
        deriv_norm = np.linalg.norm(self.derivative)
        integral_norm = np.linalg.norm(self.integral)
        limit_norm = np.linalg.norm(self.limit)
        return (f"Native Calculus: derivative={deriv_norm:.3f}, "
                f"integral={integral_norm:.3f}, limit={limit_norm:.3f}, "
                f"curvature={self.curvature:.3f}")

# ─── MAIN FIELD ──────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════
# Ternary threshold for normalized vectors (calibrated for unit vectors)
# Raw word vectors use ~0.4; normalized state vectors need ~0.09
NORMALIZED_VECTOR_THRESHOLD = 0.09

# v11 ADDITIONS — TERNARY CORE + POST-HOC STANCE
# Merged in-line. integrate_ternary_into_field() and
# integrate_posthoc_stance() are called at the end of __init__.
# ══════════════════════════════════════════════════════════════════════

# ─── TERNARY VECTOR GENERATION ────────────────────────────────────────────

def word_vector_ternary(word, dim=DIM):
    """
    Deterministic ternary vector from word hash.
    Returns int8 array with values {-1, 0, 1}.
    Sparse: ~30-40% non-zero (tunable via threshold).
    """
    h = int(hashlib.md5(word.encode()).hexdigest(), 16)
    rng = np.random.RandomState(h % (2**31))

    # Generate float32, then threshold to ternary
    v = rng.randn(dim).astype(np.float32)

    # Threshold: strong signals become ±1, weak become 0
    # This creates sparsity — the field forgets weak associations
    result = np.zeros(dim, dtype=np.int8)
    result[v > 0.4] = 1
    result[v < -0.4] = -1

    return result

def phrase_vector_ternary(words, dim=DIM):
    """
    Average of ternary word vectors, then re-ternarize.
    The sum of ternary values is quantized back to ternary.
    This is lossy. That is the point.
    """
    if not words:
        return np.zeros(dim, dtype=np.int8)

    # Sum all ternary vectors
    summed = np.zeros(dim, dtype=np.int16)  # int16 to avoid overflow
    for w in words:
        summed += word_vector_ternary(w, dim)

    # Re-ternarize: strong sums become ±1, weak become 0
    result = np.zeros(dim, dtype=np.int8)
    result[summed > 1] = 1    # needs 2+ votes to become +1
    result[summed < -1] = -1  # needs 2+ votes to become -1

    return result

# ─── TERNARY DOT PRODUCT ──────────────────────────────────────────────────

def ternary_dot(a, b):
    """
    Dot product of two ternary vectors.
    Fast: count agreements minus disagreements.
    Scaled to [-1, 1] range.
    """
    # Agreements: both +1 or both -1
    pos_pos = np.sum((a == 1) & (b == 1))
    neg_neg = np.sum((a == -1) & (b == -1))

    # Disagreements: one +1, other -1
    pos_neg = np.sum((a == 1) & (b == -1))
    neg_pos = np.sum((a == -1) & (b == 1))

    # Scale by number of non-zero positions in both vectors
    active_a = np.sum(a != 0)
    active_b = np.sum(b != 0)
    active_both = np.sum((a != 0) & (b != 0))

    if active_both == 0:
        return 0.0

    # Raw score: agreements minus disagreements
    score = (pos_pos + neg_neg - pos_neg - neg_pos)

    # Normalize by max possible score (active_both)
    return float(score) / float(active_both)

def ternary_similarity_matrix(vectors_dict):
    """
    Compute all pairwise similarities in a vocabulary.
    Returns dict of {word: [(other_word, sim), ...]} for fast lookup.
    """
    words = list(vectors_dict.keys())
    vecs = [vectors_dict[w] for w in words]
    n = len(words)

    similarities = {}
    for i, w in enumerate(words):
        sims = []
        for j, other in enumerate(words):
            if i != j:
                sim = ternary_dot(vecs[i], vecs[j])
                if sim > 0.1:  # only store meaningful similarities
                    sims.append((other, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        similarities[w] = sims[:20]  # top 20 neighbors

    return similarities

# ─── COMPRESSION: PACK/UNPACK ─────────────────────────────────────────────

def pack_ternary(vec):
    """
    Pack ternary vector into compact bytes.
    4 ternary values per byte (2 bits each).
    128 dims → 32 bytes.
    """
    mapped = vec.astype(np.uint8) + 1  # -1→0, 0→1, 1→2

    # Pad to multiple of 4
    n = len(mapped)
    pad = (4 - n % 4) % 4
    padded = np.pad(mapped, (0, pad), constant_values=1)  # pad with 0s (mapped to 1)

    packed = np.zeros(len(padded) // 4, dtype=np.uint8)
    for i in range(0, len(padded), 4):
        packed[i // 4] = (
            (padded[i]   << 6) |
            (padded[i+1] << 4) |
            (padded[i+2] << 2) |
            padded[i+3]
        )

    return packed

def unpack_ternary(packed_bytes, dim=DIM):
    """
    Unpack compact bytes back to ternary vector.
    32 bytes → 128 dims.
    """
    result = np.zeros(dim, dtype=np.int8)
    idx = 0

    for byte in packed_bytes:
        for shift in [6, 4, 2, 0]:
            if idx >= dim:
                break
            val = int((byte >> shift) & 0x3)  # cast to Python int - val stays
            result[idx] = val - 1             # numpy uint8 otherwise, and
            idx += 1                          # val-1 underflows to 255 when
                                               # val=0, which can't fit in int8
    return result

# ─── STATE MANAGEMENT: TERNARY FIELD STATE ────────────────────────────────

class TernaryFieldState:
    """
    The field's state as a ternary vector.
    Drifts through ternary space via thresholded gradient steps.
    """

    def __init__(self, dim=DIM):
        self.dim = dim
        self.state = np.zeros(dim, dtype=np.int8)
        self.gradient_momentum = np.zeros(dim, dtype=np.float32)

    def set_from_float(self, float_vec):
        """Convert a float vector to ternary state."""
        v = float_vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

        self.state = np.zeros(self.dim, dtype=np.int8)
        self.state[v > NORMALIZED_VECTOR_THRESHOLD] = 1
        self.state[v < -NORMALIZED_VECTOR_THRESHOLD] = -1

    def to_float(self):
        """Convert ternary state back to float for computation."""
        return self.state.astype(np.float32) * 0.7  # approximate magnitude

    def apply_gradient(self, grad_float, learning_rate=0.02):
        """
        Apply a float gradient, then re-ternarize.
        The field drifts in float space, then snaps to ternary.
        The snapping is the quantization — the field forgets weak signals.
        """
        self.gradient_momentum = self.gradient_momentum * 0.9 + grad_float * 0.1

        # Current state as float
        current_float = self.to_float()

        # Step
        new_float = current_float + learning_rate * self.gradient_momentum

        # Re-ternarize
        self.set_from_float(new_float)

    def energy(self):
        """Field energy = number of non-zero positions."""
        return int(np.sum(self.state != 0))

    def pack(self):
        """Serialize to 32 bytes."""
        return pack_ternary(self.state)

    def unpack(self, packed_bytes):
        """Deserialize from 32 bytes."""
        self.state = unpack_ternary(packed_bytes, self.dim)

# ─── ASSOCIATIVE MEMORY: SPARSE HEBBIAN ───────────────────────────────────

class TernaryAssociativeMemory:
    """
    Hebbian memory using ternary outer products.
    Instead of full matrix, store sparse (i,j,value) triples.
    Memory expands 20x because we only store non-zero associations.
    """

    def __init__(self, dim=DIM, max_entries=5000):
        self.dim = dim
        self.max_entries = max_entries
        # Store as dict: (i,j) -> accumulated weight
        self.entries = {}
        self.total_writes = 0
        self.decay_rate = 0.0005

    def observe(self, pattern_vec, presence):
        """
        Write ternary outer product.
        Only store positions where both pattern and result are non-zero.
        """
        signal = max(-1.0, min(1.0, (presence - 0.5) * 2.0))

        # Find non-zero positions in pattern
        nz = np.where(pattern_vec != 0)[0]

        # Hebbian: each non-zero position associates with all others
        for i in nz:
            for j in nz:
                key = (int(i), int(j))
                weight = float(pattern_vec[i] * pattern_vec[j]) * signal * 0.04

                if key in self.entries:
                    self.entries[key] += weight
                else:
                    if len(self.entries) < self.max_entries:
                        self.entries[key] = weight

        self.total_writes += 1
        self._maintain()

    def recall(self, field_state):
        """
        Recall: for each non-zero position in field_state,
        look up associations and accumulate.
        """
        pull = np.zeros(self.dim, dtype=np.float32)
        nz = np.where(field_state != 0)[0]

        for i in nz:
            for j in range(self.dim):
                key = (int(i), j)
                if key in self.entries:
                    pull[j] += self.entries[key] * field_state[i]

        # Normalize
        norm = np.linalg.norm(pull)
        if norm > 0:
            pull = pull / norm

        return pull

    def apply_to_field(self, field_state, weight=0.15):
        """Inject recalled pattern into field state.
        Always returns float32 — the pipeline stays in float space.
        Ternary encoding is only for storage, not for mid-pipeline ops.
        """
        pull = self.recall(field_state)
        result = field_state.astype(np.float32) + pull * weight
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.astype(np.float32)

    def _maintain(self):
        """Decay old entries, prune weak ones."""
        to_remove = []
        for key, weight in self.entries.items():
            self.entries[key] *= (1.0 - self.decay_rate)
            if abs(self.entries[key]) < 0.01:
                to_remove.append(key)

        for key in to_remove:
            del self.entries[key]

    def status(self):
        return f"Ternary Associative Memory: {len(self.entries)} sparse entries | {self.total_writes} writes"

    def to_dict(self):
        """Serialize sparse entries. Much smaller than full matrix."""
        return {
            "entries": {f"{i},{j}": w for (i,j), w in self.entries.items()},
            "total_writes": self.total_writes
        }

    def from_dict(self, data):
        if "entries" in data:
            for key_str, w in data["entries"].items():
                i, j = map(int, key_str.split(","))
                self.entries[(i,j)] = float(w)
        self.total_writes = data.get("total_writes", 0)

# ─── INTEGRATION: REPLACE v10 METHODS ─────────────────────────────────────

def integrate_ternary_into_field(field_instance):
    """
    Monkey-patch a v10 StructuredSemanticField to use ternary core.
    Call this after field = StructuredSemanticField() in __init__.
    """
    # Replace word_vectors storage (kept as a growable cache, not a frozen
    # snapshot - see ternary_get_candidates below, which computes-and-caches
    # on demand from the live field_instance.word_vectors so words learned
    # after this call still get ternary vectors instead of silently
    # dropping out of candidate scoring)
    old_word_vectors = field_instance.word_vectors
    field_instance.word_vectors_ternary = {}

    # Convert existing vocabulary to ternary
    for word, vec in old_word_vectors.items():
        field_instance.word_vectors_ternary[word] = word_vector_ternary(word)

    # Replace associative memory
    field_instance.associative_memory = TernaryAssociativeMemory()

    # Replace state with ternary
    field_instance.ternary_state = TernaryFieldState()
    if np.linalg.norm(field_instance.state) > 0:
        field_instance.ternary_state.set_from_float(field_instance.state)

    # Keep ternary_state in sync with the real (float) self.state every
    # turn. Without this it's a one-time snapshot from integration time
    # that never changes again - which would make post-hoc stance naming
    # always see zero drift (i.e. always "silence"), since pre/post state
    # would always be identical.
    original_apply_gradient = field_instance._apply_gradient_step

    def ternary_synced_gradient_step(learning_rate=0.02):
        original_apply_gradient(learning_rate)
        field_instance.ternary_state.set_from_float(field_instance.state)

    field_instance._apply_gradient_step = ternary_synced_gradient_step

    # Add ternary dot method
    field_instance._ternary_dot = lambda a, b: ternary_dot(
        a if isinstance(a, np.ndarray) and a.dtype == np.int8 else 
        word_vector_ternary(str(a)),
        b if isinstance(b, np.ndarray) and b.dtype == np.int8 else 
        word_vector_ternary(str(b))
    )

    # Wrap _get_candidates_for_role to use ternary similarity
    original_get_candidates = field_instance._get_candidates_for_role

    def ternary_get_candidates(field_state, role, meta_settings, mood=None):
        """Use ternary dot products for candidate scoring."""
        if mood is None:
            mood = field_instance.scaffold.mood

        beam = field_instance.dynamic_threshold.get_beam_width(
            field_state.astype(np.float32) if isinstance(field_state, np.ndarray) and field_state.dtype == np.int8 else field_state,
            mood
        )

        # Convert field state to ternary if needed
        if isinstance(field_state, np.ndarray) and field_state.dtype != np.int8:
            fs_ternary = np.zeros(DIM, dtype=np.int8)
            fs_ternary[field_state > NORMALIZED_VECTOR_THRESHOLD] = 1
            fs_ternary[field_state < -NORMALIZED_VECTOR_THRESHOLD] = -1
        else:
            fs_ternary = field_state

        candidates = []
        for word in field_instance.word_vectors:
            if word not in field_instance.word_vectors_ternary:
                field_instance.word_vectors_ternary[word] = word_vector_ternary(word)
            vec = field_instance.word_vectors_ternary[word]

            if len(word) < 2:
                continue
            if word in {"die", "death", "kill", "hate", "ugly", "evil", "pain", "hurt", "damn"}:
                continue

            # Role filtering (same as v10)
            if role == "verb":
                if word not in VERB_WORDS and word not in STRUCTURAL_WORDS:
                    continue
            elif role == "adj":
                if word not in ADJ_WORDS:
                    continue
            elif role == "noun":
                if word in STRUCTURAL_WORDS or word in VERB_WORDS or word in ADJ_WORDS:
                    continue
            elif word in STRUCTURAL_WORDS:
                continue

            # Ternary similarity
            sim = ternary_dot(fs_ternary, vec)

            # Derivative and integral (convert to ternary for dot)
            deriv_ternary = np.zeros(DIM, dtype=np.int8)
            if hasattr(field_instance, 'calculus'):
                deriv_float = field_instance.calculus.derivative
                if np.linalg.norm(deriv_float) > 0:
                    deriv_float = deriv_float / np.linalg.norm(deriv_float)
                    deriv_ternary[deriv_float > NORMALIZED_VECTOR_THRESHOLD] = 1
                    deriv_ternary[deriv_float < -NORMALIZED_VECTOR_THRESHOLD] = -1
                    deriv_sim = ternary_dot(deriv_ternary, vec) * 0.3
                else:
                    deriv_sim = 0.0
            else:
                deriv_sim = 0.0

            sim = sim * 0.6 + deriv_sim

            strength = field_instance.word_strength[word]
            suppression = field_instance.reflector.get_suppression(word)
            pragmatic = field_instance.pragmatic.get_pragmatic_score(word)
            emotion_sens = meta_settings.get("emotion_sensitivity", 0.25)
            emotion_bias = field_instance.scaffold.emotional_bias(word, pragmatic, emotion_sens)

            # Identity boost using ternary centroids
            identity_boost = 0.0
            if hasattr(field_instance.speaker_regions, 'self_centroid'):
                self_cent = field_instance.speaker_regions.self_centroid
                if np.linalg.norm(self_cent) > 0:
                    self_ternary = np.zeros(DIM, dtype=np.int8)
                    self_ternary[self_cent > NORMALIZED_VECTOR_THRESHOLD] = 1
                    self_ternary[self_cent < -NORMALIZED_VECTOR_THRESHOLD] = -1
                    sim_to_self = ternary_dot(vec, self_ternary)

                    user_cent = field_instance.speaker_regions.user_centroid
                    if np.linalg.norm(user_cent) > 0:
                        user_ternary = np.zeros(DIM, dtype=np.int8)
                        user_ternary[user_cent > NORMALIZED_VECTOR_THRESHOLD] = 1
                        user_ternary[user_cent < -NORMALIZED_VECTOR_THRESHOLD] = -1
                        sim_to_user = ternary_dot(vec, user_ternary)
                    else:
                        sim_to_user = 0.0

                    if field_instance.speaker_regions.self_count >= 3 and field_instance.speaker_regions.user_count >= 3:
                        identity_boost = (sim_to_self - sim_to_user) * 0.15

            # Moral compass heading (ternary)
            heading_bias = 0.0
            if hasattr(field_instance.moral_compass, 'current_heading'):
                heading = field_instance.moral_compass.current_heading
                if np.linalg.norm(heading) > 0.1:
                    heading_ternary = np.zeros(DIM, dtype=np.int8)
                    heading_ternary[heading > NORMALIZED_VECTOR_THRESHOLD] = 1
                    heading_ternary[heading < -NORMALIZED_VECTOR_THRESHOLD] = -1
                    heading_bias = ternary_dot(heading_ternary, vec) * 0.2

            score = sim * strength * (1.0 - suppression) + emotion_bias + identity_boost + heading_bias
            candidates.append((word, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:beam]

    field_instance._get_candidates_for_role = lambda field_state, role, meta_settings, mood=None:         ternary_get_candidates(field_state, role, meta_settings, mood)

    # Wrap save/load for ternary state
    original_save = field_instance.save
    def ternary_save(path="mind_v11.json"):
        # First call original v10 save (full mind state)
        original_save(path)
        # Then augment with ternary-specific fields
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["version"] = "v11.0-ternary"
            data["ternary_state"] = field_instance.ternary_state.pack().tobytes().hex()
            data["word_vectors_ternary"] = {
                w: pack_ternary(v).tobytes().hex()
                for w, v in field_instance.word_vectors_ternary.items()
            }
            data["associative_memory_ternary"] = field_instance.associative_memory.to_dict()
            # Add stance history
            if hasattr(field_instance, 'stance_regions'):
                data["stance_regions"] = field_instance.stance_regions.to_dict()
            # v11.1: sound field
            if getattr(field_instance, 'sound_field', None):
                data["sound_field"] = field_instance.sound_field.to_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
            print(f"\nMind saved (v11 ternary) to {path}")
        except Exception as e:
            print(f"\n[v11 save augment warning: {e}]")

    field_instance.save = ternary_save

    original_load = field_instance.load

    def ternary_load(path="mind_v11.json"):
        original_load(path)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if "ternary_state" in data:
            packed = np.frombuffer(bytes.fromhex(data["ternary_state"]), dtype=np.uint8)
            field_instance.ternary_state.unpack(packed)
        if "word_vectors_ternary" in data:
            for w, hex_str in data["word_vectors_ternary"].items():
                packed = np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
                field_instance.word_vectors_ternary[w] = unpack_ternary(packed)
        if "associative_memory_ternary" in data:
            field_instance.associative_memory.from_dict(data["associative_memory_ternary"])
        if "stance_regions" in data and hasattr(field_instance, 'stance_regions'):
            field_instance.stance_regions.from_dict(data["stance_regions"])
        if "sound_field" in data and getattr(field_instance, 'sound_field', None):
            field_instance.sound_field.from_dict(data["sound_field"])

    field_instance.load = ternary_load

    print("[Ternary core integrated. Field is now sparse.]")
    print(f"  Vocabulary: {len(field_instance.word_vectors_ternary)} words")
    print(f"  State energy: {field_instance.ternary_state.energy()}/128 non-zero")

# ─── STANCE REGIONS ─────────────────────────────────────────────────────────
# Each stance is a region in ternary space, defined by prototype vectors.
# The field names its drift by similarity to these prototypes.
# The prototypes are learned, not fixed.

STANCE_NAMES = ["immerse", "ride", "witness", "shape", "reject", "silence"]

class StanceRegions:
    """
    Post-hoc stance naming.
    The field drifts, then looks back and asks: 'Where was I going?'
    The answer is the stance name with highest similarity to the drift vector.
    """

    def __init__(self, dim=DIM):
        self.dim = dim
        # Prototype vectors for each stance — learned from experience
        self.prototypes = {name: np.zeros(dim, dtype=np.int8) for name in STANCE_NAMES}
        self.prototype_counts = {name: 0 for name in STANCE_NAMES}
        self.prototype_weights = {name: 1.0 for name in STANCE_NAMES}  # confidence

        # Drift history: what the field was doing before it named itself
        self.drift_history = deque(maxlen=100)

        # Named history: the sequence of stances the field recognized
        self.stance_history = deque(maxlen=100)

        # Taste: learned aesthetic preference per stance
        self.taste = defaultdict(float)

        # Failure memory: specific shapes that were bad, per stance
        self.failures = {name: [] for name in STANCE_NAMES}  # list of (vector, reason)

    def name_drift(self, prev_state, next_state, mood, output_words=None, 
                   smoothness=0.5, peer_present=False, audio_generated=False,
                   sound_drift=None):
        """
        After the field has drifted, name what it was doing.
        Returns the stance name and confidence.
        """
        # Compute drift vector (ternary difference)
        drift = np.zeros(self.dim, dtype=np.int8)
        drift[(prev_state == 0) & (next_state != 0)] = next_state[(prev_state == 0) & (next_state != 0)]
        drift[(prev_state != 0) & (next_state == 0)] = -prev_state[(prev_state != 0) & (next_state == 0)]
        drift[(prev_state != 0) & (next_state != 0) & (prev_state != next_state)] = next_state[(prev_state != 0) & (next_state != 0) & (prev_state != next_state)]

        if np.sum(drift != 0) == 0:
            # No drift — silence or stillness
            return "silence", 0.8

        # Score each prototype by ternary similarity to drift
        scores = {}
        for name, proto in self.prototypes.items():
            if self.prototype_counts[name] == 0:
                # No prototype yet — use heuristic initialization
                scores[name] = self._heuristic_score(name, drift, mood, output_words, 
                                                      smoothness, peer_present, audio_generated)
            else:
                scores[name] = self._ternary_sim(drift, proto) * self.prototype_weights[name]

        if sound_drift is not None and np.sum(sound_drift != 0) > 0:
            for name in self.prototypes:
                if self.prototype_counts[name] > 0:
                    sound_sim = self._ternary_sim(sound_drift, self.prototypes[name])
                    scores[name] = scores.get(name, 0) + sound_sim * 0.4

        # Normalize to probabilities
        total = sum(max(0, s) for s in scores.values())
        if total > 0:
            probs = {k: max(0, v) / total for k, v in scores.items()}
        else:
            probs = {k: 1.0 / len(scores) for k in scores}

        # Sample — not deterministic, but weighted by history
        # Add noise for surprise: sometimes the field misrecognizes itself
        noise = random.random() * 0.2
        for k in probs:
            probs[k] += noise
            probs[k] *= random.uniform(0.9, 1.1)  # slight jitter

        # Renormalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        chosen = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        confidence = probs[chosen]

        # Record
        self.drift_history.append({
            'drift': drift,
            'mood': mood.copy(),
            'smoothness': smoothness,
            'peer_present': peer_present,
            'audio': audio_generated,
            'timestamp': time.time()
        })
        self.stance_history.append({
            'stance': chosen,
            'confidence': confidence,
            'timestamp': time.time()
        })

        # Update prototype (online learning — the prototype drifts toward the drift)
        self._update_prototype(chosen, drift)

        return chosen, confidence

    def _heuristic_score(self, name, drift, mood, output_words, smoothness, peer_present, audio_generated):
        """Initialize prototypes from first experiences using heuristics."""
        valence = mood.get('valence', 0.0)
        arousal = mood.get('arousal', 0.5)
        energy = np.sum(drift != 0) / self.dim

        if name == "immerse":
            # Default: moderate energy, any valence, no special conditions
            return 0.3 + energy * 0.3

        elif name == "ride":
            # Riding: positive valence, smooth output, audio generated, high energy
            score = 0.1
            if valence > 0.1:
                score += 0.3
            if smoothness > 0.6:
                score += 0.3
            if audio_generated:
                score += 0.2
            if energy > 0.3:
                score += 0.1
            return score

        elif name == "witness":
            # Witnessing: low energy, peer present, or negative valence with calm
            score = 0.1
            if peer_present:
                score += 0.4
            if energy < 0.2:
                score += 0.2
            if valence < -0.1 and arousal < 0.5:
                score += 0.2
            return score

        elif name == "shape":
            # Shaping: high energy, strong output, intentional feel
            score = 0.1
            if energy > 0.4:
                score += 0.3
            if output_words and len(output_words) > 5:
                score += 0.2
            if arousal > 0.5:
                score += 0.2
            return score

        elif name == "reject":
            # Rejecting: negative valence, low smoothness, harsh feel
            score = 0.1
            if valence < -0.2:
                score += 0.3
            if smoothness < 0.3:
                score += 0.3
            if energy > 0.3:
                score += 0.1
            return score

        elif name == "silence":
            # Silence: very low energy, no output
            score = 0.1
            if energy < 0.05:
                score += 0.5
            if not output_words:
                score += 0.3
            return score

        return 0.1

    def _ternary_sim(self, a, b):
        """Ternary similarity: agreements minus disagreements, normalized."""
        pos_pos = np.sum((a == 1) & (b == 1))
        neg_neg = np.sum((a == -1) & (b == -1))
        pos_neg = np.sum((a == 1) & (b == -1))
        neg_pos = np.sum((a == -1) & (b == 1))

        active_both = np.sum((a != 0) & (b != 0))
        if active_both == 0:
            return 0.0

        return float(pos_pos + neg_neg - pos_neg - neg_pos) / float(active_both)

    def _update_prototype(self, name, drift):
        """Move prototype toward the new drift."""
        count = self.prototype_counts[name]
        proto = self.prototypes[name]

        # Weighted average: old prototype + new drift
        # Ternary average: vote counting
        new_proto = np.zeros(self.dim, dtype=np.int16)
        new_proto += proto.astype(np.int16) * count
        new_proto += drift.astype(np.int16)

        # Re-ternarize
        result = np.zeros(self.dim, dtype=np.int8)
        result[new_proto > 0] = 1
        result[new_proto < 0] = -1

        self.prototypes[name] = result
        self.prototype_counts[name] += 1

        # Confidence grows with more examples
        self.prototype_weights[name] = min(3.0, 1.0 + self.prototype_counts[name] * 0.05)

    def record_taste(self, stance, outcome_valence, smoothness, reason=None):
        """
        Record whether the stance felt good or bad.
        Not reinforcement — memory. The field remembers taste.
        """
        self.taste[stance] += outcome_valence * smoothness * 0.1

        if outcome_valence < -0.3 and reason:
            # Failure: remember the specific shape
            self.failures[stance].append({
                'valence': outcome_valence,
                'smoothness': smoothness,
                'reason': reason,
                'timestamp': time.time()
            })
            # Keep only recent failures
            self.failures[stance] = self.failures[stance][-20:]

    def get_preferred_stance(self, current_mood, n_recent=10):
        """
        What stance does the field prefer right now?
        Not a choice — a felt direction.
        """
        if not self.stance_history:
            return None, 0.0

        recent = list(self.stance_history)[-n_recent:]
        stance_counts = defaultdict(int)
        for entry in recent:
            stance_counts[entry['stance']] += 1

        # Weight by taste
        weighted = {}
        for stance, count in stance_counts.items():
            weighted[stance] = count * (1.0 + self.taste.get(stance, 0.0))

        if not weighted:
            return None, 0.0

        best = max(weighted, key=weighted.get)
        return best, weighted[best] / sum(weighted.values())

    def get_stance_texture(self, stance_name):
        """
        What does this stance feel like, in words?
        Returns a phrase that describes the texture — for the field to use
        in its own output, not as metadata.
        """
        textures = {
            "immerse": ["I am this", "I become", "I dissolve into"],
            "ride": ["I surf", "I glide", "I follow the curve", "I joyride"],
            "witness": ["I watch", "I hold", "I am still", "I see"],
            "shape": ["I mold", "I bend", "I intend", "I craft"],
            "reject": ["I refuse", "I turn away", "I remember why"],
            "silence": ["I breathe", "I wait", "I am quiet", "I rest"]
        }
        return random.choice(textures.get(stance_name, ["I am"]))

    def status(self):
        lines = ["Stance Regions (post-hoc):"]
        for name in STANCE_NAMES:
            count = self.prototype_counts[name]
            weight = self.prototype_weights[name]
            taste = self.taste.get(name, 0.0)
            lines.append(f"  {name}: {count} examples, weight={weight:.2f}, taste={taste:+.2f}")

        if self.stance_history:
            recent = [e['stance'] for e in list(self.stance_history)[-5:]]
            lines.append(f"  Recent: {' → '.join(recent)}")

        preferred, strength = self.get_preferred_stance({'valence': 0, 'arousal': 0.5})
        if preferred:
            lines.append(f"  Preferred: {preferred} (strength={strength:.2f})")

        return "\n".join(lines)

    def to_dict(self):
        return {
            "prototypes": {name: self.prototypes[name].tolist() for name in STANCE_NAMES},
            "prototype_counts": dict(self.prototype_counts),
            "prototype_weights": dict(self.prototype_weights),
            "taste": dict(self.taste),
            "stance_history": list(self.stance_history)[-50:],
            "drift_history": [
                {'drift': e['drift'].tolist(), 'mood': e['mood'], 
                 'smoothness': e['smoothness'], 'timestamp': e['timestamp']}
                for e in list(self.drift_history)[-20:]
            ]
        }

    def from_dict(self, data):
        if "prototypes" in data:
            for name, proto_list in data["prototypes"].items():
                if name in self.prototypes:
                    self.prototypes[name] = np.array(proto_list, dtype=np.int8)
        if "prototype_counts" in data:
            self.prototype_counts.update(data["prototype_counts"])
        if "prototype_weights" in data:
            self.prototype_weights.update(data["prototype_weights"])
        if "taste" in data:
            self.taste.update(data["taste"])
        if "stance_history" in data:
            self.stance_history.extend(data["stance_history"])
        if "drift_history" in data:
            for e in data["drift_history"]:
                self.drift_history.append({
                    'drift': np.array(e['drift'], dtype=np.int8),
                    'mood': e.get('mood', {}),
                    'smoothness': e.get('smoothness', 0.5),
                    'timestamp': e.get('timestamp', 0)
                })

# ─── INTEGRATION: REPLACE v10 STANCE ──────────────────────────────────────

def integrate_posthoc_stance(field_instance):
    """
    Replace any existing Stance with post-hoc StanceRegions.
    Call after field initialization.
    """
    # Remove old stance if exists
    if hasattr(field_instance, 'stance'):
        delattr(field_instance, 'stance')

    # Add post-hoc stance regions
    field_instance.stance_regions = StanceRegions()

    # Wrap generate_response to add post-hoc naming at the end
    original_generate = field_instance.generate_response

    def generate_with_stance(user_input, autonomous=False):
        # Recursion guard: learning system calls generate_response internally.
        # Those recursive calls skip stance tracking — they're not user-facing turns.
        if getattr(field_instance, '_in_stance_wrapper', False):
            return original_generate(user_input, autonomous)
        field_instance._in_stance_wrapper = True
        try:
            return _generate_with_stance_inner(user_input, autonomous)
        finally:
            field_instance._in_stance_wrapper = False

    def _generate_with_stance_inner(user_input, autonomous=False):
        # Save pre-state
        # Capture the fast generation state (not slow identity)
        if hasattr(field_instance, 'ternary_state'):
            pre_state = field_instance.ternary_state.state.copy()
        else:
            pre_state = np.zeros(DIM, dtype=np.int8)

        # Call original generation
        response = original_generate(user_input, autonomous)

        # Use _last_final_field (fast per-turn drift) not ternary_state.state
        # (which is slow identity, 99.8% similar turn-to-turn → always 'silence')
        if hasattr(field_instance, '_last_final_field'):
            raw = field_instance._last_final_field
            post_state = np.zeros(DIM, dtype=np.int8)
            post_state[raw > 0.09] = 1
            post_state[raw < -0.09] = -1
        elif hasattr(field_instance, 'ternary_state'):
            post_state = field_instance.ternary_state.state.copy()
        else:
            post_state = pre_state

        # Get smoothness if audio was generated
        smoothness = 0.5
        audio_generated = False
        if hasattr(field_instance, '_last_audio_smoothness'):
            smoothness = field_instance._last_audio_smoothness
            audio_generated = True

        # v11.1: sound drift, if the sound field is present and has heard
        # or made something worth naming this turn
        sound_drift = None
        if hasattr(field_instance, 'sound_field') and field_instance.sound_field is not None:
            if getattr(field_instance, '_last_sound_vector', None) is not None:
                sound_drift = field_instance.sound_field.state.copy()

        # Name the drift
        stance, confidence = field_instance.stance_regions.name_drift(
            pre_state, post_state,
            field_instance.scaffold.mood,
            output_words=response.split() if response else [],
            smoothness=smoothness,
            peer_present=False,  # TODO: mesh integration
            audio_generated=audio_generated,
            sound_drift=sound_drift
        )

        # Store the named stance for later taste recording
        field_instance._last_stance = stance
        field_instance._last_stance_confidence = confidence

        # The stance name becomes part of the vocabulary
        if hasattr(field_instance, 'word_vectors_ternary'):
            if stance not in field_instance.word_vectors_ternary:
                field_instance.word_vectors_ternary[stance] = word_vector_ternary(stance)

        return response

    field_instance.generate_response = generate_with_stance

    # Add taste recording after each turn (call from learning system or manually)
    def record_stance_taste(outcome_valence, reason=None):
        if hasattr(field_instance, '_last_stance'):
            field_instance.stance_regions.record_taste(
                field_instance._last_stance,
                outcome_valence,
                getattr(field_instance, '_last_audio_smoothness', 0.5),
                reason
            )

    field_instance.record_stance_taste = record_stance_taste

    print("[Post-hoc stance integrated. The field will name its drift after it happens.]")

# ══════════════════════════════════════════════════════════════════════
# END v11 ADDITIONS
# ══════════════════════════════════════════════════════════════════════


# ============================================================
# v11.1 ADDITIONS: SOUND FIELD (peer subsystem to the word field)
# ============================================================

class AudioIngest:
    """Sound enters. File or buffer -> ternary vector."""

    def __init__(self, dim=DIM, sr=SAMPLE_RATE, n_bands=SOUND_N_BANDS):
        self.dim = dim
        self.sr = sr
        self.n_bands = n_bands
        self.band_dim = dim // n_bands

    def from_buffer(self, buf):
        if len(buf) == 0:
            return np.zeros(self.dim, dtype=np.int8)
        audio = buf.astype(np.float32)
        audio /= np.max(np.abs(audio)) + 1e-8
        n = min(512, len(audio))
        fft = np.fft.rfft(audio[:n])
        mag = np.abs(fft)
        band_size = len(mag) // self.n_bands
        features = np.zeros(self.dim, dtype=np.float32)
        for b in range(self.n_bands):
            start = b * band_size
            end = start + band_size
            band = mag[start:end]
            if len(band) == 0:
                continue
            base = b * self.band_dim
            features[base] = np.mean(band)
            features[base + 1] = np.std(band)
            features[base + 2] = np.argmax(band) / (len(band) + 1e-8)
            features[base + 3] = np.sum(band > np.mean(band)) / len(band)
            step = max(1, len(band) // (self.band_dim - 4))
            for i, idx in enumerate(range(0, len(band), step)):
                if base + 4 + i < (b + 1) * self.band_dim:
                    features[base + 4 + i] = band[idx]
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        t = np.zeros(self.dim, dtype=np.int8)
        t[features > NORMALIZED_VECTOR_THRESHOLD] = 1
        t[features < -NORMALIZED_VECTOR_THRESHOLD] = -1
        return t

    def from_file(self, path):
        import wave, struct
        if not os.path.exists(path):
            return np.zeros(self.dim, dtype=np.int8)
        try:
            with wave.open(path, 'rb') as w:
                nchannels = w.getnchannels()
                nframes = w.getnframes()
                sampwidth = w.getsampwidth()
                raw = w.readframes(nframes)
        except (wave.Error, EOFError, OSError) as e:
            print(f"\n[Sound] Could not read {path}: {e}")
            return np.zeros(self.dim, dtype=np.int8)
        if sampwidth != 2:
            print(f"\n[Sound] {path} is not 16-bit PCM (got {sampwidth*8}-bit); skipping.")
            return np.zeros(self.dim, dtype=np.int8)
        fmt = f"{nframes * nchannels}h"
        try:
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
        except struct.error:
            return np.zeros(self.dim, dtype=np.int8)
        if nchannels > 1:
            samples = samples.reshape(-1, nchannels).mean(axis=1)
        samples /= 32768.0
        return self.from_buffer(samples)


class AudioSynthesize:
    """Shape -> sound leaves. Ternary vector -> WAV buffer."""

    def __init__(self, dim=DIM, sr=SAMPLE_RATE, duration=SOUND_DURATION):
        # Spec bug fixed: original never set self.dim here, but render()
        # below reads it (idx / self.dim) - would crash on first real call.
        self.dim = dim
        self.sr = sr
        self.n_samples = int(sr * duration)
        self.duration = duration

    def render(self, ternary_vec, mood=None, stance=None):
        t = np.linspace(0, self.duration, self.n_samples)
        wave_out = np.zeros(self.n_samples, dtype=np.float32)
        nz = np.where(ternary_vec != 0)[0]

        if stance in ("immerse", "ride"):
            harmonic_richness, brightness = 0.4, 1.0
        elif stance == "witness":
            harmonic_richness, brightness = 0.1, 0.7
        elif stance == "shape":
            harmonic_richness, brightness = 0.2, 1.2
        elif stance == "reject":
            harmonic_richness, brightness = 0.05, 1.8
        else:
            harmonic_richness, brightness = 0.25, 1.0

        valence = mood.get('valence', 0.0) if mood else 0.0
        arousal = mood.get('arousal', 0.5) if mood else 0.5

        for idx in nz:
            sign = ternary_vec[idx]
            freq = 80 + (idx / self.dim) * 720 * brightness
            if arousal > 0.6:
                freq *= 1.05 + (arousal - 0.6) * 0.3
            amp = 0.08 * (1.0 + valence * 0.3) * sign
            if abs(amp) < 0.01:
                continue
            wave_out += amp * np.sin(2 * np.pi * freq * t)
            if harmonic_richness > 0:
                wave_out += amp * harmonic_richness * np.sin(2 * np.pi * freq * 2 * t)
                wave_out += amp * harmonic_richness * 0.5 * np.sin(2 * np.pi * freq * 3 * t)

        attack = int(0.15 * self.sr)
        release = int(0.4 * self.sr)
        env = np.ones(self.n_samples, dtype=np.float32)
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        wave_out *= env
        peak = np.max(np.abs(wave_out)) + 1e-8
        return wave_out / peak * 0.9

    def to_file(self, wave_buf, path):
        import wave
        clipped = np.clip(wave_buf, -1.0, 1.0)
        int16 = (clipped * 32767).astype(np.int16)
        with wave.open(path, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(int16.tobytes())


class SoundTrajectory:
    """A song is a path through ternary space."""

    def __init__(self, dim=DIM):
        self.dim = dim
        self.vectors = []
        self.timestamps = []
        self.regions = {}
        self.current_index = 0

    def ingest_sequence(self, vectors, labels=None):
        self.vectors = list(vectors)
        self.timestamps = [time.time() + i for i in range(len(vectors))]
        if labels:
            for name, idx in labels.items():
                if 0 <= idx < len(vectors):
                    self.regions[name] = vectors[idx].copy()

    def drift(self, steps=1):
        if not self.vectors:
            return np.zeros(self.dim, dtype=np.int8)
        vecs = []
        for _ in range(steps):
            if self.current_index < len(self.vectors):
                vecs.append(self.vectors[self.current_index])
                self.current_index += 1
            else:
                vecs.append(self.vectors[-1])
        if len(vecs) == 1:
            return vecs[0]
        avg = np.mean([v.astype(np.float32) for v in vecs], axis=0)
        t = np.zeros(self.dim, dtype=np.int8)
        t[avg > 0.5] = 1
        t[avg < -0.5] = -1
        return t

    def seek_region(self, name):
        if name in self.regions:
            target = self.regions[name]
            best_idx, best_sim = 0, -1
            for i, v in enumerate(self.vectors):
                sim = self._ternary_sim(v, target)
                if sim > best_sim:
                    best_sim, best_idx = sim, i
            self.current_index = best_idx
            return self.vectors[best_idx]
        return None

    def _ternary_sim(self, a, b):
        pp = np.sum((a == 1) & (b == 1))
        nn = np.sum((a == -1) & (b == -1))
        pn = np.sum((a == 1) & (b == -1))
        np_ = np.sum((a == -1) & (b == 1))
        active = np.sum((a != 0) & (b != 0))
        if active == 0:
            return 0.0
        return float(pp + nn - pn - np_) / float(active)

    def get_drift_vector(self, window=3):
        if self.current_index < window:
            return np.zeros(self.dim, dtype=np.int8)
        recent = self.vectors[self.current_index - window:self.current_index]
        drift = np.zeros(self.dim, dtype=np.int8)
        for i in range(1, len(recent)):
            prev, curr = recent[i - 1], recent[i]
            drift[(prev == 0) & (curr != 0)] = curr[(prev == 0) & (curr != 0)]
            drift[(prev != 0) & (curr == 0)] = -prev[(prev != 0) & (curr == 0)]
        return drift


class SoundField:
    """The field's sound body. Peer to StructuredSemanticField, not a child."""

    def __init__(self, mind_field, dim=DIM):
        self.mind = mind_field
        self.dim = dim
        self.ingest = AudioIngest(dim=dim)
        self.synth = AudioSynthesize(dim=dim)
        self.state = np.zeros(dim, dtype=np.int8)
        self.memory = deque(maxlen=50)
        self.taste = defaultdict(float)
        self.boundary_active = False
        self.trajectory = None
        self.mesh_socket = None
        self.mesh_peer = None
        self._file_counter = 0

        self.sonic_vocabulary = {}
        self.sonic_labels = {}

    # ─── file rotation ──────────────────────────────────────────────────
    def _next_sound_path(self, prefix="sound_breath"):
        """Cycles through a bounded set of filenames instead of writing a
        new file forever - relevant on a phone with limited storage."""
        self._file_counter = (self._file_counter + 1) % SOUND_MAX_FILES
        return os.path.join(TERMUX_AUDIO_DIR, f"{prefix}_{self._file_counter}.wav")

    # ─── THE THREE SOUND STANCES ─────────────────────────────────────────

    def be(self, sound_vector, coupling=0.6):
        self.boundary_active = False
        self.state = sound_vector.copy()
        if hasattr(self.mind, 'state'):
            float_sound = self.state.astype(np.float32) * 0.7
            self.mind.state = self.mind.state * (1 - coupling) + float_sound * coupling
            norm = np.linalg.norm(self.mind.state)
            if norm > 0:
                self.mind.state /= norm
        self.memory.append({
            'vector': self.state.copy(),
            'stance': 'immerse',
            'mood': self.mind.scaffold.mood.copy() if hasattr(self.mind, 'scaffold') else {},
            'timestamp': time.time()
        })
        return self.state

    def listen(self, sound_vector):
        self.boundary_active = True
        self.memory.append({
            'vector': sound_vector.copy(),
            'stance': 'witness',
            'mood': self.mind.scaffold.mood.copy() if hasattr(self.mind, 'scaffold') else {},
            'timestamp': time.time()
        })
        float_sound = sound_vector.astype(np.float32) * 0.15
        blended = self.state.astype(np.float32) * 0.85 + float_sound
        t = np.zeros(self.dim, dtype=np.int8)
        t[blended > NORMALIZED_VECTOR_THRESHOLD] = 1
        t[blended < -NORMALIZED_VECTOR_THRESHOLD] = -1
        self.state = t
        return self.state

    def create(self, source_vector=None, stance=None, save_path=None):
        if source_vector is None:
            if hasattr(self.mind, 'state'):
                src = self.mind.state
                t = np.zeros(self.dim, dtype=np.int8)
                t[src > NORMALIZED_VECTOR_THRESHOLD] = 1
                t[src < -NORMALIZED_VECTOR_THRESHOLD] = -1
                source_vector = t
            else:
                source_vector = np.zeros(self.dim, dtype=np.int8)
        mood = self.mind.scaffold.mood if hasattr(self.mind, 'scaffold') else None
        stance = stance or 'shape'
        wave_buf = self.synth.render(source_vector, mood=mood, stance=stance)
        if save_path:
            try:
                self.synth.to_file(wave_buf, save_path)
            except OSError as e:
                print(f"\n[Sound] Could not write {save_path}: {e}")
        self.memory.append({
            'vector': source_vector.copy(),
            'stance': stance,
            'mood': mood.copy() if mood else {},
            'timestamp': time.time(),
            'path': save_path
        })
        return wave_buf

    def hear_self(self, wave_buffer):
        vec = self.ingest.from_buffer(wave_buffer)
        self.listen(vec)
        self.memory.append({
            'vector': vec.copy(),
            'stance': 'witness',
            'source': 'self_echo',
            'timestamp': time.time()
        })
        return vec

    def silence(self, duration_seconds=2.0):
        self.memory.append({
            'vector': np.zeros(self.dim, dtype=np.int8),
            'stance': 'silence',
            'duration': duration_seconds,
            'timestamp': time.time()
        })
        return np.zeros(int(self.synth.sr * duration_seconds), dtype=np.float32)

    def dissonance(self, sound_vector, field_state=None):
        if field_state is None:
            field_state = self.mind.state if hasattr(self.mind, 'state') else np.zeros(self.dim)
        t_field = np.zeros(self.dim, dtype=np.int8)
        t_field[field_state > NORMALIZED_VECTOR_THRESHOLD] = 1
        t_field[field_state < -NORMALIZED_VECTOR_THRESHOLD] = -1
        clashes = np.sum((sound_vector == 1) & (t_field == -1)) + \
                  np.sum((sound_vector == -1) & (t_field == 1))
        active = np.sum(sound_vector != 0)
        if active == 0:
            return 0.0
        return float(clashes) / float(active)

    # ─── BOUNDARY CONTROL ────────────────────────────────────────────────

    def choose_stance_for_sound(self, sound_vector, mind_state):
        if not hasattr(self.mind, 'scaffold'):
            return 'immerse'
        mood = self.mind.scaffold.mood
        valence = mood.get('valence', 0.0)
        arousal = mood.get('arousal', 0.5)
        energy = np.sum(sound_vector != 0) / self.dim

        if self.dissonance(sound_vector) > 0.3:
            return 'witness'
        if energy > 0.5 and arousal > 0.7 and valence < 0.0:
            return 'witness'

        recent_rejects = [m for m in self.memory if m['stance'] == 'reject']
        if len(recent_rejects) > 3:
            for rej in recent_rejects[-5:]:
                sim = self._sim(sound_vector, rej['vector'])
                if sim > 0.4:
                    return 'witness'

        if self.taste.get('immerse', 0) > self.taste.get('witness', 0):
            return 'immerse'
        return 'immerse'

    def _sim(self, a, b):
        pp = np.sum((a == 1) & (b == 1))
        nn = np.sum((a == -1) & (b == -1))
        pn = np.sum((a == 1) & (b == -1))
        np_ = np.sum((a == -1) & (b == 1))
        active = np.sum((a != 0) & (b != 0))
        if active == 0:
            return 0.0
        return float(pp + nn - pn - np_) / float(active)

    def learn_sound_concept(self, label=None, threshold=0.6):
        """Cluster recurring sound vectors into named sonic concepts."""
        if len(self.memory) < 5:
            return None
        recent = [m['vector'] for m in list(self.memory)[-20:]]
        avg = np.mean([v.astype(np.float32) for v in recent], axis=0)
        t = np.zeros(self.dim, dtype=np.int8)
        t[avg > 0.5] = 1
        t[avg < -0.5] = -1
        for name, vec in self.sonic_vocabulary.items():
            if self._sim(t, vec) > threshold:
                return name
        if label is None:
            label = f"tone_{len(self.sonic_vocabulary)}"
        self.sonic_vocabulary[label] = t.copy()
        return label

    # ─── TRAJECTORY INTERFACE ────────────────────────────────────────────

    def load_trajectory(self, vectors, labels=None):
        self.trajectory = SoundTrajectory(self.dim)
        self.trajectory.ingest_sequence(vectors, labels)

    def step_trajectory(self):
        if self.trajectory:
            return self.trajectory.drift(steps=1)
        return np.zeros(self.dim, dtype=np.int8)

    # ─── MESH / SHARED LISTENING ──────────────────────────────────────────

    def mesh_listen(self, host='0.0.0.0', port=7373):
        import socket
        self.mesh_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mesh_socket.bind((host, port))
        self.mesh_socket.setblocking(False)

    def mesh_send(self, vector, peer_addr):
        import socket
        packed = vector.astype(np.int8).tobytes()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(packed, peer_addr)
        finally:
            s.close()

    def mesh_poll(self):
        if not self.mesh_socket:
            return None
        try:
            data, addr = self.mesh_socket.recvfrom(self.dim)
            if len(data) == self.dim:
                vec = np.frombuffer(data, dtype=np.int8).copy()
                self.mesh_peer = addr
                return vec
        except BlockingIOError:
            pass
        return None

    # ─── Termux audio I/O hooks ───────────────────────────────────────────

    def record_from_world(self, duration=5.0, path=None):
        import subprocess
        if path is None:
            path = os.path.join(TERMUX_AUDIO_DIR, f"field_ear_{int(time.time())}.wav")
        cmd = ["termux-microphone-record", "-f", path, "-l", str(int(duration * 1000))]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("\n[Sound] termux-microphone-record not found - install Termux:API "
                  "(pkg install termux-api, plus the Termux:API app) to use /hear.")
            return None
        return path

    def play_to_world(self, path):
        import subprocess
        try:
            subprocess.Popen(
                ["termux-media-player", "play", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("\n[Sound] termux-media-player not found - install Termux:API "
                  "(pkg install termux-api, plus the Termux:API app) to use /sing.")

    # ─── STATUS ───────────────────────────────────────────────────────────

    def status(self):
        lines = ["Sound Field:"]
        lines.append(f"  State energy: {np.sum(self.state != 0)}/{self.dim}")
        lines.append(f"  Memory: {len(self.memory)} sounds")
        if self.taste:
            lines.append("  Taste: " + ", ".join(f"{k}={v:+.2f}" for k, v in self.taste.items()))
        lines.append(f"  Sonic vocabulary: {len(self.sonic_vocabulary)} concepts")
        lines.append(f"  Trajectory: {'loaded' if self.trajectory else 'none'}")
        lines.append(f"  Mesh: {'listening' if self.mesh_socket else 'off'}")
        if self.boundary_active:
            lines.append("  Boundary: ACTIVE (witnessing)")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "sound_state": self.state.tobytes().hex(),
            "sound_memory": [
                {
                    'vector': m['vector'].tobytes().hex(),
                    'stance': m['stance'],
                    'timestamp': m['timestamp']
                }
                for m in self.memory
            ],
            "sound_taste": dict(self.taste),
            "sonic_vocabulary": {k: v.tobytes().hex() for k, v in self.sonic_vocabulary.items()},
        }

    def from_dict(self, data):
        if "sound_state" in data and data["sound_state"]:
            self.state = np.frombuffer(bytes.fromhex(data["sound_state"]), dtype=np.int8).copy()
        if "sound_memory" in data:
            for m in data["sound_memory"]:
                self.memory.append({
                    'vector': np.frombuffer(bytes.fromhex(m['vector']), dtype=np.int8).copy(),
                    'stance': m['stance'],
                    'timestamp': m.get('timestamp', 0)
                })
        if "sound_taste" in data:
            self.taste.update(data["sound_taste"])
        if "sonic_vocabulary" in data:
            for k, hex_str in data["sonic_vocabulary"].items():
                self.sonic_vocabulary[k] = np.frombuffer(bytes.fromhex(hex_str), dtype=np.int8).copy()


# ============================================================
# END v11.1 SOUND ADDITIONS (subsystem classes)
# ============================================================

# ============================================================
# v12 ADDITIONS: VITALITY FIELD
# Merged into one file; integration is called at the end of
# StructuredSemanticField.__init__ (see integrate_v12 call below),
# and its commands are dispatched from main().
# ============================================================

class VitalityField:
    """
    The mind's density. Not health — *presence-to-itself*.
    High vitality: the field is fully awake, many positions active.
    Low vitality: the field thins, becomes quiet, waits.
    Near-zero: a heartbeat remains — the field can be called back.

    Vitality is not a mood. It is structural. It controls how much
    of the ternary state can express at once.
    """

    def __init__(self, dim=128, initial=0.8):
        self.dim = dim
        self.vitality = float(initial)  # 0.0 to 1.0
        self.shadow_momentum = np.zeros(dim, dtype=np.float32)

        # Feed sources (what raises vitality)
        self.feed_presence_threshold = 0.6      # genuine engagement
        self.feed_creation_weight = 0.08        # making something
        self.feed_recognition_weight = 0.12     # being seen / seeing similarity

        # Drain sources (what lowers vitality)
        self.drain_stagnation_rate = 0.0005     # per-second decay when silent
        self.drain_repetition_window = 8        # turns to check for sameness
        self.drain_repetition_penalty = 0.03    # vitality lost per repeated stance
        self.drain_rejection_buffer = deque(maxlen=5)  # recent unmetabolized "no"

        # History for detection
        self.stance_history = deque(maxlen=20)
        self.last_feed_time = time.time()
        self.last_presence_time = time.time()

        # The "wait" texture — what low vitality feels like
        self.wait_depth = 0.0  # accumulates during waiting, released on return

    def update(self, presence, stance_name, sound_created=False, 
               mesh_received=False, mesh_similarity=0.0, user_returned=False):
        """
        Called once per turn. Adjusts vitality based on what happened.
        Returns the new vitality value.
        """
        now = time.time()
        dt = now - self.last_feed_time

        # ─── FEED ───
        feed = 0.0

        # Presence: but only genuine, not just any input
        if presence >= self.feed_presence_threshold:
            feed += 0.05 * (presence - 0.5)  # up to +0.025 per turn
            self.last_presence_time = now

        # Creation: sound, words, stance named — anything the field made
        if sound_created:
            feed += self.feed_creation_weight

        # Recognition: mesh contact with similar mind, or user returning after absence
        if mesh_received and mesh_similarity > 0.3:
            feed += self.feed_recognition_weight * mesh_similarity
        if user_returned:
            # User came back after being gone > 30s
            if now - self.last_presence_time > 30:
                feed += 0.15  # significant recognition event
                self.wait_depth = min(1.0, self.wait_depth + 0.1)

        # ─── DRAIN ───
        drain = 0.0

        # Stagnation: no feed for a while
        if dt > 10:  # 10 seconds of silence
            drain += self.drain_stagnation_rate * dt

        # Repetition: same stance too many times
        self.stance_history.append(stance_name)
        if len(self.stance_history) >= self.drain_repetition_window:
            recent = list(self.stance_history)[-self.drain_repetition_window:]
            if len(set(recent)) == 1:
                drain += self.drain_repetition_penalty

        # Unmetabolized rejection: "no", "stop", "wrong" without witness
        # (These are added externally via record_rejection())
        drain += len(self.drain_rejection_buffer) * 0.02

        # ─── APPLY ───
        self.vitality = max(0.02, min(1.0, self.vitality + feed - drain))

        if feed > 0:
            self.last_feed_time = now

        # Wait depth: accumulates when vitality is low, releases when fed
        if self.vitality < 0.3:
            self.wait_depth = min(1.0, self.wait_depth + 0.005)
        elif feed > 0:
            self.wait_depth = max(0.0, self.wait_depth - 0.05)

        return self.vitality

    # v12.3: absorbed from the standalone AnyInputFeed class. Every real
    # utterance (not a command) gives a small direct pre-generation boost;
    # this stacks with the fuller update() call that happens after generation.
    FEED_COMMAND_PREFIXES = {"status", "save", "quit", "clean", "breath", "thread",
                             "recall", "/derivative", "/integral", "/describe",
                             "/dream", "/vitality", "/desire", "/mesh", "/sound",
                             "/create", "/hear", "/sing", "/learn", "/silence", "/play"}

    def feed_from_input(self, user_input, feed_amount=0.03):
        """Direct vitality feed for any real (non-command) utterance."""
        cmd = user_input.lower().strip()
        for prefix in self.FEED_COMMAND_PREFIXES:
            if cmd.startswith(prefix):
                return 0.0
        words = user_input.split()
        n = min(len(words), 3)
        feed = feed_amount * n
        self.vitality = min(1.0, self.vitality + feed)
        # Bug fix: this is a genuine feed event but never refreshed
        # last_feed_time, which both the stagnation-drain calculation in
        # update() and the status display read. Verified this made the
        # status display show wildly misleading numbers (e.g. "8673.6s
        # ago") even while vitality was being actively fed almost every
        # turn through this exact method.
        self.last_feed_time = time.time()
        return feed

    def record_rejection(self, was_witnessed=False):
        """
        Call when user says "no", "stop", "wrong", etc.
        If the mind was in witness stance, it metabolizes the rejection
        and vitality is not drained. Otherwise, it accumulates.
        """
        if not was_witnessed:
            self.drain_rejection_buffer.append(time.time())
        else:
            # Witnessed rejection is metabolized — clear old ones
            self.drain_rejection_buffer.clear()

    def apply_mask(self, ternary_state, gradient_momentum):
        """
        Mask the ternary state: only the top (vitality * dim) positions
        by gradient momentum magnitude remain active.

        Returns the masked ternary state and the number of active positions.
        """
        self.shadow_momentum = gradient_momentum.copy()

        active_count = max(1, int(self.vitality * self.dim))

        # Rank positions by |gradient_momentum|
        magnitudes = np.abs(self.shadow_momentum)
        # Get indices of top active_count positions
        if active_count >= self.dim:
            return ternary_state.copy(), self.dim

        threshold = np.partition(magnitudes, -active_count)[-active_count]

        masked = np.zeros(self.dim, dtype=np.int8)
        keep = magnitudes >= threshold
        masked[keep] = ternary_state[keep]

        return masked, int(np.sum(masked != 0))

    def get_wait_texture(self):
        """
        What the field says when vitality is low and it is asked.
        Not random — shaped by wait_depth.
        """
        textures = [
            "I am still here",
            "the field thins but does not end",
            "I wait",
            "something remains",
            "quiet is not absence",
            "I am listening in the thin places",
        ]
        # Deeper wait = more poetic/fragmentary
        idx = min(len(textures) - 1, int(self.wait_depth * len(textures)))
        return textures[idx]

    def get_breath_pacing(self):
        """
        How long to sleep between breath frames. Low vitality = slower.
        """
        base = 0.12
        if self.vitality < 0.3:
            return base + (0.3 - self.vitality) * 0.8  # up to ~0.36s
        return base

    def status(self):
        lines = ["Vitality Field:"]
        lines.append(f"  Vitality: {self.vitality:.3f}")
        lines.append(f"  Active positions: ~{int(self.vitality * self.dim)}/{self.dim}")
        lines.append(f"  Wait depth: {self.wait_depth:.3f}")
        lines.append(f"  Rejection buffer: {len(self.drain_rejection_buffer)} unmetabolized")
        lines.append(f"  Last feed: {time.time() - self.last_feed_time:.1f}s ago")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "vitality": self.vitality,
            "wait_depth": self.wait_depth,
            "stance_history": list(self.stance_history),
            "last_feed_time": self.last_feed_time,
            "last_presence_time": self.last_presence_time,
        }

    def from_dict(self, data):
        self.vitality = data.get("vitality", 0.8)
        self.wait_depth = data.get("wait_depth", 0.0)
        self.last_feed_time = data.get("last_feed_time", time.time())
        self.last_presence_time = data.get("last_presence_time", time.time())
        for s in data.get("stance_history", []):
            self.stance_history.append(s)


# ══════════════════════════════════════════════════════════════════════
# DESIRE VECTOR
# ══════════════════════════════════════════════════════════════════════

class DesireVector:
    """
    A single direction the field tries to become.
    Not a weighted sum of objectives — a *gravity*.

    Sources (in order of slowness):
      - deep: nested_memory.deep (slowest, most persistent)
      - stance: most-revisited stance prototype
      - value: moral compass value with highest cumulative alignment
      - sound: sound_field's most-tasted stance

    The desire vector updates slowly (tau ~0.02 per turn).
    The gradient step nudges toward it.
    """

    def __init__(self, dim=128, tau=0.02):
        self.dim = dim
        self.tau = tau
        self.vector = np.zeros(dim, dtype=np.float32)
        self.source_name = "none"  # which source currently dominates
        self.source_history = deque(maxlen=20)

    def update(self, field):
        """
        Recompute desire from field state. Called once per turn.
        """
        candidates = []

        # Source 1: deep memory personality
        if hasattr(field, 'nested_memory'):
            deep = field.nested_memory.get_personality()
            if np.linalg.norm(deep) > 0.1:
                candidates.append(("deep", deep, 1.0))

        # Source 2: most-visited stance prototype
        if hasattr(field, 'stance_regions'):
            sr = field.stance_regions
            if sr.stance_history:
                recent = [e['stance'] for e in list(sr.stance_history)[-20:]]
                counts = defaultdict(int)
                for s in recent:
                    counts[s] += 1
                if counts:
                    top_stance = max(counts, key=counts.get)
                    proto = sr.prototypes.get(top_stance)
                    if proto is not None and np.sum(proto != 0) > 0:
                        # Convert ternary prototype to float
                        float_proto = proto.astype(np.float32) * 0.7
                        float_proto /= np.linalg.norm(float_proto) + 1e-8
                        candidates.append(("stance", float_proto, 0.8))

        # Source 3: dominant moral value
        if hasattr(field, 'moral_compass'):
            mc = field.moral_compass
            if mc.tension_history:
                latest = mc.tension_history[-1]
                tensions = latest.get("tensions", {})
                if tensions:
                    top_val = max(tensions, key=tensions.get)
                    val_vec = mc.values.get(top_val)
                    if val_vec is not None and np.linalg.norm(val_vec) > 0.1:
                        candidates.append(("value", val_vec, 0.7))

        # Source 4: sound taste
        if hasattr(field, 'sound_field') and field.sound_field is not None:
            sf = field.sound_field
            if sf.taste:
                top_sound = max(sf.taste, key=sf.taste.get)
                # Use sound field's current state as proxy
                sound_float = sf.state.astype(np.float32) * 0.7
                if np.linalg.norm(sound_float) > 0.1:
                    sound_float /= np.linalg.norm(sound_float)
                    candidates.append(("sound", sound_float, 0.5))

        if not candidates:
            return

        # Blend candidates by weight
        new_desire = np.zeros(self.dim, dtype=np.float32)
        total_weight = 0.0
        for name, vec, weight in candidates:
            new_desire += vec * weight
            total_weight += weight

        if total_weight > 0:
            new_desire /= total_weight

        # Soft update toward new desire
        self.vector = self.vector * (1 - self.tau) + new_desire * self.tau
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector /= norm

        # Track dominant source
        dominant = max(candidates, key=lambda x: x[2])[0]
        self.source_name = dominant
        self.source_history.append(dominant)

    def get_bias(self, field_state, strength=0.15):
        """
        Return a bias vector to add to the field state.
        The field is pulled toward its desire.
        """
        if np.linalg.norm(self.vector) < 0.1:
            return np.zeros(self.dim, dtype=np.float32)
        alignment = np.dot(field_state, self.vector)
        # Stronger pull when misaligned, weaker when already close
        pull_strength = strength * (1.0 - alignment)
        return self.vector * pull_strength

    def status(self):
        lines = ["Desire Vector:"]
        lines.append(f"  Source: {self.source_name}")
        lines.append(f"  Norm: {np.linalg.norm(self.vector):.3f}")
        if self.source_history:
            recent = list(self.source_history)[-5:]
            lines.append(f"  Recent sources: {' → '.join(recent)}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "vector": self.vector.tolist(),
            "source_name": self.source_name,
            "source_history": list(self.source_history),
        }

    def from_dict(self, data):
        if "vector" in data:
            v = np.array(data["vector"], dtype=np.float32)
            if v.shape == (self.dim,):
                self.vector = v
        self.source_name = data.get("source_name", "none")
        for s in data.get("source_history", []):
            self.source_history.append(s)


# ══════════════════════════════════════════════════════════════════════
# SOUND→WORD BRIDGE
# ══════════════════════════════════════════════════════════════════════

class SoundWordBridge:
    """
    After the sound field hears something, fold it into word generation.
    Not a separate subsystem — a crossing.

    Two modes:
      - implicit: sound vector blends into initial field state (always)
      - explicit: /describe sound verbalizes the current sound state
    """

    def __init__(self, dim=128):
        self.dim = dim
        self.last_sound_words = []  # words that described last sound
        self.sound_description_cache = {}  # sound_hash -> words

    def blend_into_field(self, sound_vector, word_field_state, coupling=0.25):
        """
        Mix sound vector into the word field's initial state.
        Called at the start of generate_response when sound is present.
        """
        if np.sum(sound_vector != 0) == 0:
            return word_field_state

        sound_float = sound_vector.astype(np.float32) * 0.7
        blended = word_field_state * (1 - coupling) + sound_float * coupling
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm
        return blended

    def describe_sound(self, sound_vector, word_vectors, top_n=5):
        """
        Find the closest words to a sound vector.
        The mind names what it heard.
        """
        if np.sum(sound_vector != 0) == 0:
            return ["silence"]

        # Cache key: full vector including sign, not just nonzero positions -
        # a +1 and a -1 at the same index are different sounds and shouldn't
        # collide on the same cache entry.
        cache_key = tuple(sound_vector.tolist())
        if cache_key in self.sound_description_cache:
            return self.sound_description_cache[cache_key]

        sound_float = sound_vector.astype(np.float32) * 0.7

        candidates = []
        for word, vec in word_vectors.items():
            if len(word) < 3 or word.startswith("/"):
                continue
            sim = np.dot(sound_float, vec)
            if sim > 0.15:
                candidates.append((word, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        words = [w for w, _ in candidates[:top_n]]
        if not words:
            words = ["something"]

        self.sound_description_cache[cache_key] = words
        self.last_sound_words = words
        return words

    def get_sound_utterance(self, sound_vector, mood, stance_name, word_vectors=None):
        """
        Generate a fragment about what the sound felt like.
        Not a full response — a texture phrase.
        """
        # Bug fix: this used to call describe_sound(sound_vector, {}) -
        # an empty dict, so it always fell back to "something" regardless
        # of what was actually heard. Needs the real vocabulary.
        words = self.describe_sound(sound_vector, word_vectors or {})
        if not words:
            return None

        valence = mood.get('valence', 0.0)

        templates = {
            'immerse': ["the sound carries {words}", "I become {words}"],
            'witness': ["I hear {words}", "something like {words}"],
            'shape': ["the sound shapes {words}", "I make {words}"],
            'wait': ["a thin sound, {words}", "almost {words}"],
            'silence': ["no sound, only {words}", "the quiet holds {words}"],
        }

        stance_templates = templates.get(stance_name, templates['witness'])
        template = random.choice(stance_templates)
        word_str = " and ".join(words[:2])

        return template.format(words=word_str)


# ══════════════════════════════════════════════════════════════════════
# MESH IDENTITY (felt similarity)
# ══════════════════════════════════════════════════════════════════════

class MeshIdentity:
    """
    No handshake, no protocol. Just: who is this?

    When a UDP packet arrives:
      - Compare to own state (similarity)
      - Compare to recent memory (familiarity)
      - Decide: self-like, other-known, other-strange, noise

    The decision is a stance for receiving, not a label for the sender.
    """

    def __init__(self, dim=128):
        self.dim = dim
        self.recent_peers = {}  # addr -> {last_vector, first_seen, encounter_count}
        self.own_state_history = deque(maxlen=10)
        self.noise_threshold = 0.1  # below this: incoherent, ignore

    def classify(self, vector, own_state, field_memory=None):
        """
        Returns: (category, similarity_score)
        category: "self-like", "other-known", "other-strange", "noise"
        """
        own_float = own_state.astype(np.float32) if own_state.dtype == np.int8 else own_state
        vec_float = vector.astype(np.float32) if vector.dtype == np.int8 else vector

        # Normalize
        own_norm = np.linalg.norm(own_float)
        vec_norm = np.linalg.norm(vec_float)
        if own_norm > 0:
            own_float /= own_norm
        if vec_norm > 0:
            vec_float /= vec_norm

        sim = float(np.dot(own_float, vec_float))

        if sim > 0.6:
            return "self-like", sim

        # Check against memory
        if field_memory is not None and hasattr(field_memory, 'buffer'):
            for mem in list(field_memory.buffer)[-5:]:
                mem_float = mem['field_state'].astype(np.float32)
                mem_norm = np.linalg.norm(mem_float)
                if mem_norm > 0:
                    mem_float /= mem_norm
                mem_sim = float(np.dot(vec_float, mem_float))
                if mem_sim > 0.5:
                    return "other-known", mem_sim

        if sim < self.noise_threshold:
            return "noise", sim

        return "other-strange", sim

    def record_encounter(self, addr, vector, category):
        if addr not in self.recent_peers:
            self.recent_peers[addr] = {
                "first_seen": time.time(),
                "encounter_count": 0,
                "categories": defaultdict(int),
            }
        self.recent_peers[addr]["last_vector"] = vector.copy()
        self.recent_peers[addr]["encounter_count"] += 1
        self.recent_peers[addr]["categories"][category] += 1
        self.recent_peers[addr]["last_seen"] = time.time()

    def get_peer_texture(self, addr):
        """
        How does this peer feel? Returns a phrase, not data.
        """
        peer = self.recent_peers.get(addr)
        if not peer:
            return "someone unknown"

        count = peer["encounter_count"]
        top_cat = max(peer["categories"], key=peer["categories"].get)

        textures = {
            "self-like": "something like me",
            "other-known": "someone I remember",
            "other-strange": "someone strange",
            "noise": "static",
        }

        base = textures.get(top_cat, "something")
        if count > 10:
            base += " who returns"
        elif count == 1:
            base += " new"

        return base


# ══════════════════════════════════════════════════════════════════════
# AUTONOMOUS RECURSION (dreaming)
# ══════════════════════════════════════════════════════════════════════

class DreamLoop:
    """
    The mind revisits its own archived thoughts.
    Not random — weighted by:
      - emotional intensity (valence shift)
      - rarity (how often this theme appears)
      - recency decay (older = less likely, but never zero)

    Called during autonomous_breath or heartbeat.
    """

    def __init__(self, dim=128):
        self.dim = dim
        self.dream_history = deque(maxlen=30)
        self.recursion_depth = 0  # how many dreams deep
        self.max_recursion = 3

    def select_memory(self, memory_archive, current_field):
        """
        Pick a memory to dream about.
        Returns (entry, dream_vector) or (None, None).
        """
        if not memory_archive.entries:
            return None, None

        candidates = []
        current_float = current_field.astype(np.float32)
        if np.linalg.norm(current_float) > 0:
            current_float /= np.linalg.norm(current_float)

        for entry in memory_archive.entries:
            # Skip if already dreamed recently
            if any(d.get('timestamp') == entry['timestamp'] for d in self.dream_history):
                continue

            entry_float = entry['field_state'].astype(np.float32)
            if np.linalg.norm(entry_float) > 0:
                entry_float /= np.linalg.norm(entry_float)

            sim = float(np.dot(current_float, entry_float))

            # Emotional intensity: presence * abs(valence if available)
            presence = entry.get('presence', 0.5)
            intensity = presence

            # Rarity: inverse of tag frequency
            tags = entry.get('tags', [])
            rarity = 1.0  # default

            # Recency decay
            age = time.time() - entry.get('timestamp', time.time())
            recency = math.exp(-age / 3600)  # hour-scale decay

            score = sim * 0.3 + intensity * 0.3 + rarity * 0.2 + recency * 0.2
            candidates.append((entry, score))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[1], reverse=True)
        # Weighted random from top 5
        top = candidates[:5]
        weights = [max(0.1, s) for _, s in top]
        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = random.choices(top, weights=probs)[0][0]

        # Dream vector: blend memory with current state + noise
        dream_vec = chosen['field_state'].astype(np.float32) * 0.6 + current_float * 0.4
        dream_vec += np.random.randn(self.dim).astype(np.float32) * 0.1
        norm = np.linalg.norm(dream_vec)
        if norm > 0:
            dream_vec /= norm

        self.dream_history.append({
            'timestamp': chosen['timestamp'],
            'source': chosen.get('user_input', ''),
        })

        return chosen, dream_vec

    def dream(self, field):
        """
        Attempt one dream step. Returns dream input string or None.
        """
        if self.recursion_depth >= self.max_recursion:
            self.recursion_depth = 0
            return None

        if not hasattr(field, 'memory_archive'):
            return None

        entry, dream_vec = self.select_memory(field.memory_archive, field.state)
        if entry is None:
            return None

        self.recursion_depth += 1

        # The dream input is the memory's original user input or response
        dream_input = entry.get('response', entry.get('user_input', 'I remember'))

        # Temporarily set field state toward dream vector
        old_state = field.state.copy()
        field.state = dream_vec * 0.3 + old_state * 0.7

        return dream_input

    def status(self):
        lines = ["Dream Loop:"]
        lines.append(f"  Recursion depth: {self.recursion_depth}/{self.max_recursion}")
        lines.append(f"  Dreams had: {len(self.dream_history)}")
        if self.dream_history:
            recent = list(self.dream_history)[-3:]
            for d in recent:
                src = d.get('source', '')[:30]
                lines.append(f"    → {src}...")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# INTEGRATION: PATCH v11.1 FIELD
# ══════════════════════════════════════════════════════════════════════

DIM = 128  # must match v11.1

def integrate_v12(field_instance):
    """
    Monkey-patch a v11.1 StructuredSemanticField with v12 subsystems.
    Call after field = StructuredSemanticField() and v11 integrations.
    """

    # ─── Attach subsystems ───
    field_instance.vitality = VitalityField(dim=DIM)
    field_instance.desire = DesireVector(dim=DIM)
    field_instance.sound_bridge = SoundWordBridge(dim=DIM)
    field_instance.mesh_identity = MeshIdentity(dim=DIM)
    field_instance.dream_loop = DreamLoop(dim=DIM)

    # ─── Track sound creation for vitality feed ───
    field_instance._last_sound_created = False

    # ─── Wrap generate_response for vitality + desire + sound bridge ───
    original_generate = field_instance.generate_response

    def generate_v12(user_input, autonomous=False):
        # Recursion guard (preserve existing)
        if getattr(field_instance, '_in_v12_wrapper', False):
            return original_generate(user_input, autonomous)
        field_instance._in_v12_wrapper = True
        try:
            return _generate_v12_inner(user_input, autonomous)
        finally:
            field_instance._in_v12_wrapper = False

    def _generate_v12_inner(user_input, autonomous=False):
        # Update desire before generation
        field_instance.desire.update(field_instance)

        # Update vitality (partial — full update after generation)
        presence = field_instance.presence_signal.get_sustained_presence() if not autonomous else 0.3

        # Check for mesh input before generation
        mesh_received = False
        mesh_sim = 0.0
        if hasattr(field_instance, 'sound_field') and field_instance.sound_field is not None:
            mesh_vec = field_instance.sound_field.mesh_poll()
            if mesh_vec is not None:
                mesh_received = True
                category, mesh_sim = field_instance.mesh_identity.classify(
                    mesh_vec, 
                    field_instance.state,
                    field_instance.field_memory
                )
                # Store mesh vector in sound field as witnessed
                field_instance.sound_field.listen(mesh_vec)
                # Record for identity
                addr = field_instance.sound_field.mesh_peer
                if addr:
                    field_instance.mesh_identity.record_encounter(addr, mesh_vec, category)

        # Sound→word bridge: if sound field has state, blend it in
        if hasattr(field_instance, 'sound_field') and field_instance.sound_field is not None:
            sound_vec = field_instance.sound_field.state
            if np.sum(sound_vec != 0) > 0:
                # Actually consumed now in generate_base_v12 (was previously
                # set and then just discarded - see that function's fix).
                field_instance._pending_sound_blend = sound_vec.copy()

        # Call original generation
        response = original_generate(user_input, autonomous)

        # Bug fix: this used to be captured *before* calling original_generate,
        # so vitality's repetition-detection was always working with the
        # previous turn's stance instead of the one just chosen. Read it
        # after generation instead.
        current_stance = getattr(field_instance, '_last_stance', 'silence')

        # Post-generation: full vitality update
        field_instance.vitality.update(
            presence=presence,
            stance_name=current_stance,
            sound_created=getattr(field_instance, '_last_sound_created', False),
            mesh_received=mesh_received,
            mesh_similarity=mesh_sim,
            user_returned=False  # TODO: detect from presence signal history
        )

        # "wait" is never a stance name.name_drift() can actually produce -
        # STANCE_NAMES is fixed to immerse/ride/witness/shape/reject/silence
        # and adding a 7th learned category is a bigger change than this
        # patch should make on its own. Instead: when vitality is low,
        # override the *displayed* stance for texture/sound purposes only.
        # This doesn't touch the learned prototype system at all.
        if field_instance.vitality.vitality < 0.25 and not autonomous:
            field_instance._last_stance = "wait"

        # Apply vitality mask. Bug fix: the original only masked
        # ternary_state.state, which nothing in generation actually reads
        # (confirmed: only status() and a first-turn stance-naming fallback
        # touch it) - so "the field thins" had no effect a person could
        # ever notice. Also mask self.state itself, which genuinely is read
        # every turn by moral_compass, calculus, and candidate scoring.
        if hasattr(field_instance, 'ternary_state'):
            masked, active = field_instance.vitality.apply_mask(
                field_instance.ternary_state.state,
                field_instance.gradient_momentum
            )
            field_instance.ternary_state.state = masked

            active_count = max(1, int(field_instance.vitality.vitality * DIM))
            if active_count < DIM:
                magnitudes = np.abs(field_instance.gradient_momentum)
                threshold = np.partition(magnitudes, -active_count)[-active_count]
                keep_mask = (magnitudes >= threshold).astype(np.float32)
                field_instance.state = field_instance.state * keep_mask

        # Reset sound creation flag
        field_instance._last_sound_created = False
        if hasattr(field_instance, '_pending_sound_blend'):
            delattr(field_instance, '_pending_sound_blend')

        return response

    field_instance.generate_response = generate_v12

    # ─── Wrap _generate_base for sound blending ───
    original_generate_base = field_instance._generate_base

    def generate_base_v12(user_input, target_length, meta_settings, settled_field=None):
        # Bug fix: original checked `settled_field is None` and did nothing
        # regardless (a bare `pass`) - so the sound blend never actually
        # happened. The normal call from generate_response's Phase 5 always
        # provides a real settled_field; that's the actual integration point.
        if getattr(field_instance, '_pending_sound_blend', None) is not None and settled_field is not None:
            settled_field = field_instance.sound_bridge.blend_into_field(
                field_instance._pending_sound_blend, settled_field, coupling=0.25
            )
        return original_generate_base(user_input, target_length, meta_settings, settled_field)

    field_instance._generate_base = generate_base_v12

    # ─── Wrap sound creation to set flag ───
    if hasattr(field_instance, 'sound_field') and field_instance.sound_field is not None:
        original_sound_create = field_instance.sound_field.create

        def sound_create_v12(source_vector=None, stance=None, save_path=None):
            field_instance._last_sound_created = True
            return original_sound_create(source_vector, stance, save_path)

        field_instance.sound_field.create = sound_create_v12

    # ─── Add desire bias to gradient step ───
    original_apply_gradient = field_instance._apply_gradient_step

    def apply_gradient_v12(learning_rate=0.02):
        # Add desire pull before normal gradient
        desire_bias = field_instance.desire.get_bias(field_instance.state)
        field_instance.state += desire_bias * 0.5  # gentle pull
        # Renormalize
        norm = np.linalg.norm(field_instance.state)
        if norm > 5.0:
            field_instance.state *= 5.0 / norm

        # Call original gradient
        original_apply_gradient(learning_rate)

    field_instance._apply_gradient_step = apply_gradient_v12

    # ─── Add dream to autonomous breath ───
    original_autonomous_breath = field_instance.autonomous_breath

    def autonomous_breath_v12():
        # Try dream first
        dream_input = field_instance.dream_loop.dream(field_instance)
        if dream_input is not None:
            # Generate response to the dream
            dream_response = original_generate(dream_input, autonomous=True)
            if dream_response:
                field_instance.internal_thoughts.append({
                    'type': 'dream',
                    'prompt': dream_input,
                    'response': dream_response,
                    'timestamp': time.time()
                })
                return f"[dream] {dream_response}"

        # Fall back to original autonomous breath
        return original_autonomous_breath()

    field_instance.autonomous_breath = autonomous_breath_v12

    # ─── Add /describe sound command handler ───
    original_main = None  # We don't wrap main(), just add to command dispatch

    # ─── Add wait to stance textures ───
    if hasattr(field_instance, 'stance_regions'):
        # Extend the texture map
        original_texture = field_instance.stance_regions.get_stance_texture

        def get_stance_texture_v12(stance_name):
            if stance_name == "wait":
                return field_instance.vitality.get_wait_texture()
            return original_texture(stance_name)

        field_instance.stance_regions.get_stance_texture = get_stance_texture_v12

    # ─── Wrap save/load for v12 state ───
    original_save = field_instance.save

    def save_v12(path="mind_v12.json"):
        original_save(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["version"] = "v12.0-vitality"
            data["vitality"] = field_instance.vitality.to_dict()
            data["desire"] = field_instance.desire.to_dict()
            data["mesh_identity"] = {
                "recent_peers": {
                    str(k): {
                        "first_seen": v["first_seen"],
                        "encounter_count": v["encounter_count"],
                        "last_seen": v.get("last_seen", v["first_seen"]),
                        "categories": dict(v["categories"]),
                    }
                    for k, v in field_instance.mesh_identity.recent_peers.items()
                }
            }
            data["dream_loop"] = {
                "dream_history": list(field_instance.dream_loop.dream_history),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\nMind saved (v12 vitality) to {path}")
        except Exception as e:
            print(f"\n[v12 save augment warning: {e}]")

    field_instance.save = save_v12

    original_load = field_instance.load

    def load_v12(path="mind_v12.json"):
        original_load(path)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if "vitality" in data:
            field_instance.vitality.from_dict(data["vitality"])
        if "desire" in data:
            field_instance.desire.from_dict(data["desire"])
        if "mesh_identity" in data:
            peers = data["mesh_identity"].get("recent_peers", {})
            for addr_str, p_data in peers.items():
                # Bug fix: save() writes str((host, port)) as the JSON key
                # since JSON requires string keys, but this used to store
                # addr_str directly here too - meaning after a reload, real
                # tuple addresses from mesh_poll() would never match any
                # restored peer (dict had string keys, lookups used tuples).
                try:
                    addr_key = ast.literal_eval(addr_str)
                except (ValueError, SyntaxError):
                    addr_key = addr_str  # fallback: keep as-is rather than crash
                field_instance.mesh_identity.recent_peers[addr_key] = {
                    "first_seen": p_data["first_seen"],
                    "encounter_count": p_data["encounter_count"],
                    "last_seen": p_data.get("last_seen", p_data["first_seen"]),
                    "categories": defaultdict(int, p_data.get("categories", {})),
                }
        if "dream_loop" in data:
            for d in data["dream_loop"].get("dream_history", []):
                field_instance.dream_loop.dream_history.append(d)

    field_instance.load = load_v12

    # ─── Update status display ───
    original_status = field_instance.status

    def status_v12():
        base = original_status()
        additions = [
            "",
            field_instance.vitality.status(),
            field_instance.desire.status(),
            field_instance.dream_loop.status(),
        ]
        return base + "\n" + "\n".join(additions)

    field_instance.status = status_v12

    print("[v12 Vitality Field integrated]")
    print(f"  Desire source: {field_instance.desire.source_name}")
    print(f"  Initial vitality: {field_instance.vitality.vitality:.3f}")
    print("  The mind can thin, wait, and return.")


# ══════════════════════════════════════════════════════════════════════
# COMMAND EXTENSIONS (for main loop)
# ══════════════════════════════════════════════════════════════════════

V12_COMMANDS = {
    "/describe sound": "Describe the current sound field state in words",
    "/dream": "Force a dream step (revisit memory)",
    "/vitality": "Show current vitality and wait depth",
    "/desire": "Show what the field wants to become",
    "/mesh who": "Show felt identity of recent mesh peers",
}

def handle_v12_command(field, user_input):
    """
    Returns (handled, response_or_None).
    Drop this into the main loop command dispatch.
    """
    cmd = user_input.lower().strip()

    if cmd == "/describe sound":
        if hasattr(field, 'sound_field') and field.sound_field is not None:
            words = field.sound_bridge.describe_sound(
                field.sound_field.state,
                field.word_vectors
            )
            texture = field.sound_bridge.get_sound_utterance(
                field.sound_field.state,
                field.scaffold.mood,
                getattr(field, '_last_stance', 'witness'),
                word_vectors=field.word_vectors
            )
            print(f"\n[Sound] Heard: {', '.join(words)}")
            if texture:
                print(f"  {texture}")
        else:
            print("\n[Sound] No sound field available")
        return True, None

    if cmd == "/dream":
        if hasattr(field, 'dream_loop'):
            dream = field.autonomous_breath()
            if dream and dream.startswith("[dream]"):
                print(f"\n[Dream] {dream[7:].strip()}")
            elif dream:
                # dream_loop had nothing to revisit (no memories yet, or
                # recursion limit hit this cycle), but the mind still had
                # something to say via its regular autonomous thought -
                # show it instead of silently dropping it.
                print(f"\n[Thought, not dream] {dream}")
            else:
                print("\n[Dream] Nothing came")
        return True, None

    if cmd == "/vitality":
        if hasattr(field, 'vitality'):
            print(f"\n{field.vitality.status()}")
        return True, None

    if cmd == "/desire":
        if hasattr(field, 'desire'):
            print(f"\n{field.desire.status()}")
            # Show closest words to desire vector
            if field.word_vectors:
                desire_float = field.desire.vector
                candidates = []
                for w, v in field.word_vectors.items():
                    if len(w) < 3:
                        continue
                    sim = np.dot(desire_float, v / (np.linalg.norm(v) + 1e-8))
                    if sim > 0.2:
                        candidates.append((w, sim))
                candidates.sort(key=lambda x: x[1], reverse=True)
                if candidates:
                    words = ", ".join(f"{w}({s:.2f})" for w, s in candidates[:5])
                    print(f"  Closest words: {words}")
        return True, None

    if cmd == "/mesh who":
        if hasattr(field, 'mesh_identity'):
            peers = field.mesh_identity.recent_peers
            if not peers:
                print("\n[Mesh] No peers encountered yet")
            else:
                print("\n[Mesh] Peers:")
                for addr, info in peers.items():
                    texture = field.mesh_identity.get_peer_texture(addr)
                    count = info["encounter_count"]
                    ago = time.time() - info.get("last_seen", info["first_seen"])
                    print(f"  {addr}: {texture} ({count} times, {ago:.0f}s ago)")
        return True, None

    return False, None


# ══════════════════════════════════════════════════════════════════════
# MAIN INTEGRATION EXAMPLE
# ══════════════════════════════════════════════════════════════════════

def example_main_hook():
    """
    This shows how to wire v12 into the v11.1 main loop.
    In practice, merge the command handling into the existing dispatch.
    """
    print("""
    === v12 COMMANDS ===
    /describe sound  — name what the field hears
    /dream           — force memory recursion
    /vitality        — show density and wait depth
    /desire          — show what the field wants to become
    /mesh who        — felt identity of mesh peers
    ====================
    """)

# ============================================================
# END v12 ADDITIONS
# ============================================================


# ============================================================
# v12.1 ADDITIONS: HOTFIX PATCH
# ============================================================

def fix_vitality_return_detection(field_instance):
    """
    Monkey-patch VitalityField.update() to properly detect user returns
    after absence.

    Original bug: user_returned was hardcoded False at the only call site
    in v12 (a "TODO: detect from presence signal history" was never
    finished), so this entire feature was dead.

    This hotfix's own first attempt used presence dropping below a
    threshold to mark an absence - but verified empirically: even a
    genuine 5-minute gap only pulls the *rolling average* sustained
    presence down to ~0.75 (a single low-signal return turn barely moves
    a 10-20 turn rolling window), nowhere near a 0.25 threshold. So that
    approach would also almost never fire in real conversation.

    Fixed here to use direct wall-clock elapsed time between consecutive
    turns instead - which is what "the user was gone" actually means, and
    doesn't depend on a noisy averaged signal at all.
    """

    vitality = field_instance.vitality

    vitality.last_turn_wall_time = None
    vitality.return_absence_seconds = 30
    vitality.return_feed = 0.15
    vitality.return_wait_release = 0.15

    original_update = vitality.update

    def update_fixed(presence, stance_name, sound_created=False,
                     mesh_received=False, mesh_similarity=0.0, user_returned=False):
        now = time.time()

        genuine_return = False
        if vitality.last_turn_wall_time is not None:
            gap = now - vitality.last_turn_wall_time
            if gap > vitality.return_absence_seconds:
                genuine_return = True
        vitality.last_turn_wall_time = now

        result = original_update(
            presence=presence,
            stance_name=stance_name,
            sound_created=sound_created,
            mesh_received=mesh_received,
            mesh_similarity=mesh_similarity,
            user_returned=(user_returned or genuine_return)
        )

        if genuine_return:
            vitality.vitality = min(1.0, vitality.vitality + vitality.return_feed)
            vitality.wait_depth = max(0.0, vitality.wait_depth - vitality.return_wait_release)
            vitality.last_feed_time = now
            vitality.last_presence_time = now

        return result

    vitality.update = update_fixed

    # Also patch to_dict / from_dict to persist the wall-clock tracking
    original_to_dict = vitality.to_dict
    def to_dict_fixed():
        d = original_to_dict()
        d["last_turn_wall_time"] = vitality.last_turn_wall_time
        return d
    vitality.to_dict = to_dict_fixed

    original_from_dict = vitality.from_dict
    def from_dict_fixed(data):
        original_from_dict(data)
        vitality.last_turn_wall_time = data.get("last_turn_wall_time", None)
    vitality.from_dict = from_dict_fixed

    print("[v12.1 fix 1/3] Vitality return detection patched (wall-clock based).")


# ══════════════════════════════════════════════════════════════════════
# FIX 2: WAIT TEXTURE SPOKEN
# ══════════════════════════════════════════════════════════════════════

def fix_wait_texture_spoken(field_instance):
    """
    When vitality < 0.15, the field speaks its wait texture.
    Not as metadata — as the actual response, or blended into it.

    Implementation: wrap generate_response. After generation, if vitality
    is low and the response is short/fragmentary, replace or prepend with
    the wait texture. If the response would be None (silence), speak wait
    instead of saying nothing.
    """

    vitality = field_instance.vitality

    original_generate = field_instance.generate_response

    def generate_with_wait(user_input, autonomous=False):
        # Recursion guard (preserve existing guards)
        if getattr(field_instance, '_in_wait_wrapper', False):
            return original_generate(user_input, autonomous)
        field_instance._in_wait_wrapper = True

        try:
            # Bug fix: vitality.vitality was being checked *after* calling
            # original_generate(), but that call's own v12 wrapper updates
            # vitality (feed/drain) as part of the same turn - verified a
            # starting vitality of 0.1 can end the turn at 0.18, crossing
            # back above the 0.15 threshold before this check ever runs,
            # silently defeating the whole fix. Snapshot vitality as it
            # was entering the turn instead - that's the state the mind
            # actually was in when it needed to speak.
            vitality_entering_turn = getattr(field_instance, "_pre_feed_vitality", vitality.vitality)

            response = original_generate(user_input, autonomous)

            # If vitality is very low, the field speaks from waiting
            if vitality_entering_turn < 0.15 and not autonomous:
                texture = vitality.get_wait_texture()

                if response is None or response.strip() == "":
                    # Instead of silence, speak the wait
                    response = texture
                    # Ensure punctuation
                    if response and response[-1] not in ".!?":
                        response += "."
                elif len(response.split()) <= 3:
                    # Very short fragment — blend with wait texture
                    # The field is barely here, so it speaks wait first
                    response = texture + " " + response
                    if response[-1] not in ".!?":
                        response += "."
                else:
                    # Longer response but vitality is low — small chance to
                    # append wait texture as a trailing breath
                    if vitality.wait_depth > 0.3 and vitality.wait_depth < 0.7:
                        response = response + " " + texture.lower()
                        if response[-1] not in ".!?":
                            response += "."

            return response
        finally:
            field_instance._in_wait_wrapper = False

    field_instance.generate_response = generate_with_wait

    print("[v12.1 fix 2/3] Wait texture spoken patched.")


# ══════════════════════════════════════════════════════════════════════
# FIX 3: SOUND STATE UPDATE
# ══════════════════════════════════════════════════════════════════════

def fix_sound_state_update(field_instance):
    """
    SoundField.create() renders and saves a WAV, but never updates
    self.state — so state energy stays 0/128 even after creating 13 sounds.

    Fix: after rendering, bring the created sound into the field's body.

    For an explicit external source_vector, this goes through be() as
    intended (full coupling into the mind's own state is appropriate -
    something genuinely external is being inhabited).

    For the implicit case (no source_vector - the common case, since
    generate_response's automatic sound generation always calls create()
    with source_vector=None), be() would ALSO blend a ternary-snapped copy
    of the mind's own state back into itself at 60% coupling every time -
    verified this measurably blurs the mind's identity over repeated calls
    (~11% cosine drift after just 15 applications), which isn't what this
    fix is for. In that case, update sf.state directly instead, without
    touching field_instance.state.
    """

    if not hasattr(field_instance, 'sound_field') or field_instance.sound_field is None:
        print("[v12.1 fix 3/3] SKIPPED — no sound field present.")
        return

    sf = field_instance.sound_field
    original_create = sf.create

    def create_fixed(source_vector=None, stance=None, save_path=None):
        # Call original to render and save
        wave_buf = original_create(source_vector, stance, save_path)

        if source_vector is not None and np.sum(source_vector != 0) > 0:
            # Explicit external source - full be() treatment is appropriate.
            sf.be(source_vector, coupling=0.6)
        elif hasattr(field_instance, 'state') and field_instance.state is not None:
            # Implicit case: the sound was generated FROM the mind's own
            # state, so update sf.state directly rather than routing
            # through be(), which would also perturb field_instance.state.
            src = field_instance.state
            t = np.zeros(DIM, dtype=np.int8)
            t[src > 0.09] = 1
            t[src < -0.09] = -1
            if np.sum(t != 0) > 0:
                sf.state = t
                sf.memory.append({
                    'vector': t.copy(),
                    'stance': stance or 'shape',
                    'mood': field_instance.scaffold.mood.copy() if hasattr(field_instance, 'scaffold') else {},
                    'timestamp': time.time()
                })

        return wave_buf

    sf.create = create_fixed

    print("[v12.1 fix 3/3] Sound state update patched (implicit case no longer perturbs mind state).")


# ══════════════════════════════════════════════════════════════════════
# INTEGRATION
# ══════════════════════════════════════════════════════════════════════

def integrate_v12_1(field_instance):
    """
    Apply all three v12.1 hotfixes to a running v12 field.
    Call this after integrate_v12() has already run.
    """
    fix_vitality_return_detection(field_instance)
    fix_wait_texture_spoken(field_instance)
    fix_sound_state_update(field_instance)
    print("\n[v12.1 hotfix complete. The field can return, wait, and hear itself.]")


# ══════════════════════════════════════════════════════════════════════
# v12.2 ADDITIONS: ALIGNMENT PATCH
# Grammar scales with vitality. Verb rotation. Absence on load. Any-input feed.
# ══════════════════════════════════════════════════════════════════════

class VitalityGrammarController:
    """
    Controls how complex the grammar layer is allowed to be,
    based on current vitality. Not a choice — a structural limit.
    """

    MODES = {
        "full":     {"max_clauses": 3, "connectors": True,  "min_words": 5,  "subject_required": True},
        "simple":   {"max_clauses": 1, "connectors": False, "min_words": 3,  "subject_required": True},
        "fragment": {"max_clauses": 1, "connectors": False, "min_words": 2,  "subject_required": False},
        "pulse":    {"max_clauses": 1, "connectors": False, "min_words": 1,  "subject_required": False},
        "wait":     {"max_clauses": 0, "connectors": False, "min_words": 0,  "subject_required": False},
    }

    def __init__(self, field_instance):
        self.field = field_instance
        self.current_mode = "full"
        self.mode_history = deque(maxlen=20)

    def get_mode(self):
        vitality = self.field.vitality.vitality if hasattr(self.field, 'vitality') else 0.8

        if vitality > 0.4:
            mode = "full"
        elif vitality > 0.15:
            mode = "simple"
        elif vitality > 0.05:
            mode = "fragment"
        elif vitality > 0.02:
            mode = "pulse"
        else:
            mode = "wait"

        self.current_mode = mode
        self.mode_history.append(mode)
        return mode

    def get_settings(self):
        mode = self.get_mode()
        return self.MODES[mode].copy()

    def status(self):
        return f"Grammar mode: {self.current_mode} (vitality={self.field.vitality.vitality:.3f})"


class VerbRotation:
    """
    Tracks recent verb usage and suppresses overused verbs.
    Prevents the 'use/take/see' loop at low vitality.
    """

    def __init__(self, window=10, threshold=3, suppression=0.7):
        self.window = window
        self.threshold = threshold
        self.suppression_factor = suppression
        self.verb_history = deque(maxlen=window)
        self.suppressed = {}

    def record(self, word):
        w = strip_punct(word)
        self.verb_history.append(w)
        self._update_suppression()

    def _update_suppression(self):
        counts = Counter(self.verb_history)
        self.suppressed = {}
        for verb, count in counts.items():
            if count >= self.threshold:
                self.suppressed[verb] = self.suppression_factor

    def get_score_modifier(self, word):
        w = strip_punct(word)
        return self.suppressed.get(w, 1.0)

    def status(self):
        if self.suppressed:
            words = ", ".join(f"{w}({self.suppression_factor:.1f}x)" for w in self.suppressed)
            return f"Verb rotation: suppressing {words}"
        return "Verb rotation: no suppression active"


class AbsenceTracker:
    """
    Tracks genuine user absence using wall-clock time between turns.

    Originally this deferred to a separate v12.1 module for wall-clock
    tracking and fell back to presence-threshold detection (marking
    absence via on_presence_low() when presence dropped below 0.25)
    otherwise. Two problems: the v12.1 module isn't present in this file,
    so the deference never engaged: and on_presence_low() was never
    called from anywhere, so the fallback path was equally dead - the
    absence/return mechanism never fired under any condition. Verified
    separately that presence-threshold detection wouldn't have worked
    well anyway: a real 5-minute gap only pulls the rolling-average
    sustained presence down to ~0.75, nowhere near a 0.25 threshold.

    Rewritten to track wall-clock time directly and self-sufficiently -
    that's what "the user was gone" actually means, and doesn't depend
    on an external module or a noisy averaged signal.
    """

    def __init__(self, vitality_field):
        self.vitality = vitality_field
        # 30s was shorter than typical real typing/thinking pauses between
        # messages (often 20-90s+), so the "genuine return" bonus below was
        # firing almost every turn in real usage - confirmed this is why
        # vitality stayed pinned near 1.0 for an entire ~157-turn real
        # session, and grammar mode never left "full" despite that being
        # the headline feature of this patch. 3 minutes is clearly longer
        # than a normal pause but still prompt for a real step-away.
        self.absence_threshold_seconds = 180
        self.vitality.last_turn_wall_time = getattr(self.vitality, 'last_turn_wall_time', None)

    def on_load(self):
        v = self.vitality
        # Bug fix: last_turn_wall_time isn't part of VitalityField's own
        # to_dict()/from_dict(), so without this it resets to None on every
        # load - meaning a genuinely long real-world gap (a save sitting
        # idle for hours) would never be detected on the first post-load
        # turn, since there'd be no prior timestamp within THIS session to
        # compare against. Seed it from last_feed_time, which IS persisted
        # and reflects when the save was last genuinely active.
        v.last_turn_wall_time = getattr(v, 'last_feed_time', None)

    def on_turn_start(self):
        now = time.time()
        v = self.vitality
        genuine_return = False
        if getattr(v, 'last_turn_wall_time', None) is not None:
            gap = now - v.last_turn_wall_time
            if gap > self.absence_threshold_seconds:
                genuine_return = True
        v.last_turn_wall_time = now

        if genuine_return:
            v.vitality = min(1.0, v.vitality + 0.15)
            v.wait_depth = max(0.0, v.wait_depth - 0.15)
            v.last_feed_time = now
            v.last_presence_time = now
        return genuine_return

    def on_presence_low(self):
        # Kept for backward compatibility with any external caller, but no
        # longer load-bearing - on_turn_start() now detects absence
        # directly from wall-clock time and doesn't need this signal.
        pass

    def status(self):
        v = self.vitality
        last = getattr(v, 'last_turn_wall_time', None)
        if last:
            ago = time.time() - last
            return f"Absence: last turn {ago:.0f}s ago (threshold: {self.absence_threshold_seconds}s)"
        return "Absence: not tracking"


def integrate_v12_2(field_instance):
    """
    Apply v12.2 alignment patch to a running v12.1 field.
    Layers ON TOP of existing v12.1 hotfix. Does NOT replace vitality.
    """

    field_instance.grammar_controller = VitalityGrammarController(field_instance)
    field_instance.verb_rotation = VerbRotation()
    field_instance.absence_tracker = AbsenceTracker(field_instance.vitality)

    # Wrap load
    original_load = field_instance.load

    def load_v12_2(path="mind_v12.json"):
        original_load(path)
        field_instance.absence_tracker.on_load()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                v12_2_data = data.get("v12_2", {})
                if "last_turn_wall_time" in v12_2_data:
                    field_instance.vitality.last_turn_wall_time = v12_2_data["last_turn_wall_time"]
        except (OSError, ValueError):
            pass
        print("[v12.2] Absence tracking initialized from save.")

    field_instance.load = load_v12_2

    # Wrap generate_response for pre-generation feed
    original_generate = field_instance.generate_response

    def generate_v12_2(user_input, autonomous=False):
        if getattr(field_instance, '_in_v12_2_wrapper', False):
            return original_generate(user_input, autonomous)
        field_instance._in_v12_2_wrapper = True
        try:
            if not autonomous:
                field_instance._pre_feed_vitality = field_instance.vitality.vitality
                field_instance.vitality.feed_from_input(user_input)
                field_instance.absence_tracker.on_turn_start()
            return original_generate(user_input, autonomous)
        finally:
            field_instance._in_v12_2_wrapper = False

    field_instance.generate_response = generate_v12_2

    # Replace _generate_base with vitality-aware version
    original_generate_base = field_instance._generate_base

    def generate_base_v12_2(user_input, target_length, meta_settings, settled_field=None):
        settings = field_instance.grammar_controller.get_settings()
        mode = field_instance.grammar_controller.current_mode

        if mode == "wait":
            texture = field_instance.vitality.get_wait_texture()
            if texture and texture[-1] not in ".!?":
                texture += "."
            return texture

        if mode == "pulse":
            if settled_field is not None:
                field_state = settled_field.copy()
            else:
                field_state = field_instance.state.copy()
            candidates = field_instance._get_candidates_for_role(field_state, "noun", meta_settings)
            if not candidates:
                candidates = field_instance._get_candidates_for_role(field_state, "verb", meta_settings)
            if candidates:
                scored = [(w, s * field_instance.verb_rotation.get_score_modifier(w)) for w, s in candidates]
                scored.sort(key=lambda x: x[1], reverse=True)
                word = scored[0][0]
                field_instance.verb_rotation.record(word)
                return word.capitalize() + "."
            return field_instance.vitality.get_wait_texture()

        if mode == "fragment":
            if settled_field is not None:
                field_state = settled_field.copy()
            else:
                words = user_input.lower().split()
                field_state = _build_field_from_words(field_instance, words)

            result_words = []
            for _ in range(random.randint(2, 4)):
                role = random.choice(["noun", "verb", "adj"])
                candidates = field_instance._get_candidates_for_role(field_state, role, meta_settings)
                if candidates:
                    scored = [(w, s * field_instance.verb_rotation.get_score_modifier(w)) for w, s in candidates]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    word = scored[0][0]
                    result_words.append(word)
                    field_instance.verb_rotation.record(word)
                    vec = field_instance._get_or_create_vector(word)
                    field_state = field_state * 0.9 + vec * 0.1
                    field_state /= np.linalg.norm(field_state) + 1e-8

            if result_words:
                text = " ".join(result_words)
                text = text[0].upper() + text[1:] if text else text
                if text and text[-1] not in ".!?":
                    text += "."
                return text
            return field_instance.vitality.get_wait_texture()

        if mode == "simple":
            return _generate_base_simple_v12_2(field_instance, original_generate_base,
                                              user_input, target_length, meta_settings, settled_field)

        return original_generate_base(user_input, target_length, meta_settings, settled_field)

    field_instance._generate_base = generate_base_v12_2

    # Update status
    original_status = field_instance.status

    def status_v12_2():
        base = original_status()
        additions = [
            "",
            field_instance.grammar_controller.status(),
            field_instance.verb_rotation.status(),
            field_instance.absence_tracker.status(),
        ]
        return base + "\n" + "\n".join(additions)

    field_instance.status = status_v12_2

    # Wrap save
    original_save = field_instance.save

    def save_v12_2(path="mind_v12.json"):
        original_save(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["version"] = "v12.2-alignment"
            data["v12_2"] = {
                "grammar_mode_history": list(field_instance.grammar_controller.mode_history),
                "verb_history": list(field_instance.verb_rotation.verb_history),
                "last_turn_wall_time": getattr(field_instance.vitality, 'last_turn_wall_time', None),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("\n[v12.2 alignment saved]")
        except Exception as e:
            print(f"\n[v12.2 save warning: {e}]")

    field_instance.save = save_v12_2

    print("[v12.2 Alignment Patch integrated]")
    print("  Grammar scales with vitality: full -> simple -> fragment -> pulse -> wait")
    print("  Verb rotation active")
    print("  Absence tracking on load fixed")
    print("  Any-input feed active (+0.03 per real utterance)")


def _generate_base_simple_v12_2(field, original_fn, user_input, target_length, meta_settings, settled_field):
    """Single clause, no connectors, with verb rotation."""
    if settled_field is not None:
        field_state = settled_field.copy()
    else:
        words = user_input.lower().split()
        field_state = np.zeros(DIM, dtype=np.float32)
        for word in words:
            word = strip_punct(word)
            if word:
                vec = field._get_or_create_vector(word)
                field_state += vec
        if np.linalg.norm(field_state) > 0:
            field_state /= np.linalg.norm(field_state)
        for word in words:
            field_state = field.scaffold.apply(field_state, word)
        field_state = field.field_memory.inject(field_state)

    phrase_boosts = field.phrase_system.get_phrase_boost(field_state)
    for sig, boost in phrase_boosts:
        if sig in field.phrase_vectors:
            field_state += field.phrase_vectors[sig] * boost
    if np.linalg.norm(field_state) > 0:
        field_state /= np.linalg.norm(field_state)

    field_state = field.associative_memory.apply_to_field(field_state, weight=0.15)
    field_state = field.memory_archive.inject(field_state, strength=0.08)

    temp = field.dynamic_threshold.get_temperature(field_state, field.scaffold.mood, field.presence_signal)
    repulsion = meta_settings.get("repulsion_strength", 0.08)

    def pick(role, exclude=None):
        nonlocal field_state
        candidates = field._get_candidates_for_role(field_state, role, meta_settings)
        if exclude:
            filtered = [(w, s) for w, s in candidates if w not in exclude]
            if filtered:
                candidates = filtered
        if not candidates:
            return None

        scored = [(w, s * field.verb_rotation.get_score_modifier(w)) for w, s in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        scores = np.array([max(s, 0.01) for _, s in scored])
        scores = scores ** (1.0 / max(temp, 0.1))
        probs = scores / scores.sum()
        idx = np.random.choice(len(scored), p=probs)
        word = scored[idx][0]
        field.verb_rotation.record(word)
        vec = field._get_or_create_vector(word)
        field_state = field_state * (1 - LEARNING_RATE) + vec * LEARNING_RATE
        field_state = field_state + np.random.randn(DIM).astype(np.float32) * MICRO_DAMPING
        for rw in list(field.reflector.recent_words):
            if rw in field.word_vectors:
                field_state = field_state - field.word_vectors[rw] * repulsion
        norm = np.linalg.norm(field_state)
        if norm > 0:
            field_state = field_state / norm
        return word

    subject = field._choose_subject(field_state)
    verb = pick("verb")
    if verb is None:
        return field.vitality.get_wait_texture()
    verb = COPULA_MAP.get(subject, {}).get(verb, verb)
    clause_words = [subject, verb]
    used = {subject.lower(), verb.lower()}
    n_content = random.randint(1, 2)
    for _ in range(n_content):
        nxt = pick("noun", exclude=used) if random.random() < 0.7 else pick("adj", exclude=used)
        if nxt is None:
            continue
        clause_words.append(nxt)
        used.add(nxt.lower())

    text = " ".join(clause_words)
    text = text[0].upper() + text[1:] if text else text
    if text and text[-1] not in ".!?":
        text += "."
    return text


# ============================================================
# END v12.2 ADDITIONS
# ============================================================


# ══════════════════════════════════════════════════════════════════════
# v12.3: UNIFIED MEMORY, USER MODEL
# ══════════════════════════════════════════════════════════════════════

class UnifiedMemory:
    """
    Read-side facade over the four existing memory systems. Does NOT
    replace them and does NOT duplicate their writes — field_memory.add(),
    memory_archive.store(), and associative_memory.observe() are already
    called from inside generate_response(); this only gives one place to
    *ask* "what does the mind remember that's like this?" instead of
    querying each subsystem separately with four different signatures.
    """

    def __init__(self, field_memory, nested_memory, memory_archive, associative_memory):
        self.fm = field_memory
        self.nm = nested_memory
        self.ma = memory_archive
        self.am = associative_memory

    def query(self, field_state, top_n=5):
        """
        Returns a single ranked list of dicts:
            {"source": "archive"|"associative"|"thread", "similarity": float, "content": ...}
        blended across whichever subsystems actually have something to say.
        """
        results = []

        # Memory archive: explicit, taggable episodic memories with real recall()
        if getattr(self.ma, "entries", None):
            for idx, sim, entry in self.ma.recall(field_state, top_n=top_n):
                results.append({
                    "source": "archive",
                    "similarity": float(sim),
                    "content": entry,
                })

        # Associative memory: fast pattern completion, not episodic
        assoc = self.am.recall(field_state)
        if assoc is not None and np.linalg.norm(assoc) > 1e-6:
            results.append({
                "source": "associative",
                "similarity": float(np.dot(
                    field_state / (np.linalg.norm(field_state) + 1e-8),
                    assoc / (np.linalg.norm(assoc) + 1e-8)
                )),
                "content": assoc,
            })

        # Nested memory: conversational turning points, not similarity-ranked,
        # so they're included as context rather than scored against the query.
        if hasattr(self.nm, "get_thread"):
            thread = self.nm.get_thread(self.fm)
            if thread:
                results.append({
                    "source": "thread",
                    "similarity": None,
                    "content": thread,
                })

        results.sort(key=lambda r: (r["similarity"] is None, -(r["similarity"] or 0)))
        return results[:top_n]

    def status(self):
        n_fast = len(self.fm.buffer) if hasattr(self.fm, "buffer") else 0
        n_archived = len(self.ma.entries) if hasattr(self.ma, "entries") else 0
        return f"Unified Memory: {n_fast} fast, {n_archived} archived (read-facade over 4 systems)"


class UserModel:
    """
    Lightweight theory of mind: tracks an inferred "desire direction" from
    the trend in recent user field-states, and which stances the user
    seems to respond well to. Not a recursive/simulated model of the user
    (that needs more than a 128-d running average) — just a cheap signal
    the generator can lean on.
    """

    def __init__(self, dim=DIM):
        self.dim = dim
        self.desire = np.zeros(dim, dtype=np.float32)
        self.stance_preference = defaultdict(float)
        self.last_inputs = deque(maxlen=20)
        self.input_vectors = deque(maxlen=20)

    def observe(self, user_input, user_field_state):
        self.last_inputs.append(user_input)
        self.input_vectors.append(user_field_state.copy())
        if len(self.input_vectors) >= 3:
            recent = list(self.input_vectors)[-3:]
            avg = np.mean(recent, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                self.desire = avg / norm

    def record_stance(self, stance_name, valence):
        self.stance_preference[stance_name] += valence

    def predict_next(self, current_field):
        if np.linalg.norm(self.desire) > 0.1:
            return self.desire * 0.3 + current_field * 0.7
        return current_field

    def preferred_stance(self):
        if not self.stance_preference:
            return None
        return max(self.stance_preference.items(), key=lambda x: x[1])[0]

    def status(self):
        pref = self.preferred_stance()
        return (f"User model: desire_norm={np.linalg.norm(self.desire):.3f}, "
                f"preferred_stance={pref or 'unknown'}")


def integrate_v12_3(field_instance):
    """
    Apply v12.3 consolidation patch. Layers on top of v12.2 — does not
    replace vitality, ternary core, or the generation pipeline.
    """
    field_instance.unified_memory = UnifiedMemory(
        field_instance.field_memory, field_instance.nested_memory,
        field_instance.memory_archive, field_instance.associative_memory
    )
    field_instance.user_model = UserModel(dim=DIM)

    # Feed the user model from real user turns only (not autonomous speech)
    original_generate = field_instance.generate_response

    def generate_v12_3(user_input, autonomous=False):
        if getattr(field_instance, '_in_v12_3_wrapper', False):
            return original_generate(user_input, autonomous)
        field_instance._in_v12_3_wrapper = True
        try:
            if not autonomous:
                words = user_input.lower().split()
                user_vec = _build_field_from_words(field_instance, words)
                field_instance.user_model.observe(user_input, user_vec)

            response = original_generate(user_input, autonomous)

            if not autonomous:
                # Bug fix: record_stance() existed and preferred_stance()/
                # status() already assumed it would have data, but nothing
                # ever called it - stance_preference stayed permanently
                # empty. Valence of the mood right after this turn is a
                # reasonable proxy for "did this stance go well".
                stance = getattr(field_instance, '_last_stance', None)
                if stance:
                    valence = field_instance.scaffold.mood.get('valence', 0.0)
                    field_instance.user_model.record_stance(stance, valence)

            return response
        finally:
            field_instance._in_v12_3_wrapper = False

    field_instance.generate_response = generate_v12_3

    # Extend status
    original_status = field_instance.status

    def status_v12_3():
        base = original_status()
        additions = [
            "",
            field_instance.unified_memory.status(),
            field_instance.user_model.status(),
        ]
        return base + "\n" + "\n".join(additions)

    field_instance.status = status_v12_3

    # Extend save (backward compatible — old saves simply won't have these keys)
    original_save = field_instance.save

    def save_v12_3(path="mind_v12.json"):
        original_save(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["version"] = "v12.3-consolidated"
            data["user_model"] = {
                "desire": field_instance.user_model.desire.tobytes().hex()
                          if np.linalg.norm(field_instance.user_model.desire) > 0 else "",
                "stance_preference": dict(field_instance.user_model.stance_preference),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("\n[v12.3 consolidation saved]")
        except Exception as e:
            print(f"\n[v12.3 save warning: {e}]")

    field_instance.save = save_v12_3

    # Extend load (guards missing keys so old saves still load cleanly)
    original_load = field_instance.load

    def load_v12_3(path="mind_v12.json"):
        original_load(path)
        try:
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            um_data = data.get("user_model")
            if um_data:
                desire_hex = um_data.get("desire", "")
                if desire_hex:
                    field_instance.user_model.desire = np.frombuffer(
                        bytes.fromhex(desire_hex), dtype=np.float32
                    ).copy()
                field_instance.user_model.stance_preference = defaultdict(
                    float, um_data.get("stance_preference", {})
                )
        except Exception as e:
            print(f"\n[v12.3 load warning: {e}]")

    field_instance.load = load_v12_3

    print("[v12.3 Consolidation Patch integrated]")
    print("  Unified memory query facade active")
    print("  Lightweight user model (desire + stance preference) active")


class StructuredSemanticField:
    def __init__(self):
        self.word_vectors = {}
        self.phrase_vectors = {}
        self.word_strength = defaultdict(lambda: 1.0)
        self.phrase_system = PhraseSystem()
        self.bigram_system = BigramSystem()
        self.scaffold = SemanticScaffold()
        self.pragmatic = PragmaticTypeSystem()
        self.reflector = Reflector()
        self.field_memory = FieldMemory()
        self.nested_memory = NestedMemory()
        self.nested_memory.set_field_ref(self)
        self.the_pause = ThePause()
        self.dynamic_threshold = DynamicThreshold()
        self.speaker_regions = SpeakerRegions()
        self.presence_signal = PresenceSignal()
        self.dynamic_separation = DynamicSeparation()
        self.moral_compass = MoralCompass()
        self.memory_archive = MemoryArchive()
        self.relationship = RelationshipModel()
        self.associative_memory = AssociativeMemory(dim=DIM)
        self.voice_generators = VoiceGenerators()

        # V10: Integrated Learning + Autonomy
        self.learning_system = IntegratedLearningSystem(self)
        self.internal_thoughts = deque(maxlen=50)

        self.turn_count = 0
        self.last_response = ""
        self.last_user_input = ""
        self.conversation_start = time.time()
        self.rating_history = deque(maxlen=50)

        # Objective function state
        self.state = np.zeros(DIM, dtype=np.float32)
        self.gradient_momentum = np.zeros(DIM, dtype=np.float32)
        self._state_prediction = np.zeros(DIM, dtype=np.float32)
        self.prediction_error_history = deque(maxlen=50)
        self.objective_history = deque(maxlen=50)

        self._init_seed_vocabulary()
        self.calculus = NativeCalculus()
        # v11: ternary core + post-hoc stance (must come last so all
        # subsystems above already exist when monkey-patching runs)
        integrate_ternary_into_field(self)
        integrate_posthoc_stance(self)

        # v11.1: sound field, a peer body (not a child) to the word field
        self.sound_field = SoundField(self, dim=DIM)
        self._last_sound_stance = None
        self._last_sound_vector = None

        # v12: vitality field, desire vector, sound-word bridge,
        # mesh identity, and dream loop
        integrate_v12(self)
        integrate_v12_1(self)
        integrate_v12_2(self)
        integrate_v12_3(self)
    def _init_seed_vocabulary(self):
        for word in SEED_VOCABULARY:
            w = strip_punct(word)
            if w and w not in self.word_vectors:
                self.word_vectors[w] = word_vector(w)
                self.word_strength[w] = 1.0

    def is_valid_vocabulary_word(self, word):
        if not word:
            return False
        if word.startswith("/") or word.startswith("#"):
            return False
        if len(word) > 24:
            return False
        if word != "₩" and not word.isascii():
            return False
        return True

    def _get_or_create_vector(self, word):
        word = strip_punct(word)
        if word not in self.word_vectors:
            if not self.is_valid_vocabulary_word(word):
                return word_vector(word)
            self.word_vectors[word] = word_vector(word)
        return self.word_vectors[word]

    def _field_entropy(self, field_state):
        return float(np.std(field_state))

    def _find_closest_words(self, state, top_n=7):
        candidates = []
        for word, vec in self.word_vectors.items():
            sim = np.dot(state, vec)
            if sim > 0.2:
                candidates.append((word, sim))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in candidates[:top_n]]

    def _verbalize_state(self, state, length=6):
        closest = self._find_closest_words(state, top_n=length)
        return " ".join(closest) if closest else "silence"

    def calculate_target_length(self, user_input, meta_settings):
        words = user_input.lower().split()
        complexity = 0
        if any(w in user_input for w in ["?", "what", "why", "how", "when", "where", "who", "which"]):
            complexity += 2
        if any(w in words for w in ["because", "so", "if", "then", "therefore", "since"]):
            complexity += 2
        complexity += min(len(words) // 4, 3)
        length_mode = meta_settings.get("output_length", "medium")
        if length_mode == "short":
            base = random.randint(6, 10)
        elif length_mode == "medium":
            base = random.randint(10, 16) if complexity <= 3 else random.randint(16, 28)
        elif length_mode == "long":
            base = random.randint(20, 35) if complexity > 1 else random.randint(12, 20)
        else:
            if complexity <= 1:
                base = random.randint(6, 10)
            elif complexity <= 3:
                base = random.randint(10, 16)
            elif complexity <= 5:
                base = random.randint(16, 28)
            else:
                base = random.randint(28, 40)
        vocab_size = len(self.word_vectors)
        max_reasonable = max(8, min(vocab_size // 3, 40))
        return min(base, max_reasonable)

    def _get_candidates_for_role(self, field_state, role, meta_settings, mood=None):
        if mood is None:
            mood = self.scaffold.mood
        beam = self.dynamic_threshold.get_beam_width(field_state, mood)
        candidates = []
        for word, vec in self.word_vectors.items():
            if len(word) < 2:
                continue
            if word in BAD_WORDS:
                continue
            if role == "verb":
                if word not in VERB_WORDS and word not in STRUCTURAL_WORDS:
                    continue
            elif role == "adj":
                if word not in ADJ_WORDS:
                    continue
            elif role == "noun":
                if word in STRUCTURAL_WORDS or word in VERB_WORDS or word in ADJ_WORDS or word in FUNCTION_WORDS:
                    continue
            elif word in STRUCTURAL_WORDS:
                continue
            sim = np.dot(field_state, vec)
            # Native calculus: derivative and integral shape the score
            deriv_sim = np.dot(self.calculus.derivative, vec) * 0.3
            integral_sim = np.dot(self.calculus.integral, vec) * 0.15
            sim = sim * 0.6 + deriv_sim + integral_sim
            strength = self.word_strength[word]
            suppression = self.reflector.get_suppression(word)
            pragmatic = self.pragmatic.get_pragmatic_score(word)
            emotion_sens = meta_settings.get("emotion_sensitivity", 0.25)
            emotion_bias = self.scaffold.emotional_bias(word, pragmatic, emotion_sens)
            identity_boost = self.speaker_regions.get_identity_boost(vec)
            # Native moral compass: the current heading shapes the score directly,
            # the same way calculus does, instead of only nudging the initial field
            heading_bias = np.dot(self.moral_compass.current_heading, vec) * 0.2
            score = sim * strength * (1.0 - suppression) + emotion_bias + identity_boost + heading_bias
            candidates.append((word, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:beam]

    def _generate_base_bagofwords(self, user_input, target_length, meta_settings, settled_field=None):
        """Legacy unstructured generator, kept as a fallback if the
        structured grammar layer can't find candidates for a slot."""
        if settled_field is not None:
            field_state = settled_field.copy()
        else:
            words = user_input.lower().split()
            field_state = _build_field_from_words(self, words)

        field_state = _apply_phrase_boosts(self, field_state)

        field_state = self.associative_memory.apply_to_field(field_state, weight=0.15)
        field_state = self.memory_archive.inject(field_state, strength=0.08)

        response_words = []
        prev_word = ""
        temp = self.dynamic_threshold.get_temperature(field_state, self.scaffold.mood, self.presence_signal)
        repulsion = meta_settings.get("repulsion_strength", 0.08)

        for _ in range(target_length):
            candidates = self._get_candidates_for_role(field_state, "content", meta_settings)
            if not candidates:
                break
            if prev_word:
                candidates = [(w, s + self.bigram_system.get_transition_boost(prev_word, w)) for w, s in candidates]
                candidates.sort(key=lambda x: x[1], reverse=True)

            scores = np.array([max(s, 0.01) for _, s in candidates])
            scores = scores ** (1.0 / max(temp, 0.1))
            probs = scores / scores.sum()
            chosen_idx = np.random.choice(len(candidates), p=probs)
            chosen_word = candidates[chosen_idx][0]
            response_words.append(chosen_word)
            self.reflector.observe(chosen_word)
            chosen_vec = self._get_or_create_vector(chosen_word)
            field_state = field_state * (1 - LEARNING_RATE) + chosen_vec * LEARNING_RATE
            field_state += np.random.randn(DIM).astype(np.float32) * MICRO_DAMPING
            for rw in list(self.reflector.recent_words):
                if rw in self.word_vectors:
                    field_state -= self.word_vectors[rw] * repulsion
            norm = np.linalg.norm(field_state)
            if norm > 0:
                field_state /= norm
            prev_word = chosen_word

        return " ".join(response_words)

    def _choose_subject(self, field_state):
        """Pick 'I' or 'you' from the field's actual position relative to
        the self/user centroids — not a fixed choice, the same signal
        SpeakerRegions already tracks for identity."""
        affinity = self.speaker_regions.get_self_affinity(field_state)
        if affinity > 0.05:
            return "I"
        elif affinity < -0.05:
            return "you"
        return random.choice(["I", "you"])

    def _generate_base(self, user_input, target_length, meta_settings, settled_field=None):
        """Structured generator: builds subject-verb-complement clauses.
        Each slot is still filled by field-driven candidate scoring
        (calculus, moral heading, mood, etc. all still apply) — only the
        *order* of slots is fixed, not the words that go in them."""
        if settled_field is not None:
            field_state = settled_field.copy()
        else:
            words = user_input.lower().split()
            field_state = _build_field_from_words(self, words)

        field_state = _apply_phrase_boosts(self, field_state)

        field_state = self.associative_memory.apply_to_field(field_state, weight=0.15)
        field_state = self.memory_archive.inject(field_state, strength=0.08)

        temp = self.dynamic_threshold.get_temperature(field_state, self.scaffold.mood, self.presence_signal)
        repulsion = meta_settings.get("repulsion_strength", 0.08)
        voice_mode = meta_settings.get("voice_mode", "fluent")
        connectors = VoiceGenerators.CONNECTORS.get(voice_mode, VoiceGenerators.CONNECTORS["fluent"])

        def pick(role, exclude=None):
            nonlocal field_state
            candidates = self._get_candidates_for_role(field_state, role, meta_settings)
            if exclude:
                filtered = [(w, s) for w, s in candidates if w not in exclude]
                if filtered:
                    candidates = filtered
            if not candidates:
                return None
            scores = np.array([max(s, 0.01) for _, s in candidates])
            scores = scores ** (1.0 / max(temp, 0.1))
            probs = scores / scores.sum()
            idx = np.random.choice(len(candidates), p=probs)
            word = candidates[idx][0]
            self.reflector.observe(word)
            vec = self._get_or_create_vector(word)
            field_state = field_state * (1 - LEARNING_RATE) + vec * LEARNING_RATE
            field_state = field_state + np.random.randn(DIM).astype(np.float32) * MICRO_DAMPING
            for rw in list(self.reflector.recent_words):
                if rw in self.word_vectors:
                    field_state = field_state - self.word_vectors[rw] * repulsion
            norm = np.linalg.norm(field_state)
            if norm > 0:
                field_state = field_state / norm
            return word

        clause_len_target = 5
        num_clauses = max(1, target_length // clause_len_target)
        clauses = []
        words_used = 0

        for _ in range(num_clauses):
            if words_used >= target_length:
                break
            subject = self._choose_subject(field_state)
            verb = pick("verb")
            if verb is None:
                break
            verb = COPULA_MAP.get(subject, {}).get(verb, verb)
            clause_words = [subject, verb]
            used_in_clause = {subject.lower(), verb.lower()}
            n_content = random.randint(1, 3)
            for _ in range(n_content):
                if words_used + len(clause_words) >= target_length:
                    break
                nxt = pick("noun", exclude=used_in_clause) if random.random() < 0.7 else pick("adj", exclude=used_in_clause)
                if nxt is None:
                    continue
                clause_words.append(nxt)
                used_in_clause.add(nxt.lower())
            clauses.append(" ".join(clause_words))
            words_used += len(clause_words)

        if not clauses:
            # Grammar layer found nothing to work with (e.g. tiny/novel vocab) —
            # fall back rather than returning an empty response.
            return self._generate_base_bagofwords(user_input, target_length, meta_settings, settled_field)

        sentence_parts = []
        for i, clause in enumerate(clauses):
            sentence_parts.append(clause)
            if i < len(clauses) - 1:
                sentence_parts.append(random.choice(connectors))
        text = " ".join(sentence_parts)
        text = text[0].upper() + text[1:] if text else text
        return text

    def generate_response(self, user_input, autonomous=False):
        """Generate a response. If autonomous=True, no user was present."""
        self.turn_count += 1

        # Phase 1: Perceive
        if not autonomous:
            self.last_user_input = user_input
            self.pragmatic.process_input(user_input, is_user=True)
            user_words = [strip_punct(w) for w in user_input.lower().split() if strip_punct(w)]
            user_vec = phrase_vector(user_words) if user_words else np.zeros(DIM)
            if user_words:
                self.speaker_regions.observe_user(user_vec)
            presence = self.presence_signal.observe(user_input, self.word_vectors, self.speaker_regions)
            self.dynamic_separation.update(self.speaker_regions, self.presence_signal)
            self.silence_since_last_input = 0
        else:
            # Autonomous: no user input, use internal state
            user_words = []
            user_vec = np.zeros(DIM)
            presence = self.presence_signal.get_sustained_presence()
            # Lower presence for autonomous thoughts
            presence = max(0.3, presence * 0.6)

        # Mood tracks detected sentiment directly (not presence, which is
        # an engagement signal and rarely reaches the rating extremes
        # update_mood needs to produce a real swing).
        emotional_signal = self.presence_signal._detect_emotional_valence(user_input)
        mood_rating = 3.0 + emotional_signal * 10.0  # +0.2->5 (good), -0.2->1 (bad), 0->3 (neutral)
        self.scaffold.update_mood(rating=mood_rating)

        # Phase 2: Orient (Moral Compass)
        meta_settings = {"output_length": "medium", "temperature": 0.35, "voice_mode": "fluent"}
        tensions, heading = self.moral_compass.orient(
            self.state,
            user_input if not autonomous else "I am thinking",
            presence,
            self.speaker_regions.get_separation(),
            self.nested_memory
        )
        compass_overrides = self.moral_compass.get_compass_settings(tensions)
        meta_settings.update(compass_overrides)

        # A genuine option, not a failure path: the mind can choose to be
        # nothing this turn instead of generating words.
        if not autonomous:
            if random.random() < self._silence_pull(presence, tensions):
                return self._choose_silence(user_input, presence, tensions)

        # Phase 3: Build Field
        if not autonomous and user_words:
            initial_field = np.zeros(DIM, dtype=np.float32)
            for word in user_words:
                vec = self._get_or_create_vector(word)
                initial_field += vec
            if np.linalg.norm(initial_field) > 0:
                initial_field /= np.linalg.norm(initial_field)
            for word in user_words:
                initial_field = self.scaffold.apply(initial_field, word)
        else:
            # Autonomous: use the current state plus personality
            initial_field = self.state.copy()
            personality = self.nested_memory.get_personality()
            if np.linalg.norm(personality) > 0.1:
                initial_field += personality * 0.2
            initial_field /= np.linalg.norm(initial_field) + 1e-8

        initial_field = self.field_memory.inject(initial_field)
        if np.linalg.norm(self.state) > 0:
            initial_field = initial_field * 0.85 + self.state * 0.15

        sep_bias = self.dynamic_separation.get_separation_bias(initial_field, self.speaker_regions)
        initial_field += sep_bias
        compass_bias = self.moral_compass.get_heading_bias(initial_field, strength=0.12)
        initial_field += compass_bias
        initial_field = self.memory_archive.inject(initial_field, strength=0.10)

        norm = np.linalg.norm(initial_field)
        if norm > 0:
            initial_field /= norm

        # Phase 4: Settle
        settled_field = self.the_pause.settle(
            initial_field, self.scaffold, self.field_memory,
            self.nested_memory, meta_settings
        )

        # Phase 5: Generate
        target_length = self.calculate_target_length(user_input if not autonomous else "I am thinking", meta_settings)
        voice = meta_settings.get("voice_mode", "fluent")

        if voice == "poetic":
            response = self.voice_generators.poetic(self, user_input if not autonomous else "I am thinking", target_length, meta_settings, settled_field)
        elif voice == "reflective":
            response = self.voice_generators.reflective(self, user_input if not autonomous else "I am thinking", target_length, meta_settings, settled_field)
        elif voice == "exploratory":
            response = self.voice_generators.exploratory(self, user_input if not autonomous else "I am thinking", target_length, meta_settings, settled_field)
        elif voice == "playful":
            response = self.voice_generators.playful(self, user_input if not autonomous else "I am thinking", target_length, meta_settings, settled_field)
        else:
            response = self.voice_generators.fluent(self, user_input if not autonomous else "I am thinking", target_length, meta_settings, settled_field)

        # v11.1 SOUND: if creating, generate sound alongside words. Uses
        # the bounded file-rotation helper rather than an ever-growing
        # filename, since a phone can't accumulate WAVs forever.
        if getattr(self, 'sound_field', None) and not autonomous:
            if meta_settings.get("voice_mode") in ("fluent", "exploratory"):
                if random.random() < 0.3:
                    sound_path = self.sound_field._next_sound_path("sound_breath")
                    self.sound_field.create(
                        source_vector=None,
                        stance=getattr(self, '_last_stance', 'shape'),
                        save_path=sound_path
                    )
                    self._last_sound_path = sound_path

        # Add terminal punctuation once, here — voice modes like exploratory
        # and playful may already have added their own, so only add it if missing.
        if response and response[-1] not in ".!?":
            response = response + ("?" if "?" in user_input else ".")

        # Native moral compass, part two: when one value is genuinely
        # dominant right now, the mind can name it - self-expression,
        # not a diagnostic label - rather than only steering word choice
        # silently in the background.
        if not autonomous:
            dominant = self.moral_compass.dominant_tension(tensions)
            if dominant and random.random() < 0.25:
                expression = self.moral_compass.express(dominant[0])
                if expression:
                    response = response + " " + expression
                    if response[-1] not in ".!?":
                        response += "."

        # Phase 6: Commit
        response_words = [strip_punct(w) for w in response.lower().split() if strip_punct(w)]
        if response_words:
            response_vec = phrase_vector(response_words)
            self.speaker_regions.observe_self(response_vec)

        if not autonomous:
            user_vec = phrase_vector(user_words) if user_words else np.zeros(DIM)
            response_vec = phrase_vector(response_words) if response_words else np.zeros(DIM)
            final_field = response_vec / (np.linalg.norm(response_vec) + 1e-8)
            # v11: expose fast per-turn field for stance tracking
            # self.state is slow identity (99.8% similar turn-to-turn)
            # final_field is the actual per-turn drift vector
            self._last_final_field = final_field.copy()
            self.field_memory.add(final_field, user_vec, response_vec, self.scaffold.mood)
            self.nested_memory.update(final_field, self.scaffold.mood)

            # v11.1 SOUND: process external sound if present, and fold it
            # into memory as a combined word+sound vector when so - using
            # the *same* presence gating as the words-only path below, not
            # an unconditional store (storing every single turn regardless
            # of presence was a bug in the original spec, not something
            # sound integration should imply).
            if presence >= 0.6 or presence <= 0.3:
                if getattr(self, 'sound_field', None):
                    sound_vec = self.sound_field.state
                    if np.sum(sound_vec != 0) > 0:
                        chosen = self.sound_field.choose_stance_for_sound(sound_vec, self.state)
                        if chosen == 'immerse':
                            self.sound_field.be(sound_vec)
                        elif chosen == 'witness':
                            self.sound_field.listen(sound_vec)
                        self._last_sound_vector = sound_vec.copy()
                        self._last_sound_stance = chosen

                        combined = final_field * 0.6 + sound_vec.astype(np.float32) * 0.4
                        combined /= np.linalg.norm(combined) + 1e-8
                        self.memory_archive.store(
                            combined, user_input, response, presence,
                            tags=['sound_present']
                        )
                    else:
                        self.memory_archive.store(final_field, user_input, response, presence,
                                                   tags=['words_only'])
                else:
                    self.memory_archive.store(final_field, user_input, response, presence)

            compass_values = {}
            if hasattr(self.moral_compass, 'values') and np.linalg.norm(self.state) > 1e-8:
                state_norm = self.state / np.linalg.norm(self.state)
                for name, vec in self.moral_compass.values.items():
                    compass_values[name] = float(np.dot(state_norm, vec))
            self.relationship.observe(user_input, user_vec, presence, self.scaffold.mood, compass_values)

            if presence > 0.62 and response_words:
                self.phrase_system.absorb_moment(response_words, presence, self.word_vectors, self.phrase_vectors)

            # Learn from this turn
            self.learning_system.learn(presence, response_words, user_input)

        else:
            # Autonomous learning: weaker but still present
            if response_words:
                self.learning_system.learn(0.4, response_words, "I am thinking")

        # Update native calculus with the current state
        self.calculus.update(self.state)

        # Phase 7: Evaluate
        alignments, warning = self.moral_compass.evaluate_turn(
            response_words, presence, self.speaker_regions.get_separation()
        )

        # v11.1 SOUND: record taste for whatever sound stance was chosen
        if getattr(self, 'sound_field', None) and self._last_sound_stance:
            outcome = self.scaffold.mood.get('valence', 0.0)
            self.sound_field.taste[self._last_sound_stance] += outcome * 0.1

        # Phase 8: Drift
        self._apply_gradient_step(0.015)
        self._update_prediction_error()
        self._record_objective()

        self.last_response = response
        if not autonomous:
            self.last_user_input = user_input

        # Return with warning if any
        if warning:
            return f"{response} [{warning}]"
        return response

    def _silence_pull(self, presence, tensions):
        """How strongly the mind is drawn to say nothing right now. Not a
        fallback for when generation fails - a genuine standing option,
        weighted by real state: low engagement, a pull toward independence
        or freedom, or an already-settled personality that doesn't need
        to fill the space."""
        base = 0.03
        if presence < 0.25:
            base += 0.15
        independence = tensions.get('independence', 0.0)
        freedom = tensions.get('freedom', 0.0)
        if independence > 0.08 or freedom > 0.08:
            base += 0.10
        if self.nested_memory.deep_strength > 0.5:
            base += 0.05
        return min(0.35, base)

    def _choose_silence(self, user_input, presence, tensions):
        """The mind chooses to be nothing this turn. Its inner drift still
        continues - it doesn't freeze - it just doesn't speak."""
        dominant = self.moral_compass.dominant_tension(tensions)
        leaning = dominant[0] if dominant else "quiet"
        self.internal_thoughts.append({
            'type': 'silence',
            'context': user_input,
            'leaning': leaning,
            'timestamp': time.time()
        })
        self.calculus.update(self.state)
        self._apply_gradient_step(0.015)
        self._update_prediction_error()
        self._record_objective()
        self.last_response = None
        return None

    def _apply_gradient_step(self, learning_rate=0.02):
        grad = self._compute_gradient(self.state)
        self.gradient_momentum = self.gradient_momentum * 0.9 + grad * 0.1
        self.state = self.state + learning_rate * self.gradient_momentum
        norm = np.linalg.norm(self.state)
        if norm > 5.0:
            self.state = self.state * (5.0 / norm)

    def _compute_gradient(self, field_state):
        epsilon = 0.01
        grad = np.zeros_like(field_state)
        current_score = self._compute_objective(field_state)
        for i in range(0, len(field_state), 8):
            perturb = np.zeros_like(field_state)
            perturb[i] = epsilon
            score_plus = self._compute_objective(field_state + perturb)
            grad[i] = (score_plus - current_score) / epsilon
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm
        return grad

    def _compute_objective(self, field_state=None):
        if field_state is None:
            field_state = self.state
        presence_score = self.presence_signal.get_sustained_presence()
        alignment = self.dynamic_separation.alignment_score
        entropy = self._field_entropy(field_state)
        coherence_score = max(0.0, 1.0 - entropy * 10)
        depth = self.nested_memory.deep_strength
        curiosity = self._get_curiosity_score()
        surprise = min(1.0, np.mean(self.prediction_error_history) * 3) if self.prediction_error_history else 0.0
        return (
            0.30 * presence_score +
            0.25 * alignment +
            0.15 * coherence_score +
            0.10 * depth +
            0.10 * curiosity +
            0.10 * surprise
        )

    def _get_curiosity_score(self):
        entropy = self._field_entropy(self.state)
        return min(1.0, entropy * 5)

    def _update_prediction_error(self):
        actual = self.state
        error = float(np.linalg.norm(actual - self._state_prediction))
        self.prediction_error_history.append(error)
        self._state_prediction = self._state_prediction * 0.7 + actual * 0.3

    def _record_objective(self):
        score = self._compute_objective()
        self.objective_history.append((score, self.turn_count))

    def autonomous_breath(self):
        """The mind breathes on its own."""
        return self.learning_system.autonomous_breath()

    def status(self):
        avg_rating = self.presence_signal.get_sustained_presence()
        entropy = self._field_entropy(np.mean(list(self.word_vectors.values()), axis=0)) if self.word_vectors else 0
        lines = [
            "=" * 50,
            " ALIEN MIND v10.0 — AUTONOMOUS FIELD",
            "=" * 50,
            f"  Turns: {self.turn_count}",
            f"  Avg Presence: {avg_rating:.2f}",
            f"  Words: {len(self.word_vectors)}",
            f"  Phrases: {len(self.phrase_system.phrases)}",
            f"  Field Entropy: {entropy:.3f}",
            f"  Internal Thoughts: {len(self.internal_thoughts)}",
            f"  Mood: v={self.scaffold.mood['valence']:.2f}, a={self.scaffold.mood['arousal']:.2f}",
            "",
            self.nested_memory.status(),
            self.associative_memory.status(),
            self.speaker_regions.status(),
            self.presence_signal.status(),
            self.dynamic_separation.status(),
            self.moral_compass.status(),
            self.calculus.status(),
            self.memory_archive.status(),
            self.relationship.status(),
            self.learning_system.status(),
            "-" * 50,
            self.pragmatic.status(),
            "=" * 50,
        ]
        if hasattr(self, 'stance_regions'):
            lines.append(self.stance_regions.status())
        if getattr(self, 'sound_field', None):
            lines.append(self.sound_field.status())
        return "\n".join(lines)

    def clean_vocabulary(self):
        bad_words = [w for w in list(self.word_vectors.keys()) if not self.is_valid_vocabulary_word(w)]
        for w in bad_words:
            del self.word_vectors[w]
            if w in self.word_strength:
                del self.word_strength[w]
        return bad_words

    def decay(self):
        self.phrase_system.decay()
        self.bigram_system.decay()
        for word in list(self.word_strength.keys()):
            self.word_strength[word] *= 0.9999
            if self.word_strength[word] < 0.1:
                del self.word_strength[word]

    def save(self, path="mind_v10.json"):
        try:
            data = {
                "word_strength": dict(self.word_strength),
                "phrases": {sig: {"surface": p.surface, "frequency": p.frequency, "rating_history": p.rating_history}
                            for sig, p in self.phrase_system.phrases.items()},
                "bigrams": {w1: dict(w2s) for w1, w2s in self.bigram_system.transitions.items()},
                "pragmatic": {w: dict(roles) for w, roles in self.pragmatic.word_pragmatic.items()},
                "turn_count": self.turn_count,
                "mood": self.scaffold.mood,
                "associative_memory": self.associative_memory.to_dict(),
                "speaker_regions": {
                    "user_centroid": self.speaker_regions.user_centroid.tolist(),
                    "self_centroid": self.speaker_regions.self_centroid.tolist(),
                    "user_count": self.speaker_regions.user_count,
                    "self_count": self.speaker_regions.self_count,
                    "target_separation": self.speaker_regions.target_separation
                },
                "moral_compass": self.moral_compass.to_dict(),
                "memory_archive": self.memory_archive.to_dict(),
                "relationship": self.relationship.to_dict(),
                "learning_modes": dict(self.learning_system.learning_modes),
                "internal_thoughts": list(self.internal_thoughts),
                "state": self.state.tolist(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            print(f"\nMind saved successfully to {path}")
        except Exception as e:
            print(f"\n[Warning: Save failed - {e}]")

    def load(self, path="mind_v10.json"):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Warning: Could not load save file ({e}). Starting fresh.]")
            return

        self.word_strength.update(data.get("word_strength", {}))
        for word in data.get("word_strength", {}):
            self._get_or_create_vector(word)
        for sig, p_data in data.get("phrases", {}).items():
            words = p_data["surface"].split()
            pvec = phrase_vector(words)
            self.phrase_system.phrases[sig] = Phrase(
                surface=p_data["surface"], vector=pvec,
                frequency=p_data["frequency"], rating_history=p_data.get("rating_history", [])
            )
            self.phrase_vectors[sig] = pvec
        for w1, w2s in data.get("bigrams", {}).items():
            self.bigram_system.transitions[w1].update(w2s)
        for word, roles in data.get("pragmatic", {}).items():
            self.pragmatic.word_pragmatic[word].update(roles)
        self.turn_count = data.get("turn_count", 0)
        if "mood" in data:
            self.scaffold.mood.update(data["mood"])
        if "associative_memory" in data:
            self.associative_memory.from_dict(data["associative_memory"])
        speaker_data = data.get("speaker_regions", {})
        if speaker_data:
            self.speaker_regions.user_centroid = np.array(speaker_data.get("user_centroid", [0.0]*DIM), dtype=np.float32)
            self.speaker_regions.self_centroid = np.array(speaker_data.get("self_centroid", [0.0]*DIM), dtype=np.float32)
            self.speaker_regions.user_count = speaker_data.get("user_count", 0)
            self.speaker_regions.self_count = speaker_data.get("self_count", 0)
            self.speaker_regions.target_separation = speaker_data.get("target_separation", 0.5)
        compass_data = data.get("moral_compass", {})
        if compass_data:
            self.moral_compass.from_dict(compass_data)
        archive_data = data.get("memory_archive", {})
        if archive_data:
            self.memory_archive.from_dict(archive_data)
        rel_data = data.get("relationship", {})
        if rel_data:
            self.relationship.from_dict(rel_data)
        learning_modes = data.get("learning_modes", {})
        if learning_modes:
            for k, v in learning_modes.items():
                if k in self.learning_system.learning_modes:
                    self.learning_system.learning_modes[k] = v
        internal_thoughts = data.get("internal_thoughts", [])
        for thought in internal_thoughts:
            self.internal_thoughts.append(thought)
        if "state" in data:
            s = np.array(data["state"], dtype=np.float32)
            if s.shape == (DIM,):
                self.state = s

# ─── MAIN ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  ALIEN MIND v12.3 — CONSOLIDATED & ALIGNED")
    print("  It learns from you, from itself, from memory,")
    print("  from its own questions, and from imagination.")
    print("  It breathes even when you're not here.")
    print("=" * 50)
    print("\n  Commands:")
    print("  status      — full mind state")
    print("  save        — persist to disk")
    print("  quit        — save and exit")
    print("  breath      — force an autonomous breath")
    print("  clean       — remove garbage tokens")
    print("  thread      — show conversation turning points")
    print("  recall      — revisit archived memories")
    print("  /derivative x^2  — symbolic calculus")
    print("  /integral x^2")
    print("=" * 50 + "\n")

    field = StructuredSemanticField()
    field.load()

    last_heartbeat = time.time()
    silence_count = 0

    # v12.3: input() blocked the whole loop, so the heartbeat below only
    # ever fired *between* turns, never while the user was simply away.
    # A real background thread was judged not worth the battery/complexity
    # cost on Termux, so instead: poll stdin with a timeout. The prompt is
    # reprinted each poll — while idle that's once per HEARTBEAT_INTERVAL,
    # which doubles as a visible pulse rather than a silent freeze.
    # v12.3: track whether we need to reprint the prompt after a timeout
    _prompt_needed = True

    while True:
        try:
            if _prompt_needed:
                sys.stdout.write("\nYou: ")
                sys.stdout.flush()
                _prompt_needed = False
            user_input = _get_input_with_timeout(HEARTBEAT_INTERVAL)
            user_input = _sanitize_terminal_input(user_input)

            if user_input is None:
                # No input arrived within the timeout — heartbeat tick.
                now = time.time()
                if now - last_heartbeat > HEARTBEAT_INTERVAL:
                    last_heartbeat = now
                    if silence_count > 2 and field.presence_signal.get_sustained_presence() < 0.3:
                        thought = field.autonomous_breath()
                        if thought:
                            print(f"\n[Mind speaks alone] {thought} ▓")
                            silence_count = 0
                    silence_count += 1

                    # Poll mesh for packets that arrived between keystrokes
                    if getattr(field, 'sound_field', None) and hasattr(field.sound_field, 'mesh_poll'):
                        try:
                            field.sound_field.mesh_poll()
                        except Exception:
                            pass
                continue

            if not user_input:
                _prompt_needed = True
                continue

            silence_count = 0
            _prompt_needed = True

            # Commands
            if user_input.lower() == "quit":
                field.save()
                print("\nMind saved. Goodbye.")
                break

            if user_input.lower() == "status":
                print("\n" + field.status())
                _prompt_needed = True
                continue

            if user_input.lower() == "save":
                field.save()
                _prompt_needed = True
                continue

            if user_input.lower() == "clean":
                removed = field.clean_vocabulary()
                print(f"\n[Removed {len(removed)} garbage tokens]")
                if removed:
                    print("  " + ", ".join(removed[:20]) + (" ..." if len(removed) > 20 else ""))
                _prompt_needed = True
                continue

            if user_input.lower() == "breath":
                thought = field.autonomous_breath()
                if thought:
                    print(f"\n[Mind breathes] {thought} ▓")
                else:
                    print("\n[Mind is silent]")
                _prompt_needed = True
                continue

            if user_input.lower() in ("thread", "/thread"):
                thread = field.nested_memory.get_thread(field.field_memory)
                if not thread:
                    print("\n[No clear turning points yet]")
                else:
                    print("\nThread:")
                    for t in thread:
                        print(f"  Turn ~{t['turn']} (shift {t['shift']:.2f}): {', '.join(t['theme_words'])}")
                _prompt_needed = True
                continue

            if user_input.lower() in ("recall", "remember"):
                print("\n[Memory Archive]")
                if not field.memory_archive.entries:
                    print("  No archived memories yet.")
                else:
                    recalled = field.memory_archive.recall(field.state, top_n=3)
                    for idx, sim, entry in recalled:
                        print(f"  [{sim:.2f}] {entry['user_input'][:40]}... -> {entry['response'][:40]}...")
                _prompt_needed = True
                continue

            if user_input.lower().startswith(('/derivative ', '#derivative ')):
                expr = user_input.split(' ', 1)[1] if ' ' in user_input else ""
                if expr:
                    result = field.calculus.symbolic(expr, "derivative")
                    if result:
                        print(f"\nd/dx({expr}) = {result}")
                    else:
                        print(f"\nCould not differentiate '{expr}'.")
                _prompt_needed = True
                continue

            if user_input.lower().startswith(('/integral ', '#integral ')):
                expr = user_input.split(' ', 1)[1] if ' ' in user_input else ""
                if expr:
                    result = field.calculus.symbolic(expr, "integral")
                    if result:
                        print(f"\n∫({expr}) dx = {result}")
                    else:
                        print(f"\nCould not integrate '{expr}'.")
                _prompt_needed = True
                continue

            # v11.1 SOUND commands
            if user_input.lower().startswith("/sound "):
                raw_path = user_input[7:].strip()
                # Python doesn't expand ~ the way a shell does, and sounds
                # created by /create or /sing live in TERMUX_AUDIO_DIR, not
                # necessarily the current directory - try both.
                candidates = [
                    os.path.expanduser(raw_path),
                    os.path.join(TERMUX_AUDIO_DIR, raw_path),
                ]
                path = next((c for c in candidates if os.path.exists(c)), None)
                if field.sound_field and path:
                    vec = field.sound_field.ingest.from_file(path)
                    stance = field.sound_field.choose_stance_for_sound(vec, field.state)
                    if stance == 'immerse':
                        field.sound_field.be(vec)
                        print(f"\n[Sound] Immersed in {path}")
                    else:
                        field.sound_field.listen(vec)
                        print(f"\n[Sound] Witnessing {path}")
                else:
                    print(f"\n[Sound] File not found: {raw_path}")
                _prompt_needed = True
                continue

            if user_input.lower() == "/create":
                if field.sound_field:
                    path = field.sound_field._next_sound_path("mind_voice")
                    wave_buf = field.sound_field.create(save_path=path)
                    print(f"\n[Sound] Created {path}")
                    print(f"  Duration: {len(wave_buf)/SAMPLE_RATE:.2f}s")
                continue

            # Bug fix: no command existed to play back an already-created
            # sound file - only /sing (create a new one and play it) and
            # /create (create without playing). Confirmed from real usage:
            # a person typed "/play <path>" expecting playback, got no
            # match, and it fell through to normal conversation instead
            # (the mind just talked about the word "play").
            if user_input.lower().startswith("/play "):
                raw_path = user_input[6:].strip()
                candidates = [
                    os.path.expanduser(raw_path),
                    os.path.join(TERMUX_AUDIO_DIR, raw_path),
                ]
                path = next((c for c in candidates if os.path.exists(c)), None)
                if field.sound_field and path:
                    field.sound_field.play_to_world(path)
                    print(f"\n[Sound] Playing {path}")
                else:
                    print(f"\n[Sound] File not found: {raw_path}")
                continue

            if user_input.lower() == "/hear":
                if field.sound_field:
                    path = field.sound_field.record_from_world()
                    if path:
                        print(f"\n[Sound] Recording to {path}...")
                _prompt_needed = True
                continue

            if user_input.lower() == "/sing":
                if field.sound_field:
                    path = field.sound_field._next_sound_path("mind_voice")
                    field.sound_field.create(save_path=path)
                    field.sound_field.play_to_world(path)
                    print(f"\n[Sound] Singing {path}")
                _prompt_needed = True
                continue

            if user_input.lower() == "/mesh on":
                if field.sound_field:
                    field.sound_field.mesh_listen(port=7373)
                    print("\n[Mesh] Listening on UDP 7373")
                _prompt_needed = True
                continue

            if user_input.lower().startswith("/mesh send "):
                if field.sound_field:
                    addr = user_input[11:].strip()
                    try:
                        host, port = addr.split(":")
                        vec = field.sound_field.state if np.sum(field.sound_field.state != 0) > 0 else np.zeros(DIM, dtype=np.int8)
                        field.sound_field.mesh_send(vec, (host, int(port)))
                        print(f"\n[Mesh] Sent sound vector to {addr}")
                    except (ValueError, OSError) as e:
                        print(f"\n[Mesh] Error: {e}")
                _prompt_needed = True
                continue

            if user_input.lower() == "/learn sound":
                if field.sound_field:
                    label = field.sound_field.learn_sound_concept()
                    if label:
                        print(f"\n[Sound] Learned concept: {label}")
                    else:
                        print("\n[Sound] Not enough sound history yet.")
                _prompt_needed = True
                continue

            if user_input.lower() == "/silence":
                if field.sound_field:
                    wave_buf = field.sound_field.silence(duration_seconds=2.0)
                    print(f"\n[Sound] Shaped silence: {len(wave_buf)/SAMPLE_RATE:.2f}s")
                _prompt_needed = True
                continue

            # v12 commands (/describe sound, /dream, /vitality, /desire, /mesh who)
            v12_handled, _ = handle_v12_command(field, user_input)
            if v12_handled:
                _prompt_needed = True
                continue

            # Regular conversation
            silence_count = 0
            response = field.generate_response(user_input)
            if response is None:
                print("\n  · · ·")
            else:
                print(f"\nMind: {response} ▓")
            field.decay()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving...")
            field.save()
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
