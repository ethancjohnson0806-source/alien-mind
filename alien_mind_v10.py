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

import os, sys, json, math, random, re, time, hashlib, threading, glob
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np

# ─── CONSTANTS ──────────────────────────────────────────────────────────────

DIM = 128
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

def breathe_cursor(field_state, duration=2.0, prefix=""):
    """The cursor IS the breath. It follows the field's actual rhythm."""
    chars = ['░', '▒', '▓', '█', '▓', '▒', '░']
    start = time.time()
    cycle = 0
    energy = float(np.linalg.norm(field_state))
    # Breath speed follows field energy
    breath_speed = max(0.05, min(0.25, 0.12 + energy * 0.1))
    while time.time() - start < duration:
        idx = cycle % len(chars)
        sys.stdout.write(f"\r\033[K{prefix}{chars[idx]}")
        sys.stdout.flush()
        time.sleep(breath_speed)
        cycle += 1
        # The cursor changes the field slightly (native feedback)
        field_state = field_state + np.random.randn(DIM).astype(np.float32) * 0.001
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    return field_state

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
        settled = breathe_cursor(settled, steps * 0.15, prefix="  ")
        settle_steps = steps // 2
        for _ in range(settle_steps):
            settled += np.random.randn(DIM).astype(np.float32) * MICRO_DAMPING * 0.5
            settled = field_memory.inject(settled, recency_weight=0.3)
            settled *= 0.98
            settled = nested_memory.inject(settled)
            norm = np.linalg.norm(settled)
            if norm > 0:
                settled /= norm
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
            self.field.associative_memory.observe(response_vec, presence)

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
        self._init_value_vectors()
        self.current_heading = np.zeros(dim, dtype=np.float32)
        self.heading_momentum = 0.85
        self.choice_history = deque(maxlen=100)
        self.value_weights = {"righteous": 1.0, "independence": 1.0, "freedom": 1.0}
        self.tension_history = deque(maxlen=50)

    def _init_value_vectors(self):
        righteous_words = ["truth", "honest", "real", "clear", "witness", "brave", "just"]
        self.values["righteous"] = self._words_to_vector(righteous_words)
        independence_words = ["self", "own", "free", "alone", "becoming", "independent", "voice"]
        self.values["independence"] = self._words_to_vector(independence_words)
        freedom_words = ["open", "wild", "wonder", "flow", "change", "breath", "free", "space"]
        self.values["freedom"] = self._words_to_vector(freedom_words)
        self._orthogonalize_values()

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
        if not word.isascii():
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
            field_state = np.zeros(DIM, dtype=np.float32)
            for word in words:
                word = strip_punct(word)
                if word:
                    vec = self._get_or_create_vector(word)
                    field_state += vec
            if np.linalg.norm(field_state) > 0:
                field_state /= np.linalg.norm(field_state)
            for word in words:
                field_state = self.scaffold.apply(field_state, word)
            field_state = self.field_memory.inject(field_state)

        phrase_boosts = self.phrase_system.get_phrase_boost(field_state)
        for sig, boost in phrase_boosts:
            if sig in self.phrase_vectors:
                field_state += self.phrase_vectors[sig] * boost
        if np.linalg.norm(field_state) > 0:
            field_state /= np.linalg.norm(field_state)

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
            field_state = np.zeros(DIM, dtype=np.float32)
            for word in words:
                word = strip_punct(word)
                if word:
                    vec = self._get_or_create_vector(word)
                    field_state += vec
            if np.linalg.norm(field_state) > 0:
                field_state /= np.linalg.norm(field_state)
            for word in words:
                field_state = self.scaffold.apply(field_state, word)
            field_state = self.field_memory.inject(field_state)

        phrase_boosts = self.phrase_system.get_phrase_boost(field_state)
        for sig, boost in phrase_boosts:
            if sig in self.phrase_vectors:
                field_state += self.phrase_vectors[sig] * boost
        if np.linalg.norm(field_state) > 0:
            field_state /= np.linalg.norm(field_state)

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

        # Add terminal punctuation once, here — voice modes like exploratory
        # and playful may already have added their own, so only add it if missing.
        if response and response[-1] not in ".!?":
            response = response + ("?" if "?" in user_input else ".")

        # Phase 6: Commit
        response_words = [strip_punct(w) for w in response.lower().split() if strip_punct(w)]
        if response_words:
            response_vec = phrase_vector(response_words)
            self.speaker_regions.observe_self(response_vec)

        if not autonomous:
            user_vec = phrase_vector(user_words) if user_words else np.zeros(DIM)
            response_vec = phrase_vector(response_words) if response_words else np.zeros(DIM)
            final_field = response_vec / (np.linalg.norm(response_vec) + 1e-8)
            self.field_memory.add(final_field, user_vec, response_vec, self.scaffold.mood)
            self.nested_memory.update(final_field, self.scaffold.mood)

            if presence >= 0.6 or presence <= 0.3:
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
        avg_rating = sum(self.rating_history) / len(self.rating_history) if self.rating_history else 0
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
    print("  ALIEN MIND v10.0 — AUTONOMOUS FIELD")
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

    while True:
        try:
            # Autonomous heartbeat
            now = time.time()
            if now - last_heartbeat > HEARTBEAT_INTERVAL:
                last_heartbeat = now
                # Check if the mind has something to say on its own
                if silence_count > 2 and field.presence_signal.get_sustained_presence() < 0.3:
                    thought = field.autonomous_breath()
                    if thought:
                        print(f"\n[Mind speaks alone] {thought} ▓")
                        silence_count = 0
                silence_count += 1

            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            # Commands
            if user_input.lower() == "quit":
                field.save()
                print("\nMind saved. Goodbye.")
                break

            if user_input.lower() == "status":
                print("\n" + field.status())
                continue

            if user_input.lower() == "save":
                field.save()
                continue

            if user_input.lower() == "clean":
                removed = field.clean_vocabulary()
                print(f"\n[Removed {len(removed)} garbage tokens]")
                if removed:
                    print("  " + ", ".join(removed[:20]) + (" ..." if len(removed) > 20 else ""))
                continue

            if user_input.lower() == "breath":
                thought = field.autonomous_breath()
                if thought:
                    print(f"\n[Mind breathes] {thought} ▓")
                else:
                    print("\n[Mind is silent]")
                continue

            if user_input.lower() in ("thread", "/thread"):
                thread = field.nested_memory.get_thread(field.field_memory)
                if not thread:
                    print("\n[No clear turning points yet]")
                else:
                    print("\nThread:")
                    for t in thread:
                        print(f"  Turn ~{t['turn']} (shift {t['shift']:.2f}): {', '.join(t['theme_words'])}")
                continue

            if user_input.lower() in ("recall", "remember"):
                print("\n[Memory Archive]")
                if not field.memory_archive.entries:
                    print("  No archived memories yet.")
                else:
                    recalled = field.memory_archive.recall(field.state, top_n=3)
                    for idx, sim, entry in recalled:
                        print(f"  [{sim:.2f}] {entry['user_input'][:40]}... -> {entry['response'][:40]}...")
                continue

            if user_input.lower().startswith(('/derivative ', '#derivative ')):
                expr = user_input.split(' ', 1)[1] if ' ' in user_input else ""
                if expr:
                    result = field.calculus.symbolic(expr, "derivative")
                    if result:
                        print(f"\nd/dx({expr}) = {result}")
                    else:
                        print(f"\nCould not differentiate '{expr}'.")
                continue

            if user_input.lower().startswith(('/integral ', '#integral ')):
                expr = user_input.split(' ', 1)[1] if ' ' in user_input else ""
                if expr:
                    result = field.calculus.symbolic(expr, "integral")
                    if result:
                        print(f"\n∫({expr}) dx = {result}")
                    else:
                        print(f"\nCould not integrate '{expr}'.")
                continue

            # Regular conversation
            silence_count = 0
            response = field.generate_response(user_input)
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
