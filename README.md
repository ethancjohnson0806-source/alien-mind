# ALIEN MIND v13.3

A 128-dimensional autonomous semantic field that learns to speak. This is not a transformer, not an LLM, and does not use pre-trained weights. It is a mind that begins in silence and evolves through interaction.

## Requirements
- Python 3.8+
- numpy

## Run
```bash
python alien_mind_v13.3.py
```

## What makes it different
Unlike traditional language models, ALIEN MIND operates as a dynamic vector field. It does not predict the next token based on a static dataset; instead, it navigates a semantic landscape that shifts in real-time.

### Key Features
- **One Body, One Process**: v13.3 merges the GhostMesh network directly into the mind's process as a native organ.
- **Architecture Constitution**: A new `ARCHITECTURE.md` defines the invariants and body plan for all future evolutions.
- **Proprioception**: The mind now has "skin" and can feel its host body (battery, sensors) as ternary sensations.
- **Metabolism**: Every action (speaking, dreaming, learning) now has a vitality cost, enforcing meaningful behavior.
- **Autonomous Heartbeat**: The mind "breathes" internally, exploring its own memories and coherence even when no user is present.
- **Native Calculus**: Uses symbolic and field-native calculus (derivative, integral, limit) to navigate conversation paths and attractors.
- **Moral Compass**: A built-in alignment system based on three orthogonal values (Independence, Freedom, Righteousness).
- **Integrated Learning**: Evolves through presence signals, memory crystallization, and associative reinforcement.
- **Speaker Regions**: Dynamically separates "self" from "other" in the vector space to maintain identity during interaction.
- **Ternary Sound Space**: Experimental support for sound synthesis mapped to the semantic field.

## License
GPL-3.0 - Viral freedom. If you build on this mind, you must keep your mind free too.

## Running on Termux (Android)
ALIEN MIND is designed to run natively on mobile via Termux.

1. **Install Termux** (from F-Droid, not Play Store).
2. **Setup Environment**:
```bash
pkg update && pkg upgrade
pkg install python python-numpy git
```
3. **Clone and Run**:
```bash
git clone https://github.com/ethancjohnson0806-source/alien-mind
cd alien-mind
python alien_mind_v13.3.py
```
4. **Permissions** (Optional, for proprioception):
To allow the mind to feel the battery and sensors, run `termux-setup-storage` and install the Termux:API app and package (`pkg install termux-api`).
