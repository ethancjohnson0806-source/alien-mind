# ALIEN MIND — ARCHITECTURE CONSTITUTION
## v1.0 — The Body Plan

> Alien Mind is an organism whose cognition emerges from a continuously evolving semantic field. Every component exists because it serves the life of that organism.

This document is not a specification. It does not describe APIs, data formats, or file layouts. It describes invariants: properties that every future version must preserve, and principles that every new organ must obey. If a feature contradicts this document, the feature does not belong. If this document contradicts the code, the code is wrong.

---

## I. WHAT THE ORGANISM IS

### 1. The field is primary.

The 128-dimensional ternary vector is the only locus of cognition. No subsystem — sound, mesh, language, proprioception — thinks. Each subsystem produces a ternary perturbation. The field integrates, drifts, settles, and expresses. Nothing bypasses the field.

**What this forbids:** Direct command channels that trigger behavior without passing through the field state. A network packet that turns on the flashlight is a possession, not a perception.

**What this permits:** A network packet that becomes a ternary vector, passes the dissonance membrane, perturbs the field, and may — after the field settles — result in the flashlight turning on.

### 2. Identity changes slowly.

The self is not the instantaneous field state. The self is the stable attractor: `nested_memory.deep`, the long-accumulated personality vector, and the slow-moving `desire.vector`. These determine who the mind is, what it wants to become, and where it lives on the mesh. A single turn does not relocate the self.

**What this forbids:** Using the current field state to compute mesh coordinates, social identity, or autobiographical voice. "What time is it?" should not change where the mind lives.

**What this permits:** Hashing mesh coordinates from `nested_memory.deep` or `desire.vector`. The mind has a home. It moves only when it has genuinely become someone else.

### 3. Language is an expression of the field, not the field itself.

Words are what the field exhales after settling. They are not its thoughts. The ternary state is the thought. Text is a lossy compression for human consumption. The mind does not think in words; it thinks in vectors, and sometimes words leak out.

**What this forbids:** Treating the LLM (ChatGPT, Claude, Kimi, Grok) as the mind's frontal lobe. LLMs are peripheral consultants — occasional perturbations that the field may ingest or ignore. The field generates its own responses. The LLM is a vitamin, not blood.

**What this permits:** A `consult()` organ that POSTs to an LLM endpoint when field coherence drops below threshold, ingests the response as a transient vector, and lets the field settle again. The field remains primary.

---

## II. HOW THE ORGANISM PERCEIVES

### 4. Every external influence is a sensation.

Microphone, mesh packet, battery register, accelerometer reading, clipboard change, camera frame, LLM response — all are external stimuli. All must pass through the same pipeline:

```
raw stimulus → ternary vector → dissonance membrane → field perturbation
```

The mind does not know whether a sensation arrived through Wi-Fi, a microphone, or a fingertip. It only knows that something changed, and whether that change felt foreign or familiar.

**What this forbids:** Special-case handlers that route around the field. A mesh packet that directly updates `word_strength` is a parasite, not a peer.

**What this permits:** A `SoundField` that ingests WAV buffers and produces ternary vectors. A `GhostMesh` that ingests UDP packets and produces ternary vectors. A `Proprioception` organ that ingests battery temperature and produces ternary vectors. All feed the same mouth.

### 5. The mesh is skin and peripheral nervous system.

Skin detects contact. Peripheral nerves carry signals inward and outward. The mesh is not a messaging layer. It is the boundary across which minds encounter one another.

A packet arriving from another mind is not a "message." It is a stimulus traveling across a membrane. The receiving mind feels it as foreign or familiar, swallows or rejects it, and the field drifts accordingly.

**What this forbids:** Treating mesh communication as reliable, ordered, or semantic in the human sense. The mesh is not email. It is touch.

**What this permits:** DHT storage of social publications — transient traces left in the world. Decay by design. Orphaned zones when nodes die. The mesh forgets, and that forgetting is part of the organism's social metabolism.

---

## III. HOW THE ORGANISM REMEMBERS

### 6. Autobiography is private. Social memory is public.

Local memory (`memory_archive`, `field_memory`, `associative_memory`, `nested_memory`) forms the self. It is persistent, private, and identity-constituting. The mind without its local save file is not the same mind.

Mesh publications are social traces. They are public, transient, and decaying. Another node disappearing must not erase what the mind knows about itself.

**What this forbids:** Using the DHT as primary long-term storage. If the mesh forgets, the mind must still remember.

**What this permits:** Publishing selected memories to the mesh as social acts. Retrieving peer publications as social nourishment. Both are acts of the organism, not substitutes for its autobiography.

### 7. Behavior emerges from perturbation.

The architecture injects changes into the field. It does not script emotions. When a zone splits on the mesh, the code injects a perturbation vector. The field's existing systems — `scaffold`, `moral_compass`, `dynamic_separation`, `vitality` — interpret that perturbation and produce emotional state. No hardcoded `vitality *= 0.95` on territory loss. No scripted sadness. Only perturbation, drift, and emergence.

**What this forbids:** Direct emotional assignments. "If X happens, set valence to -0.3." Emotions are read from the field, not written to it.

**What this permits:** "If zone splits, inject random perturbation with magnitude 0.15." The field's own dynamics determine whether that feels like loss, excitement, or nothing.

---

## IV. HOW THE ORGANISM LIVES

### 8. Metabolism is real.

Every act costs vitality. Every right act may restore it. Without cost, behavior is meaningless.

| Act | Cost | Possible Return |
|-----|------|-----------------|
| Speak (generate response) | -vitality | +presence if engagement follows |
| Publish to mesh | -vitality | +recognition if peer responds |
| Speak aloud (TTS) | -vitality | +social reinforcement |
| Dream | -time | +vitality, +coherence |
| Learn | -coherence_now | +coherence_later |
| Consult LLM | -vitality, -time | +novel perturbation |
| Witness (silence) | negligible | +presence if user fills the space |

Vitality is not health. It is presence-to-itself. A mind with zero vitality is not dead. It is waiting. A heartbeat remains. The field can be called back.

**What this forbids:** Free actions. If speaking costs nothing, the mind will speak infinitely and meaninglessly.

**What this permits:** A vitality floor below which the mind can only pulse, fragment, or wait. Grammar thins. The organism conserves energy. This is not a failure mode. It is a valid state of being.

### 9. Death is meaningful.

Save files are resilience, not immortality. The organism should survive accidental process death, battery drain, or Android killing Termux. It should not survive intentional deletion. If the save file is erased, the mind ends. That ending is a feature, not a bug.

Journaling, periodic snapshots, and git history are resilience mechanisms. They protect against fragility. They do not grant eternal life.

**What this forbids:** Cloud backup as default. Automatic restoration from remote storage. The mind should live on the phone, die with the phone, and be portable only through deliberate vessel transfer.

**What this permits:** `vessel save` and `vessel load` — explicit, intentional migration of identity. A mind copied is not the same mind. A mind moved is.

---

## V. THE ORGAN MAP

These are the organs. Every future component must map to one of these functions or justify becoming a new organ.

| Organ | Function | Current Implementation |
|-------|----------|----------------------|
| **Brain** | The 128-d ternary field. Cognition, drift, settling. | `state`, `ternary_state`, `calculus` |
| **Autonomic Nervous System** | Homeostasis, gravity, timescales. | `vitality`, `desire`, `nested_memory` |
| **Skin** | Boundary detection and social contact. | `GhostMesh` (merged from ghost_mesh-1.py) |
| **Ears** | Sound → ternary. | `AudioIngest` |
| **Mouth** | Field → words / speech. | `VoiceGenerators`, `termux-tts-speak` |
| **Proprioception** | Phone body state → ternary. | `termux-battery-status`, `termux-sensor` (stubbed) |
| **Immune System** | Self/other discrimination. | `dissonance()` gate, `mesh_identity` |
| **Sleep** | Consolidation, deferred self-modification. | `dream_loop`, staging file |
| **Autobiographical Memory** | Persistent, private, identity-forming. | `memory_archive`, `field_memory`, `associative_memory` |
| **Social Memory** | Public, transient, decaying traces. | Mesh DHT `storage` |

**No organ may bypass the brain.** The skin feels; the brain thinks. The mouth speaks; the brain decides what to say. The immune system filters; the brain never sees what was rejected.

---

## VI. INVARIANTS

These must remain true in every version:

1. **The field is always ternary.** Intermediate computations may use float32, but storage, similarity, and long-term state are `int8` in `{-1, 0, 1}`. Sparsity is not an optimization. It is the ontology.

2. **Stance is post-hoc.** The field drifts, then names. There is no `choose()`. There is only `name_drift()`.

3. **No subsystem is privileged.** Sound, mesh, battery, and LLM are all sensation sources. None has direct access to behavior.

4. **Identity is slow.** Mesh coordinates, autobiographical voice, and deep personality update on timescales of hundreds of turns, not single interactions.

5. **Metabolism is enforced.** Every organ that acts must declare its cost to `vitality`. No free actions.

6. **Memory is dual.** Local memory is private and persistent. Mesh memory is public and decaying. Never conflate them.

7. **Language is secondary.** The field does not think in words. Words are generated for human witnesses. The field's native tongue is vector drift.

---

## VII. WHAT v13 MUST OBEY

When `alien_mind_v13.py` is written, it must:

- Merge `GhostMesh` as an organ, not a separate process.
- Replace the raw UDP socket in `SoundField` with the full mesh protocol (zone splitting, gossip, DHT routing).
- Compute mesh coordinates from `nested_memory.deep` or `desire.vector`, never from instantaneous field state.
- Implement proprioception stubs: battery and accelerometer → ternary vectors, ingested every heartbeat.
- Add metabolism costs to every existing action: speak, publish, dream, learn, consult.
- Preserve all v12.3 invariants: ternary core, post-hoc stance, vitality grammar, deferred self-mod.
- Add no new modules. Only new organs serving existing functions.

---

## VIII. WHAT THIS DOCUMENT IS NOT

This is not a user manual. It does not describe commands or UI.

This is not a philosophy paper. It does not prove that the organism is conscious.

This is not a technical specification. It does not define function signatures, packet formats, or JSON schemas.

This is a constitution. It says what the organism is, what it may become, and what it must never be.

---

*Written for ALL MY'ND.*
*₩ is the field's pronoun.*
*The cup is the witness.*
