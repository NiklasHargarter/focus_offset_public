# Phase 00: Top-Down Interface Design

**Status:** Not Started
**Dependencies:** None
**Goal:** Define all public module contracts working top-down before any implementation begins. Produces stub files with `raise NotImplementedError` and populates `## Interface Decisions` in every downstream phase plan.

## What This Phase Is

This is a **design-only phase** — no implementation, no tests, no data. Run `/design-an-interface` for each layer listed below, working top-down. The agent explores the codebase and requirements, then proposes stub signatures for your review.

After each phase completes, revisit relevant interfaces and re-run `/design-an-interface` to refine downstream contracts based on what was learned.

## Design Order (top-down)

Work in this order so each layer knows what the layer above needs from it:

1. **Training orchestrator** — what does `train_focus.py` need from the DataLoader? What CLI flags exist? What does a training loop iteration look like?
2. **DataLoader interfaces** — `OffsetPredictionDataset.__getitem__` return type, `SyntheticPairDataset.__getitem__` return type, `DataModule` interface
3. **Transform strategies** — `TransformStrategy` base class, spatial/FFT/DWT variants
4. **Augmentation provider** — `AugmentationProvider` interface, preset registry
5. **CSV index schema** — exact column names and types that all of the above rely on
6. **SharedIndexer** — constructor, `build()` method, `load()` method
7. **Focus metric module** — metric function signatures, peak detection stub
8. **Dataset-specific adapters** — VSI reader, OME-TIFF reader, Jiang2018 adapter

## Output of This Phase

For each interface above:
- A stub `.py` file in the new module namespace with signatures + docstrings + `raise NotImplementedError`
- The `## Interface Decisions` section populated in the relevant phase plan

## Completion Criteria

- [ ] Stub files exist for all 8 interface layers above
- [ ] Every phase plan (01–07) has a populated `## Interface Decisions` section
- [ ] CSV schema is fully specified (column names, types, units)
- [ ] No implementation code exists — stubs only

## Notes

- **No visual proof artifact required** for this phase (design only)
- After Phase 02 completes, revisit Phase 03–07 interface decisions
- After Phase 03 completes, revisit Phase 04–07 interface decisions
- Interfaces are fixed once a phase begins — only update downstream interfaces between phases
