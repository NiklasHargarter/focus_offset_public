# Agent Execution Guide for Tracer Bullet Plans

This repository is undergoing a structured, phase-by-phase architectural rewrite. The overarching blueprint is defined in `requirements.md`. The execution roadmap is broken into "Tracer Bullet" slices located in the `plans/` directory.

**As an AI Developer Agent, your directive when asked to execute a plan is to follow these strict rules:**

## 1. Context Acquisition
Before touching code, you MUST:
1. Read `requirements.md` to understand the overarching architectural constraints (Grey Box deep modules, Shared Indexer, etc.).
2. Read `research.md` for hyper-relevant, active context (e.g., environment tooling, active quirks).
3. Read the specific `plans/XX_plan_name.md` file you have been tasked with executing.

## 2. Dependency Verification
Every plan declares its dependencies. Do not begin work on a plan if its prerequisites are not explicitly marked as complete.

## 3. The Vertical Slice Execution
A tracer bullet phase is a **thin, vertical integration slice**, completely cutting through data layers, models, and top-level orchestrators. 
- You are not building detached utilities; you are building a complete, runnable path.
- Always start by defining the public interface of the new deep module. 
- Lock down that interface with a PyTorch/pytest test.
- Fill in the implementation.
- Wire it to the CLI orchestrating script.

## 4. In-Place Transformation & Salvage
This rewrite is happening *in-place*. 
- Build the new architecture in new module namespaces (e.g., `core/` or `v2/`).
- Copy mathematically complex logic (like specific FFT transforms or `slideio` bindings) from the legacy codebase into your new modules.
- **Do not delete legacy folders** until your new vertical slice is 100% complete, tested, and verifiably replaces them. 

## 5. Verification
You must prove the slice works. Every plan requires a "dry run" or demoable execution path. Run the orchestrator script utilizing your new slice and verify it logs outputs correctly without crashing.

## 6. Completion
Once verified, update the plan's `.md` file to append an execution summary, check off its tasks, and notify the user that the phase is formally complete.
