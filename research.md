# Active Research Context (`research.md`)

This is a dynamic, living document intended purely to bring AI developer agents up to speed on **short-lived, currently hyper-relevant** technical constraints and context. As project phases conclude (e.g., dataset loader creation finishes), old information here should be boldly deleted to prevent context drift and keep this document short and punchy.

---

## 🏗️ Current Environment & Tooling
- **Package Management:** This project strictly uses **`uv`** for dependency management and virtual environments. Do not use `pip` natively; instruct `uv` instead.

## 🛠️ Active Development Quirks
- **VSI File Reading & Warnings:** The Olympus `.vsi` datasets contain non-standard tags that cause invasive, noisy warnings in the terminal when read. When writing new data loading scripts utilizing `slideio`, you must **always** wrap the open call with the `suppress_stderr` helper (found in `focus_offset.utils.io_utils`) to capture and hide these warnings, otherwise logs will be impossibly bloated.
