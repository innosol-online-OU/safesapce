# ⚠️ CRITICAL ARCHITECTURE MISMATCH DETECTED

Yes, **Phase 18 (Ghost-Mesh) includes a major refactoring** that moved code from `invisible_core/` to `src/core/`.

## Current Branch Status
- **`feature/ghost-mesh-live` (Integrated into `dev`)**: Uses the **NEW** structure (`src/core`, `src/views`).
- **`research` (Current)**: Uses the **OLD** structure (`invisible_core`, `app_interface`).

## The Problem
If we merge/rebase `research` onto `dev` now, git will mix both structures. 
You will end up with duplicate files (`src/core/cloaking.py` AND `invisible_core/cloaking.py`), causing massive confusion and import errors.

## Recommendation
Since `dev` is now the "Up To Date" source of truth (containing the completed Phase 18):
1.  **STOP using `research`** for now (it is legacy structure).
2.  **Switch to `dev`** to use the new Ghost-Mesh features.
3.  If you have critical experiments in `research`, we should cherry-pick them into the new structure manually.

**Next Step:**
Switch to `dev` to verify Phase 18?
Run: `git checkout dev`
