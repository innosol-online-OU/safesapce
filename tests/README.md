# SafeSpace Tests

Verification scripts for validating attack methods.

## Running Tests

```bash
# Single test
python tests/verify_liquid_warp.py

# All tests (when implemented)
# pytest tests/
```

## Test Scripts

- `verify_liquid_warp.py` - Phase 17 Liquid Warp validation
- `verify_resonant_ghost.py` - Phase 16 Resonant Ghost validation
- `verify_*.py` - Legacy verification scripts

## Adding Tests

1. Create `tests/verify_your_method.py`
2. Implement verification logic
3. Document expected metrics
4. Update this README

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for testing standards.
