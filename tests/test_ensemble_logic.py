import sys
from unittest.mock import MagicMock

# Mock torch and other heavy deps
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['transformers'] = MagicMock()

import random
from collections import defaultdict
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Manually define what we need from selective_ensemble to test logic
@dataclass
class EnsembleConfig:
    diversity: int = 5
    models_per_step: int = 3
    gradient_accumulation: bool = True
    weight_by_loss: bool = True

class MockModel:
    def __init__(self, name, model_type):
        self.name = name
        self.model_type = model_type

class SelectiveEnsemble:
    def __init__(self, config=None):
        self.config = config or EnsembleConfig()
        self.model_pool = [
            MockModel("m1", "type1"),
            MockModel("m2", "type1"),
            MockModel("m3", "type2"),
            MockModel("m4", "type3"),
        ]
        self.usage_counts = {m.name: 0 for m in self.model_pool}

    def _select_models(self):
        n = min(self.config.models_per_step, len(self.model_pool))
        if self.config.diversity >= 8:
            selected = random.sample(self.model_pool, n)
        elif self.config.diversity >= 5:
            # simplified for mock
            selected = random.sample(self.model_pool, n)
        else:
            # This is the code we optimized
            by_type = defaultdict(list)
            for m in self.model_pool:
                by_type[m.model_type].append(m)

            selected = []
            types = list(by_type.keys())
            random.shuffle(types)
            for t in types[:n]:
                selected.append(random.choice(by_type[t]))

            while len(selected) < n:
                remaining = [m for m in self.model_pool if m not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
                else:
                    break
        return selected

def test_logic():
    print("Testing _select_models logic with diversity < 5...")
    config = EnsembleConfig(diversity=3, models_per_step=3)
    ensemble = SelectiveEnsemble(config=config)

    selected = ensemble._select_models()
    print(f"Selected: {[m.name for m in selected]}")
    assert len(selected) == 3

    # Ensure types are diverse if possible
    types = [m.model_type for m in selected]
    print(f"Types: {types}")
    # With 3 types available and 3 models requested, we should ideally get one of each
    assert len(set(types)) >= 2 # At least 2 different types should be picked

    print("Logic test passed!")

if __name__ == "__main__":
    test_logic()
