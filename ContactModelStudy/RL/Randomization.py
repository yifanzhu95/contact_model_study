"""Reserved domain lifecycle. This version supports fixed domains only.

Physical parameters, solver parameters and solver-category mixtures are three
different future operations. Declaring one cannot silently turn into a no-op.
The lifecycle is invoked before backend creation and before selected resets;
future implementations must report effective values and serialize their RNG.
"""
from copy import deepcopy


def reserved_spec(value):
    if not isinstance(value, dict): raise ValueError("randomization must be a JSON object")
    defaults = {"schema": "leap_domain_plan_v1", "enabled": False, "seed": 0,
        "sampling_unit": "world_reset", "physical_parameters": {}, "solver_parameters": {},
        "model_probabilities": {}}
    if set(value)-set(defaults): raise ValueError("Unknown randomization setting")
    spec = {**defaults, **deepcopy(value)}
    if (spec["schema"] != defaults["schema"] or type(spec["enabled"]) is not bool
            or type(spec["seed"]) is not int or spec["seed"] < 0
            or spec["sampling_unit"] != "world_reset"
            or any(not isinstance(spec[k], dict) for k in
                   ("physical_parameters", "solver_parameters", "model_probabilities"))):
        raise ValueError("Invalid reserved randomization contract")
    if spec["enabled"] or any(spec[k] for k in ("physical_parameters", "solver_parameters", "model_probabilities")):
        raise NotImplementedError("Domain randomization is reserved only: physical/solver changes and model mixing "
                                  "are not implemented; fixed-domain execution only")
    return spec


class FixedDomainLifecycle:
    """Actual extension points, currently with no draws or parameter mutations."""
    def __init__(self, config):
        self.spec = reserved_spec(config.randomization or {})
        self.model, self.backend = config.contact_preset, config.physics_backend

    def before_backend_create(self):
        # Validation happens before expensive physics allocations. Future
        # structural/model changes must be resolved here, never halfway through
        # an episode or by relabeling an already compiled backend.
        reserved_spec(self.spec)

    def before_reset(self, ids, cases):
        return [{"world_id": i, "case_id": case.case_id, "domain_id": f"fixed:{self.backend}:{self.model}",
                 "contact_model": self.model, "randomized": False,
                 "effective_physical_overrides": {}, "effective_solver_overrides": {}}
                for i, case in zip(ids, cases, strict=True)]

    def describe(self):
        return {"schema": "leap_domain_lifecycle_v1", "requested": self.spec,
            "status": "reserved_only_fixed_domain", "effective_contact_model": self.model,
            "physical_parameter_updates": [], "solver_parameter_updates": [], "model_mixing": False,
            "future_requirements": ["backend capability validation and rebuild rules",
                "per-world effective values in reset ledger", "independent serialized random streams",
                "mixed-model collector routing; no label-only substitutions"]}
