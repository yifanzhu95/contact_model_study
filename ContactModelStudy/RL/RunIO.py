"""Shared run contracts, checkpoint integrity and evaluation accounting."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import torch

from ContactModelStudy.RL.Contracts import EnvConfig


def training_inputs(config_path=None, resume_path=None, bank_path=None):
    """One contract resolver for direct runs, preflight and independent jobs."""
    from dataclasses import asdict
    from ContactModelStudy.RL.CaseBank import CaseBank
    from ContactModelStudy.RL.SkrlPPO import PPOConfig
    metadata = checkpoint_metadata(resume_path) if resume_path else None
    if config_path is None and metadata is None:
        raise ValueError('A new run requires --config')
    config = (json.loads(Path(config_path).read_text()) if config_path else
              {'environment': recorded_environment_config(metadata['environment']).to_dict(), 'ppo': metadata['ppo']})
    env_config, ppo_config = EnvConfig(**config['environment']), PPOConfig(**config['ppo'])
    if bank_path is None and resume_path:
        bank_path = next((Path(resume_path).parent/n for n in ('cases.json', 'training_cases.json')
                          if (Path(resume_path).parent/n).exists()), None)
        if bank_path is None:
            raise ValueError('Provide --case-bank for this legacy checkpoint')
    bank = (CaseBank.load(bank_path) if bank_path else
            CaseBank.development(count=32, seed=env_config.seed, difficulty=env_config.goal_difficulty,
                                 task_name=env_config.task_name if env_config.task_name != 'cube_reorient' else None))
    bank.require_task(env_config.task_name)
    if metadata:
        validate_resume_request(metadata, env_config, asdict(ppo_config), bank)
    return env_config, ppo_config, bank, metadata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name+".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    pending.replace(path)


def close_resources(resources):
    """Attempt every close without hiding an earlier execution exception."""
    errors = []
    for name, resource in resources:
        if resource is None: continue
        try:
            resource.close()
        except (Exception, KeyboardInterrupt) as exc:
            errors.append({"resource": name, "exception": type(exc).__name__, "message": str(exc)})
    return errors


def recorded_environment_config(description):
    """Resolve saved physics explicitly instead of applying today's defaults.

    Early MJWarp artifacts omitted the requested profile in config but recorded
    the actual dispatch in physics. Preserve that choice for historical restore
    and evaluation. New runs serialize the native MPC profile explicitly.
    """
    config = dict(description["config"])
    if config.get("physics_backend") == "mjwarp" and config.get("numerics_profile") is None:
        profile = description.get("physics", {}).get("numerics_profile")
        if profile is None:
            raise ValueError("Saved MJWarp environment is missing its numerical profile")
        config["numerics_profile"] = profile
    return EnvConfig(**config)


def with_contact_model(config, preset):
    if preset in ("mujoco", "pinocchio", "drake"):
        return replace(config, contact_preset="native", physics_backend=preset+"_eval",
                       device="cpu", numerics_profile=None)
    backends = {"M1": "mjwarp", "M2": "mjwarp", "M3": "comfree", "M4": "xpbd", "M5": "kamino"}
    if preset not in backends:
        raise ValueError("Unknown contact model")
    backend = backends[preset]
    if preset in ("M1", "M2") and config.physics_backend == "mujoco_cpu_diagnostic":
        backend = config.physics_backend
    return replace(config, contact_preset=preset, physics_backend=backend,
                   device="cpu" if backend == "mujoco_cpu_diagnostic" else "cuda:0",
                   numerics_profile=config.numerics_profile if backend == "mjwarp" else None)


def evaluation_config(training, *, environment=None, preset=None):
    """Choose physics independently; keep the task and policy command clock fixed.

    Evaluation intentionally executes one explicit case at a time. The requested
    parallel count is recorded by the caller; it is not an evaluation sample size.
    """
    if environment is not None and preset is not None:
        raise ValueError("Choose an evaluation config or a contact preset, not both")
    original = recorded_environment_config(training)
    config = EnvConfig(**environment) if environment is not None else original
    if preset is not None:
        config = with_contact_model(original, preset)
    for key in ("physics_dt", "control_dt", "episode_steps", "cost_reference", "target_rate"):
        if getattr(config, key) != getattr(original, key):
            raise ValueError(f"Evaluation must preserve the policy task/action contract: {key}")
    return replace(config, num_envs=1)


def validate_policy_environment(training, current, *, allow_scene_transfer=False):
    keys = ("observation_profile", "observation_size", "observation_frame", "quaternion_order",
            "observation_scales", "action", "actuator_names", "joint_ranges", "joint_qpos_indices",
            "joint_qvel_indices", "scene_xml_sha256", "reward_profile", "cost_weights", "task_protocol")
    if 'scene_model_fingerprint' in training:
        keys += ('scene_model_fingerprint',)
    for key in keys:
        if allow_scene_transfer and key in ("scene_xml_sha256", "scene_model_fingerprint", "cost_weights"):
            continue
        if current.get(key) != training[key]:
            raise ValueError(f"Evaluation/policy contract mismatch: {key}")


def episode_aggregates(records, allocated):
    """Keep every allocated case in the success denominator, even on interruption."""
    successes = sum(r["reason"] == "success" for r in records)
    return {"allocated_cases": allocated, "completed_cases": len(records),
        "executed_cases": len(records), "successes": successes,
        "physical_drops": sum(r["reason"] == "physical_drop" for r in records),
        "deadlines": sum(r["reason"] == "deadline" for r in records),
        "observed_successes_per_allocated": successes/allocated,
        "outcomes": {k: sum(r["reason"] == k for r in records) for k in ("success", "physical_drop", "deadline")},
        "mean_final_angle_error_deg_completed": (
            sum(r["final_angle_error_deg"] for r in records)/len(records) if records else None),
        "mean_position_error_m_completed": (
            sum(r["position_error_m"] for r in records)/len(records) if records else None)}


def checkpoint_metadata(path, *, verify=True):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    metadata = torch.load(path.with_suffix(".resume.pt"), map_location="cpu", weights_only=True)
    marker = path.parent/"checkpoint.json"
    if verify and ("progress" in metadata or "checkpoint_schema" in metadata or marker.exists()):
        if ("progress" not in metadata or
                metadata.get("checkpoint_schema", "leap_training_metadata_v1") != "leap_training_metadata_v1"):
            raise ValueError("Invalid checkpoint metadata schema/progress")
        progress = metadata["progress"]
        if (not isinstance(progress, dict) or set(progress) != {"completed_ppo_updates", "vector_steps", "transitions"}
                or any(v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0)
                       for v in progress.values())):
            raise ValueError("Invalid checkpoint progress counters")
        if not marker.exists(): raise ValueError("Incomplete checkpoint: missing completion marker")
        bundle = json.loads(marker.read_text())
        expected = {path.name, path.with_suffix(".resume.pt").name, "actor.pt", "cases.json"}
        if (bundle.get("schema") != "leap_training_bundle_v1"
                or set(bundle.get("sha256", {})) != expected
                or bundle.get("progress") != metadata["progress"]):
            raise ValueError("Invalid checkpoint bundle manifest")
        for name, digest in bundle["sha256"].items():
            if sha(path.parent/name) != digest: raise ValueError(f"Checkpoint integrity mismatch: {name}")
    return metadata


def validate_resume_request(metadata, environment, ppo, bank):
    """Reject changed settings before allocating a simulator or a new run folder."""
    if (recorded_environment_config(metadata["environment"]).to_dict() != environment.to_dict()
            or metadata["ppo"] != ppo
            or metadata["environment"]["case_bank_sha256"] != bank.sha256):
        raise ValueError("Resume requires the same PPO/environment/case-bank contract")


def save_run_checkpoint(agent, actor, env, config, directory, progress):
    """Publish a completion marker only after all four bundle files are saved.

    A partial write is retained for diagnosis but cannot be resumed by the CLI.
    Checkpoints always live in new paths; old source checkpoints are never edited.
    """
    from ContactModelStudy.RL.SkrlPPO import save_training_checkpoint
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    names = ("training.pt", "training.resume.pt", "actor.pt", "cases.json", "checkpoint.json")
    if any((directory/n).exists() for n in names if n != "cases.json"):
        raise FileExistsError("Refusing to overwrite an existing checkpoint bundle")
    if (directory/"cases.json").exists():
        from ContactModelStudy.RL.CaseBank import CaseBank
        if CaseBank.load(directory/"cases.json").sha256 != env.case_bank.sha256:
            raise ValueError("Checkpoint case bank differs from the existing run bank")
    else:
        env.case_bank.save(directory/"cases.json")
    save_training_checkpoint(agent, actor, env, config, directory/"training.pt", progress=progress)
    save_json(directory/"checkpoint.json", {"schema": "leap_training_bundle_v1", "progress": progress,
        "resume_mode": "reset_all_episodes", "sha256": {n: sha(directory/n) for n in names[:-1]}})
    return directory/"training.pt"
