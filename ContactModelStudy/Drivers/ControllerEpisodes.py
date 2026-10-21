"""One finite-case executor for saved RL actors and the existing MPPI."""
import json
from dataclasses import asdict
from pathlib import Path
import time
from contextlib import ExitStack
import numpy as np
import torch

from ContactModelStudy.RL.Contracts import NumericalTransitionError
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.RunIO import episode_aggregates, save_json, close_resources
from ContactModelStudy.Drivers.ContinuousGoals import prepare_goal_sequences


def run_controller_cases(config, bank, output, controller_factory, *, identity, manifest_extra=None,
                         validate_environment=None, env_factory=LeapVectorEnv, save_steps=False,
                         video=None, video_factory=None, goals=None):
    """Own reset/step/termination/accounting; controllers only produce commands."""
    if config.num_envs != 1: raise ValueError("Case evaluation is sequential and requires num_envs=1")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    bank.save(output/"cases.json")
    sequences = [[case] for case in bank.cases]
    protocol = {"name": "single_goal_finite_case_v1"}
    continuous = goals is not None and goals.enabled
    if continuous:
        sequences, payload, protocol = prepare_goal_sequences(config, bank, goals)
        save_json(output/"goal_sequences.json", payload)
    records, env, controller, active_case, stage = [], None, None, None, "initializing"
    sequence_progress = None
    recorder = None
    cleanup_errors = []
    def cleanup():
        nonlocal env, controller, recorder
        resources = (("video", recorder), ("controller", controller), ("environment", env))
        controller = env = recorder = None  # Never retry a partially failed close.
        cleanup_errors.extend(close_resources(resources))
    manifest = {**identity, **(manifest_extra or {}), "resolved_environment": config.to_dict(),
        "schema": "leap_controller_episodes_v1", "execution_layout": "one_explicit_case_at_a_time",
        "controller_timing": "wall_seconds_with_cuda_synchronization_before_and_after_act",
        "new_PPO_updates": 0, "save_steps": save_steps, "evaluation_protocol": protocol}
    if video is not None:
        manifest['video'] = asdict(video)
    save_json(output/"manifest.json", manifest)
    try:
        env = env_factory(config, bank)
        current = env.describe()
        if validate_environment: validate_environment(current)
        controller = controller_factory(env)
        if continuous and not callable(getattr(controller, 'set_goal', None)):
            raise ValueError('Continuous evaluation requires controller.set_goal')
        if video is not None:
            stage = 'video'
            if video_factory is None:
                from ContactModelStudy.Drivers.EvaluationVideo import EpisodeVideo
                video_factory = EpisodeVideo
            recorder = video_factory(video, output/'videos', env)
        stage = 'initializing'
        manifest.update(environment=current, controller=controller.describe())
        save_json(output/"manifest.json", manifest)
        with ExitStack() as stack:
            stream = stack.enter_context((output/"episodes.jsonl").open("w"))
            trace = stack.enter_context((output/"steps.jsonl").open("w")) if save_steps else None
            events = stack.enter_context((output/"goals.jsonl").open("w")) if continuous else None
            for i, case in enumerate(bank.cases):
                active_case, stage = case.case_id, "reset"
                obs = env.reset_all([i]); controller.reset(case); stage = "stepping"
                sequence, goal_index, goal_results, goal_start = sequences[i], 0, [], 0
                goal_min_angle = float('inf')
                def goal_event(kind, step):
                    if events:
                        target = sequence[goal_index]
                        events.write(json.dumps({'case_id': case.case_id, 'event': kind,
                            'goal_index': goal_index, 'control_step': step, 'sim_seconds': step*config.control_dt,
                            'goal_position': list(target.goal_position),
                            'goal_quaternion_wxyz': list(target.goal_quaternion)}, allow_nan=False)+'\n')
                        events.flush()
                if continuous:
                    sequence_progress = {'case_id': case.case_id, 'goals_reached': 0,
                                         'active_goal_index': 0, 'goals_requested': len(sequence)}
                    goal_event('activated', 0)
                if recorder is not None:
                    stage = 'video'; recorder.begin(case, i); stage = 'stepping'
                total_return, min_angle = 0., float("inf")
                minimum_height = float(obs[0, 34]/10)
                projected, decision_seconds = 0, 0.
                for k in range(config.episode_steps):
                    if torch.device(env.device).type == "cuda": torch.cuda.synchronize(env.device)
                    started = time.perf_counter()
                    command = controller.act(env, obs)
                    if torch.device(env.device).type == "cuda": torch.cuda.synchronize(env.device)
                    elapsed = time.perf_counter()-started; decision_seconds += elapsed
                    command_info = getattr(controller, "last_command_info", {})
                    projected += command_info.get("projected_components", 0)
                    tr = env.step_command(command); obs = tr.next_obs
                    total_return += float(tr.reward[0, 0])
                    angle = float(2*torch.asin(tr.diagnostics["quat"][0].clamp(0, 1).sqrt())*180/np.pi)
                    min_angle = min(min_angle, angle)
                    goal_min_angle = min(goal_min_angle, angle)
                    minimum_height = min(minimum_height, float(obs[0, 34]/10))
                    if recorder is not None:
                        stage = 'video'; recorder.capture(k+1, (k+1)*config.control_dt); stage = 'stepping'
                    if trace:
                        frame = env.read_frame()
                        extra = ({'goal_index': goal_index,
                                  'goal_quaternion_wxyz': list(sequence[goal_index].goal_quaternion),
                                  'goal_position': list(sequence[goal_index].goal_position)} if continuous else {})
                        trace.write(json.dumps({"case_id": case.case_id, "step": k, **extra,
                            "qpos_after": frame.qpos[0].tolist(), "qvel_after": frame.qvel[0].tolist(),
                            "command": command[0].tolist(), "controller_seconds": elapsed,
                            "command_info": command_info, "reason": tr.reasons[0],
                            "reward": float(tr.reward[0, 0]), "raw_cost": float(tr.diagnostics["raw_cost"][0])},
                            allow_nan=False)+"\n"); trace.flush()
                    reason = tr.reasons[0]
                    if continuous and (tr.terminated | tr.truncated)[0, 0]:
                        goal_results.append({'goal_index': goal_index, 'reason': reason,
                            'start_step': goal_start, 'end_step': k+1, 'control_steps': k+1-goal_start,
                            'goal_position': list(sequence[goal_index].goal_position),
                            'goal_quaternion_wxyz': list(sequence[goal_index].goal_quaternion),
                            'final_angle_error_deg': angle, 'min_angle_error_deg': goal_min_angle,
                            'position_error_m': float(tr.diagnostics['pos'][0])})
                        goal_event(reason, k+1)
                        if reason == 'success':
                            sequence_progress['goals_reached'] += 1
                            if goal_index+1 < len(sequence):
                                if k+1 >= config.episode_steps:
                                    reason = 'deadline'
                                    break
                                goal_index += 1
                                sequence_progress['active_goal_index'] = goal_index
                                stage = 'goal_switch'
                                obs = env.advance_evaluation_goal(sequence[goal_index])
                                controller.set_goal(sequence[goal_index])
                                goal_start, goal_min_angle = k+1, float('inf')
                                goal_event('activated', k+1)
                                if recorder is not None:
                                    stage = 'video'
                                    recorder.set_goal(sequence[goal_index], k+1, (k+1)*config.control_dt, goal_index)
                                stage = 'stepping'
                                continue
                    if (tr.terminated | tr.truncated)[0, 0]: break
                record = {"case_id": case.case_id, "reason": reason, "control_steps": k+1,
                    "domain": getattr(env, "reset_domains", {}).get(0),
                    "sim_seconds": (k+1)*config.control_dt, "return": total_return,
                    "initial_angle_error_deg": float(2*np.arccos(np.clip(abs(np.dot(case.qpos[19:23], case.goal_quaternion)), 0, 1))*180/np.pi),
                    "final_angle_error_deg": angle, "min_angle_error_deg": min_angle,
                    "position_error_m": float(tr.diagnostics["pos"][0]), "minimum_height_m": minimum_height,
                    "controller_seconds": decision_seconds, "projected_command_components": projected}
                if continuous:
                    record.update(goals_requested=len(sequence), goals_reached=sequence_progress['goals_reached'],
                        goals_attempted=len(goal_results), goals_unattempted=len(sequence)-len(goal_results),
                        sequence_complete=reason == 'success', goal_results=goal_results,
                        angle_error_scope='final_active_goal_and_minimum_across_all_attempted_goals')
                if recorder is not None:
                    stage = 'video'; record['video'] = recorder.finish(); stage = 'stepping'
                stream.write(json.dumps(record, allow_nan=False)+"\n"); stream.flush()
                records.append(record); active_case = None; sequence_progress = None
        description = controller.describe()
        stage = "cleanup"
        cleanup()
        if cleanup_errors: raise RuntimeError(cleanup_errors[0]["message"])
        stage = "reporting"
        result = {"execution_status": "completed", "purpose": "controller_evaluation_not_training",
            **episode_aggregates(records, len(bank.cases)), **identity, "environment": current,
            "controller": description, "failed_cases": 0, "not_executed": 0, "new_PPO_updates": 0,
            "evaluation_protocol": protocol}
        if continuous:
            result.update(goals_reached_completed=sum(r['goals_reached'] for r in records),
                          goals_requested_per_case=goals.goals_per_episode,
                          sequence_successes=result['successes'])
        save_json(output/"summary.json", result)
        return result
    except (Exception, KeyboardInterrupt) as exc:
        cleanup()
        reason = ("cleanup_failure" if stage == "cleanup" else
                  "numerical_failure" if isinstance(exc, NumericalTransitionError) else
                  "video_failure" if stage == 'video' else
                  "reset_invalid" if stage == "reset" else "infrastructure_interrupted")
        count = int(active_case is not None)
        result = {"execution_status": "failed", **episode_aggregates(records, len(bank.cases)),
            **identity, "failed_case_id": active_case, "failure_reason": reason,
            "executed_cases": len(records)+count, "failed_cases": count,
            "not_executed": len(bank.cases)-len(records)-count,
            "exception": type(exc).__name__, "message": str(exc), "cleanup_errors": cleanup_errors,
            "new_PPO_updates": 0, "evaluation_protocol": protocol}
        if continuous:
            result.update(goals_reached_completed=sum(r['goals_reached'] for r in records),
                          goals_requested_per_case=goals.goals_per_episode,
                          sequence_successes=result['successes'], active_sequence_progress=sequence_progress)
        result["outcomes"][reason] = count
        result["outcomes"]["not_executed"] = result["not_executed"]
        save_json(output/"summary.json", result)
        if isinstance(exc, NumericalTransitionError):
            np.savez(output/"fault_state.npz", **{k: getattr(exc.frame, k).cpu().numpy()
                for k in ("qpos", "qvel", "sites", "object_twist")})
        raise
    finally:
        cleanup()
