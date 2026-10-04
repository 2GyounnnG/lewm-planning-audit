"""Native pinned SWM history comparison. No training or solver modifications."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from r4.common import STREAMS, atomic_json, atomic_npz, digest, file_record, replan_seed, verify_complete
from r4.export_cases import CaseWindows
from r3 import planning as old_helpers
from r3.common import fp32_policy
from r3.data import action_processor, image_transform
from check_module_a_strict import build_model
import stable_worldmodel as swm
from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE
from stable_worldmodel.planning.solver.cem import CEMSolver
from stable_worldmodel.policy import WorldModelPolicy


def array(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_case(model, receipt, entry, assets, folder, stream, history_len, phase, reference=None):
    case = dict(entry["case"])
    if phase == "FORMAL":
        if case["role"] != "R6_FRESH":
            raise ValueError("Formal Module A requires the fixed R6 fresh cases")
        case["role"] = "EVAL"
    old_helpers.validate_case("reacher", case, phase)
    identity = {
        "version": "R10_NATIVE_A_V1", "case": case, "phase": phase,
        "stream": stream, "history_len": history_len, "arm": "H0", "model": receipt,
        "plan": {**old_helpers.PLAN_CONFIG, "history_len": history_len},
        "cem": old_helpers.CEM_CONFIG, "budget_raw": 50,
        "asset": {k: entry[k] for k in ("bytes", "sha256")},
        "code": {name: file_record(Path(__file__).parent / name) for name in ("module_a_runner.py", "check_module_a_strict.py")},
        "optimizer_updates": 0,
    }
    folder.mkdir(parents=True, exist_ok=True)
    identity_sha = digest(identity)
    saved = verify_complete(folder, identity_sha)
    if saved:
        return saved
    if (folder / "STARTED.json").exists():
        raise RuntimeError("Unfinished attempt exists; no implicit retry")
    atomic_json(folder / "STARTED.json", {"identity": identity, "identity_sha256": identity_sha})
    actions, observations, plans, probes = [], [], [], []
    scaler, transform = action_processor("reacher"), image_transform()
    current_probe = None

    class Cost(ShootingCostEvaluator):
        def get_cost(self, info, candidates):
            nonlocal current_probe
            if "model_pixels_shape" not in current_probe:
                current_probe.update(
                    model_pixels_shape=list(info["pixels"].shape),
                    model_action_history_shape=list(info["action_history"].shape) if "action_history" in info else None,
                    candidate_shape=list(candidates.shape),
                )
                assert tuple(candidates.shape) == (1, 300, 5, 10)
                if len(actions) == 0:
                    assert info["pixels"].shape[2] == 1
                    assert "action_history" not in info
                    current_probe["warmup"] = "ONE_REAL_FRAME_NO_ACTION_HISTORY"
                elif history_len == 3:
                    assert len(actions) == 25
                    expected = scaler.transform(np.stack(actions[15:25])).reshape(2, 10).astype(np.float32)
                    actual = array(info["action_history"])[0, 0]
                    assert np.array_equal(expected, actual), "Executed action history mismatch"
                    indices = [15, 20, 25]
                    pixels = torch.stack([transform(torch.from_numpy(observations[i].transpose(2, 0, 1).copy())) for i in indices])
                    assert torch.equal(pixels, info["pixels"][0, 0].cpu()), "Real frame history mismatch"
                    current_probe.update(history_raw_indices=indices, real_actions_verified=True, real_pixels_verified=True)
            value = super().get_cost(info, candidates)
            if not torch.isfinite(value).all():
                raise old_helpers.PlanningMethodFailure("NONFINITE_MODEL_COST")
            return value

    cost = Cost(model, GoalMSE())
    solver = CEMSolver(cost=cost, device=next(model.parameters()).device,
                       seed=replan_seed("reacher", case["case_id"], 0, stream), **old_helpers.CEM_CONFIG)
    original_solve = solver.solve

    def solve(info, init_action=None):
        nonlocal current_probe
        index = len(probes)
        seed = replan_seed("reacher", case["case_id"], index, stream)
        solver.torch_gen.manual_seed(seed)
        current_probe = {"replan_index": index, "anchor_raw": len(actions), "seed_uint64": seed}
        probes.append(current_probe)
        output = original_solve(info, init_action=init_action)
        if not torch.isfinite(output["actions"]).all():
            raise old_helpers.PlanningMethodFailure("NONFINITE_PLAN")
        plan = array(output["actions"])[0]
        plans.append(plan)
        current_probe["plan_sha256"] = hashlib.sha256(plan.tobytes()).hexdigest()
        if index == 0 and reference is not None:
            equal = plan.dtype == reference.dtype and plan.tobytes() == reference.tobytes()
            current_probe["first_plan_bitwise_equal"] = equal
            current_probe["first_plan_max_abs_difference"] = float(np.max(np.abs(plan - reference)))
            if not equal:
                raise RuntimeError("NATIVE_HISTORY_FIRST_PLAN_GATE_FAILED")
        return output

    solver.solve = solve
    policy = WorldModelPolicy(solver=solver, config=swm.PlanConfig(**old_helpers.PLAN_CONFIG, history_len=history_len),
                              process={"action": scaler}, transform={"pixels": transform, "goal": transform})

    class ObservedPolicy(old_helpers.IsolatedPolicy):
        def get_action(self, info, **kwargs):
            observations.append(array(info["pixels"][0, -1]))
            return super().get_action(info, **kwargs)

    world = None
    started = time.monotonic()
    method_failure = None
    metrics = None
    try:
        dataset = CaseWindows(assets, entry)
        view = old_helpers.CaseDatasetView(dataset, "reacher", case)
        world = swm.World(env_name="swm/ReacherDMControl-v0", num_envs=1, max_episode_steps=100, image_shape=(224, 224), task="qpos_match")
        world.set_policy(ObservedPolicy(policy))
        original_step = world.envs.step

        def step(value, *args, **kwargs):
            result = original_step(value, *args, **kwargs)
            actions.append(array(value[0]))
            return result

        world.envs.step = step
        with torch.inference_mode():
            metrics = world.evaluate(dataset=view, episodes_idx=[case["source_episode_idx"]],
                                     start_steps=[case["start_raw_index"]], goal_offset=25, eval_budget=50,
                                     callables=old_helpers.CALLABLES["reacher"], video=None)
    except old_helpers.PlanningMethodFailure as exc:
        method_failure = str(exc)
    except BaseException as exc:
        atomic_json(folder / "FAILURE.json", {"status": "TECHNICAL_MISSING", "error_type": type(exc).__name__, "error": str(exc), "replans": probes})
        raise
    finally:
        if world is not None:
            world.close()
    result = {"status": "COMPLETE", "identity_sha256": identity_sha, "case_id": case["case_id"], "arm": "H0",
              "phase": phase, "stream": stream, "history_len": history_len,
              "success": int(bool(metrics is not None and metrics["episode_successes"][0])),
              "method_failure": method_failure, "replans": probes, "entered_replanning": len(probes) > 1,
              "executed_raw_steps": len(actions), "wall_seconds": time.monotonic() - started,
              "first_plan_checked": reference is not None, "optimizer_updates": 0}
    atomic_npz(folder / "trajectory.npz", raw_actions=np.asarray(actions), returned_plans_normalized=np.asarray(plans))
    atomic_json(folder / "result.json", result)
    atomic_json(folder / "COMPLETE.json", {"identity_sha256": identity_sha, "files": {name: file_record(folder / name) for name in ("trajectory.npz", "result.json")}})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True); p.add_argument("--phase", choices=["TECH", "FORMAL"], required=True)
    p.add_argument("--worker-index", type=int, default=0); p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); fp32_policy()
    entries = json.loads(Path(a.manifest).read_text())["cases"]
    if a.phase == "TECH":
        entries = [e for e in entries if e["case"]["role"] == "TECH"]
        assert len(entries) == 4
    else:
        assert len(entries) == 100
    model, receipt = build_model()
    model = model.to(a.device)
    before = model_hash(model)
    streams = STREAMS[:1] if a.phase == "TECH" else STREAMS
    for entry in entries[a.worker_index::a.workers]:
        for stream in streams:
            base = Path(a.output) / a.phase / stream / entry["case"]["case_id"] / "H0"
            one = run_case(model, receipt, entry, a.assets, base / "HISTORY1", stream, 1, a.phase)
            with np.load(base / "HISTORY1/trajectory.npz", allow_pickle=False) as saved:
                first = saved["returned_plans_normalized"][0].copy()
            three = run_case(model, receipt, entry, a.assets, base / "HISTORY3", stream, 3, a.phase, first)
            print(json.dumps({"case_id": entry["case"]["case_id"], "stream": stream, "history1": one["success"], "history3": three["success"], "replans3": three["replans"]}), flush=True)
    assert model_hash(model) == before, "Model state changed"


if __name__ == "__main__":
    main()
