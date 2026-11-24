"""复核 08 的历史长度效应。参数为 EP00 步数、输出目录，可加 --extended 使用 41 秒时限。"""

import importlib.util
import json
from pathlib import Path
import sys


def main():
    first_steps = int(sys.argv[1])
    output = Path(sys.argv[2]).resolve()
    extended = "--extended" in sys.argv[3:]
    output.mkdir(parents=True, exist_ok=True)
    baseline = Path(__file__).resolve().parents[4] / "demos" / "08_staged_baseline.py"
    suffix = "_41s" if extended else ""
    prefix = output / f"history_{first_steps}{suffix}"
    sys.argv = [str(baseline), "--episodes", "2", "--seed", "42", "--max_steps", "600",
                "--headless", "--csv", str(prefix.with_suffix(".csv"))]
    if not extended:
        sys.argv.append("--keep_official_limit")
    spec = importlib.util.spec_from_file_location("review_baseline", baseline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = []
    make = module.gym.make

    def observe(*args, **kwargs):
        env = make(*args, **kwargs)
        raw = env.unwrapped
        manager = raw.event_manager
        index = manager.active_terms["interval"].index("dead_zone_thresholds")

        def seconds_left():
            return float(manager._interval_term_time_left[index][0])

        reset = env.reset
        step = env.step
        reward = raw._get_rewards
        ep = -1
        step_number = 0

        def observed_reset(*args, **kwargs):
            nonlocal ep, step_number
            result = reset(*args, **kwargs)
            ep += 1
            step_number = 0
            trace.append(dict(event="reset", ep=ep, seconds_left=seconds_left()))
            return result

        def observed_step(action):
            nonlocal step_number
            before = seconds_left()
            result = step(action)
            after = seconds_left()
            if after > before and ep == 1:
                trace.append(dict(event="interval", ep=ep, step=step_number,
                                  dead_zone=raw.dead_zone_thresholds[0].tolist()))
            step_number += 1
            return result

        def observed_reward():
            value = reward()
            if bool(raw.reset_buf[0]):
                success = raw._get_curr_successes(raw.cfg.task.success_threshold, check_rot=False)
                trace.append(dict(event="terminal_before_reset", ep=ep, step=step_number,
                                  success=bool(success[0]), threshold=float(raw.contact_penalty_thresholds[0]),
                                  force=raw.force_sensor_smooth[0, :3].tolist()))
            return value

        env.reset = observed_reset
        env.step = observed_step
        raw._get_rewards = observed_reward
        return env

    step_loops = 0

    def cap_first_episode(stop):
        nonlocal step_loops
        if stop == module.args_cli.max_steps:
            step_loops += 1
            if step_loops == 1:
                return range(first_steps)
        return range(stop)

    module.range = cap_first_episode
    module.gym.make = observe
    module.main()
    prefix.with_suffix(".json").write_text(json.dumps(trace, indent=2) + "\n")
    module.simulation_app.close()


if __name__ == "__main__":
    main()
