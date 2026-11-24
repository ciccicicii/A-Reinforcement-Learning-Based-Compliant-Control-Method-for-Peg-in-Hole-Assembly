"""两轮实验的配对比较工具。

用途：对比两次运行（如同一脚本、只改一个参数）的逐步 CSV，检查
    1. 两轮是否**真的配上了**（同回合号的初始位姿是否一致）；
    2. 逐回合命运表（成功/中止/超时）与配对转移统计；
    3. 汇总指标差异。

为什么需要它：Isaac Lab 的 interval 事件（FORGE 死区每 2 s 重随机）会在
回合进行中消耗随机数，回合长度不同 → 下一回合 reset 时 RNG 状态不同 →
即便 --seed 相同，除第 0 回合外初态也会不同。不做"逐回合重播种"时，
跨轮比较实际上是在比较两个不同的初始条件样本。本工具第一条检查就是
用来暴露这个问题。

用法::

    python3 tools/compare_runs.py \
        outputs/staged_baseline_A.csv outputs/staged_baseline_B.csv \
        --label-a 3mm --label-b 4mm
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


def load(path):
    eps = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            eps[int(r["episode"])].append(r)
    return eps


def outcome(rows):
    note = rows[-1].get("note", "")
    if note == "success":
        return "OK"
    if note == "force_over_threshold":
        return "ABORT"
    return "TIMEOUT"


def init_state(rows):
    r = rows[0]
    return (float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"]))


def boundary_violations(eps):
    """检测窗口内是否发生环境自动复位（force_threshold 只在复位时重抽）。"""
    bad = []
    for ep, rows in eps.items():
        thr = [float(r["force_threshold"]) for r in rows]
        if any(abs(a - b) > 1e-6 for a, b in zip(thr, thr[1:])):
            bad.append(ep)
    return bad


def stats(eps):
    n = len(eps)
    c = {"OK": 0, "ABORT": 0, "TIMEOUT": 0}
    steps_ok, maxf = [], []
    for rows in eps.values():
        c[outcome(rows)] += 1
        if outcome(rows) == "OK":
            steps_ok.append(int(rows[-1]["step"]))
        maxf.append(max(math.hypot(float(r["F_x"]), float(r["F_y"]), float(r["F_z"])) for r in rows))
    return {
        "n": n,
        "success": c["OK"],
        "rate": 100.0 * c["OK"] / n if n else 0.0,
        "abort": c["ABORT"],
        "timeout": c["TIMEOUT"],
        "mean_steps_ok": (sum(steps_ok) / len(steps_ok)) if steps_ok else None,
        "mean_max_force": sum(maxf) / len(maxf) if maxf else 0.0,
    }


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100.0 * (centre - half) / d, 100.0 * (centre + half) / d)


def exact_mcnemar(b, c):
    """配对 McNemar 双侧精确检验（b、c 为不一致对数）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def fisher_exact(a_ok, a_bad, b_ok, b_bad):
    """2×2 双侧 Fisher 精确检验（用于非配对比较）。"""
    n = a_ok + a_bad + b_ok + b_bad
    row1 = a_ok + a_bad
    col1 = a_ok + b_ok
    if n == 0:
        return 1.0
    def prob(k):
        return math.comb(col1, k) * math.comb(n - col1, row1 - k) / math.comb(n, row1)
    lo = max(0, row1 + col1 - n)
    hi = min(row1, col1)
    p_obs = prob(a_ok)
    tail = sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= p_obs + 1e-12)
    return min(1.0, tail)


def main():
    ap = argparse.ArgumentParser(description="配对比较两轮实验 CSV。")
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--tol", type=float, default=1e-3, help="判定同一初态的坐标容差 [m]。")
    args = ap.parse_args()

    a, b = load(args.csv_a), load(args.csv_b)

    # ---------- 1. 配置与配对性检查 ----------
    refuse = []
    print("=" * 68)
    print("配置快照")
    cfgs = {}
    for tag, path in (("A", args.csv_a), ("B", args.csv_b)):
        cfg_path = path.replace(".csv", "_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfgs[tag] = json.load(f)
            cli = cfgs[tag].get("cli", {})
            print(f"  {tag}: SHA={cfgs[tag].get('script_sha256_16')}  episodes={cli.get('episodes')} seed={cli.get('seed')}")
        else:
            print(f"  {tag}: 无配置快照")
            refuse.append("缺少配置快照，无法归因")

    if len(cfgs) == 2:
        sha_a, sha_b = cfgs["A"].get("script_sha256_16"), cfgs["B"].get("script_sha256_16")
        if sha_a != sha_b:
            refuse.append(f"脚本 SHA 不同（{sha_a} vs {sha_b}），不是同一份代码")
        cli_a, cli_b = cfgs["A"].get("cli", {}), cfgs["B"].get("cli", {})
        ignore = {"csv"}  # 输出路径差异不算实验变量
        diffs = {k: (cli_a.get(k), cli_b.get(k)) for k in set(cli_a) | set(cli_b)
                 if cli_a.get(k) != cli_b.get(k) and k not in ignore}
        print(f"  CLI 差异（忽略输出路径）：{diffs if diffs else '无'}")
        if len(diffs) > 1:
            refuse.append(f"配置差异不止一项（{len(diffs)} 项），不构成单因素对照")

    # 回合集合完整性
    if len(a) != len(b):
        refuse.append(f"回合数不同（{len(a)} vs {len(b)}）")
    miss_a = sorted(set(b) - set(a))
    miss_b = sorted(set(a) - set(b))
    if miss_a or miss_b:
        refuse.append(f"回合缺失：A 缺 {miss_a}，B 缺 {miss_b}")

    # ---------- 1b. 回合边界检查 ----------
    bad_a, bad_b = boundary_violations(a), boundary_violations(b)
    print("-" * 68)
    print(f"回合边界检查：A 窗口内自动复位 {len(bad_a)} 个回合，B {len(bad_b)} 个")
    if bad_a or bad_b:
        print("  ⚠ 窗口内发生环境自动复位（force_threshold 重抽），该轮不是「一次试验=一个初始条件」。")
        print("  ⚠ 成功率是「含成功状态的窗口数」，不是尝试成功率；任何比较结论不成立。")
        print(f"  A 受影响回合：{bad_a[:10]}{' ...' if len(bad_a) > 10 else ''}")
        print(f"  B 受影响回合：{bad_b[:10]}{' ...' if len(bad_b) > 10 else ''}")
        print("  修复：脚本默认延长 episode_length_s（非 --keep_official_limit），需重跑两轮。")
        refuse.append("存在回合边界异常（窗口内自动复位）")
    else:
        print("  ✔ 两轮均为单一初始条件的连续试验。")

    common = sorted(set(a) & set(b))
    fp_a = (cfgs.get("A") or {}).get("reset_fingerprints")
    fp_b = (cfgs.get("B") or {}).get("reset_fingerprints")
    unpaired = []
    if fp_a and fp_b and len(fp_a) >= len(a) and len(fp_b) >= len(b):
        # 完整指纹比对（销/孔/末端真值、力阈值、增益、EMA、死区）
        unpaired = [ep for ep in common if fp_a[ep] != fp_b[ep]]
        mode = "完整指纹"
    else:
        # 退化为首步观测比对
        unpaired = [ep for ep in common
                    if any(abs(x - y) > args.tol for x, y in zip(init_state(a[ep]), init_state(b[ep])))]
        mode = "首步观测（无指纹快照）"
    print("-" * 68)
    print(f"配对性检查（{mode}）：共同回合 {len(common)}，初态不一致 {len(unpaired)}")
    if unpaired:
        print("  ⚠ 本轮比较不是配对比较——跨轮差异混入了初始条件差异。")
        print(f"  不一致回合：{unpaired[:10]}{' ...' if len(unpaired) > 10 else ''}")
        print("  修复方式：脚本开启逐回合重播种（默认 env.seed(seed*1000+ep)），再重跑两轮。")
        refuse.append("初态不一致，非配对比较")
    else:
        print("  ✔ 所有共同回合初态一致，可做配对比较。")

    # ---------- 2. 汇总 ----------
    sa, sb = stats(a), stats(b)
    print("-" * 68)
    print(f"{'指标':<16}{args.label_a:>14}{args.label_b:>14}")
    print(f"{'回合数':<16}{sa['n']:>14}{sb['n']:>14}")
    print(f"{'成功':<16}{sa['success']:>14}{sb['success']:>14}")
    ra, rb = wilson(sa["success"], sa["n"]), wilson(sb["success"], sb["n"])
    print(f"{'成功率':<16}{sa['rate']:>13.1f}%{sb['rate']:>13.1f}%")
    print(f"{'  95% Wilson':<16}{'[' + f'{ra[0]:.1f},{ra[1]:.1f}' + ']':>14}{'[' + f'{rb[0]:.1f},{rb[1]:.1f}' + ']':>14}")
    print(f"{'中止':<16}{sa['abort']:>14}{sb['abort']:>14}")
    print(f"{'超时':<16}{sa['timeout']:>14}{sb['timeout']:>14}")
    if sa["mean_steps_ok"] and sb["mean_steps_ok"]:
        print(f"{'成功均步数':<16}{sa['mean_steps_ok']:>14.1f}{sb['mean_steps_ok']:>14.1f}")
    print(f"{'平均峰值力(N)':<16}{sa['mean_max_force']:>14.2f}{sb['mean_max_force']:>14.2f}")

    # ---------- 3. 逐回合命运 ----------
    if common:
        print("-" * 68)
        print("逐回合命运（A → B）")
        trans = defaultdict(int)
        flips = []
        for ep in common:
            oa, ob = outcome(a[ep]), outcome(b[ep])
            trans[(oa, ob)] += 1
            if oa != ob:
                flips.append(f"EP{ep:02d}: {oa}→{ob}")
        for (oa, ob), c in sorted(trans.items(), key=lambda x: -x[1]):
            mark = "  (不一致)" if oa != ob else ""
            print(f"  {oa:<8}→ {ob:<8} {c:3d}{mark}")
        if flips:
            print("  变化回合: " + "；".join(flips))

        # McNemar（仅比较 OK / 非OK 的配对）
        b_ = sum(1 for ep in common if outcome(a[ep]) == "OK" and outcome(b[ep]) != "OK")
        c_ = sum(1 for ep in common if outcome(a[ep]) != "OK" and outcome(b[ep]) == "OK")
        print("-" * 68)
        if not unpaired and b_ + c_:
            p = exact_mcnemar(b_, c_)
            print(f"配对 McNemar（成功 vs 非成功）：{args.label_a}独成 {b_} 例，{args.label_b}独成 {c_} 例，p≈{p:.3f}")
        else:
            p = fisher_exact(sa["success"], sa["n"] - sa["success"], sb["success"], sb["n"] - sb["success"])
            print(f"非配对 Fisher 精确检验（{sa['success']}/{sa['n']} vs {sb['success']}/{sb['n']}）：p≈{p:.3f}")
            print("  注：非配对比较的统计功效低（各 30 回合），差异不可直接归因于参数。")

    # ---------- 4. 结论闸门 ----------
    print("=" * 68)
    if refuse:
        print("结论：⛔ 拒绝给出比较结论。原因：")
        for i, r in enumerate(refuse, 1):
            print(f"  {i}. {r}")
        print("=" * 68)
        return 1
    print("结论：✔ 检查全部通过，上述数字可作为单因素配对对照使用。")
    print("      仍需注意：单种子 pilot，正式结论需 ≥3 种子 × ≥30 回合。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
