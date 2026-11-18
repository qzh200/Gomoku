"""导出可发布的 AlphaZero 权重：只保留网络结构与 state_dict。

训练产物的 `extra`（对局棋谱）在 run_last.pt 里就有 100+ MB，推理完全用不到；
`meta` 里的完整训练配置也一并去掉，`step`/`games` 等摘要保留。

用法：python packaging/export_model.py models/az/run_last.pt dist/run_last.pt
"""

from __future__ import annotations

import argparse
import sys

import torch


SUMMARY_KEYS = ("step", "total_steps", "iteration", "games")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args(argv[1:])

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    trimmed = {
        "state": payload["state"],
        "channels": payload.get("channels"),
        "blocks": payload.get("blocks"),
        "value_hidden": payload.get("value_hidden"),
        "policy_channels": payload.get("policy_channels"),
        "value_channels": payload.get("value_channels"),
        "planes": payload.get("planes"),
        "meta": {key: meta[key] for key in SUMMARY_KEYS if key in meta},
    }
    torch.save(trimmed, args.target)
    print(
        f"{args.target}: planes={trimmed['planes']} channels={trimmed['channels']} "
        f"blocks={trimmed['blocks']} meta={trimmed['meta']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
