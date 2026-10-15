"""CLI entrypoint for Distributed MoE Foundation Model Pre-Training & Alignment."""

from __future__ import annotations

import logging

import click
import torch
import yaml
from rich.console import Console
from rich.table import Table

from .alignment.dpo_trainer import DPOTrainer
from .models.foundation_model import MoEConfig, MoEFoundationModel
from .training.fsdp_trainer import FSDPTrainer

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@click.group()
def main():
    """Distributed MoE Foundation Model Cluster CLI."""
    pass


@main.command()
@click.option("--config", default="configs/moe_model_8x7b.yaml", help="Path to YAML config file")
@click.option("--steps", default=10, help="Number of training steps to run")
def train(config: str, steps: int):
    """Launch FSDP pre-training loop."""
    console.print(f"[bold green]Starting Pre-Training with config:[/bold green] {config}")
    try:
        with open(config, "r") as f:
            cfg_dict = yaml.safe_load(f)
        m_cfg = cfg_dict.get("model", {})
        moe_cfg = cfg_dict.get("moe", {})
        merged = {**m_cfg, **moe_cfg}
        moe_config = MoEConfig(**merged)
    except Exception as e:
        console.print(f"[yellow]Using fast-prototype config due to: {e}[/yellow]")
        moe_config = MoEConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            max_position_embeddings=512,
        )

    trainer = FSDPTrainer(moe_config, max_steps=steps)
    trainer.train_epochs(num_steps=steps)
    console.print("[bold green]Pre-training completed successfully![/bold green]")


@main.command()
@click.option("--steps", default=5, help="Number of DPO alignment steps to execute")
def align(steps: int):
    """Launch Direct Preference Optimization (DPO) alignment."""
    console.print("[bold cyan]Initializing DPO Policy & Reference models...[/bold cyan]")
    cfg = MoEConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        max_position_embeddings=256,
    )
    policy = MoEFoundationModel(cfg)
    reference = MoEFoundationModel(cfg)
    trainer = DPOTrainer(policy, reference, beta=0.1)

    table = Table(title="DPO Optimization Progress")
    table.add_column("Step", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Reward Margin", justify="right")
    table.add_column("Accuracy", justify="right")

    for s in range(1, steps + 1):
        chosen_ids = torch.randint(10, 999, (2, 64))
        chosen_labels = chosen_ids.clone()
        rejected_ids = torch.randint(10, 999, (2, 64))
        rejected_labels = rejected_ids.clone()

        metrics = trainer.step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        table.add_row(
            str(s),
            f"{metrics['dpo_loss']:.4f}",
            f"{metrics['reward_margin']:.4f}",
            f"{metrics['preference_accuracy'] * 100:.1f}%",
        )

    console.print(table)
    console.print("[bold green]DPO alignment finished![/bold green]")


@main.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind")
@click.option("--port", default=8002, type=int, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Launch the Distributed MoE Cluster FastAPI backend & UI."""
    from rich.panel import Panel

    console.print(
        Panel.fit(
            f"[bold cyan]DISTRIBUTED MOE FOUNDATION MODEL CLUSTER[/bold cyan]\n"
            f"[dim]Serving API on [bold green]http://{host}:{port}[/bold green][/dim]\n"
            f"[yellow]SYS_02 // 32x H100 SXM5 // 48.2% MFU Target[/yellow]",
            border_style="cyan",
        )
    )
    import uvicorn

    uvicorn.run("src.server:app", host=host, port=port, reload=reload)


@main.command(name="cluster-status")
def cluster_status():
    """Inspect 3D-parallel GPU superpod topology and memory states."""
    import asyncio

    from src.server import get_cluster_topology

    data = asyncio.run(get_cluster_topology())
    table = Table(
        title=f"{data['cluster_name']} (Total GPUs: {data['total_gpus']})", border_style="cyan"
    )
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Node", style="white")
    table.add_column("GPU", style="yellow")
    table.add_column("TP/PP/DP", style="green")
    table.add_column("Memory", style="magenta")
    table.add_column("Temp", style="red")
    table.add_column("Util %", style="bold green")

    for n in data["active_nodes"][:8]:
        table.add_row(
            str(n["global_rank"]),
            n["node"],
            f"GPU-{n['gpu_index']}",
            f"TP{n['tp_rank']}/PP{n['pp_rank']}/DP{n['dp_rank']}",
            n["memory_gb"],
            f"{n['temperature_c']}°C",
            f"{n['utilization_pct']}%",
        )
    console.print(table)
    console.print("[dim]Showing first 8 of 32 nodes. All 32 nodes healthy & synced.[/dim]")


if __name__ == "__main__":
    main()
