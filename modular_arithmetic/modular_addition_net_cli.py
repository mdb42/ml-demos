"""
cli.py - Neural Network Laboratory Interface

Core principle: Every interaction should be:
- Self-documenting
- Discoverable
- Logged for future reference
- Stateless (can reconstruct context from logs)
"""

import click
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
import json

# Setup logging
log_path = Path("logs")
log_path.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_path / f"lab_{datetime.now():%Y%m%d_%H%M%S}.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

console = Console()

def list_available_models():
    """Return dictionary of available models with metadata"""
    models_path = Path("models")
    if not models_path.exists():
        return {}
    
    models = {}
    for model_path in models_path.glob("*.pt"):
        # Load metadata from accompanying .json
        meta_path = model_path.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path) as f:
                models[model_path.stem] = json.load(f)
    return models

def display_lab_status():
    """Show current laboratory state"""
    models = list_available_models()
    
    table = Table(title="Available Models")
    table.add_column("ID")
    table.add_column("Architecture")
    table.add_column("Training Status")
    table.add_column("Last Modified")
    
    for model_id, meta in models.items():
        table.add_row(
            model_id,
            f"{meta['architecture']} ({meta['variant']})",
            meta['status'],
            meta['last_modified']
        )
    
    console.print(table)

def lab_greeting():
    console.print(
        "\n[bold blue]Neural Network Laboratory - Modular Arithmetic Study[/bold blue]"
        "\nSession started. Laboratory state restored."
    )
    display_lab_status()
    console.print("\nAvailable commands:")
    console.print("- inspect <model_id>  : View detailed model information")
    console.print("- train              : Initialize new model training")
    console.print("- test <model_id>    : Test model on arithmetic problems")
    console.print("- analyze <model_id> : Perform model analysis")
    console.print("- help               : Show detailed command information")
    console.print("\nEnter command or 'exit' to close laboratory session.")

@click.group()
def cli():
    """Neural Network Laboratory Interface"""
    pass

@cli.command()
@click.argument('model_id', required=False)
def inspect(model_id=None):
    """Inspect model architecture and training history"""
    if model_id:
        # Show specific model details
        pass
    else:
        # List all models with brief summaries
        pass

@cli.command()
@click.option('--architecture', type=click.Choice(['minimal', 'medium', 'full']))
@click.option('--variant', type=click.Choice(['feedforward', 'residual', 'attention']))
def train(architecture, variant):
    """Initialize new model training"""
    logging.info(f"Initiating training: {architecture} - {variant}")
    # Training logic here
    pass

@cli.command()
@click.argument('model_id')
@click.argument('operation', type=click.Choice(['add', 'multiply']))
@click.argument('a', type=int)
@click.argument('b', type=int)
def test(model_id, operation, a, b):
    """Test model on specific arithmetic problem"""
    logging.info(f"Testing {model_id}: {a} {operation} {b} mod 113")
    # Testing logic here
    pass

def main():
    console.clear()
    lab_greeting()
    
    while True:
        try:
            command = console.input("\n> ").strip()
            if command.lower() == 'exit':
                console.print("Laboratory session ended. All state preserved.")
                break
                
            # Parse and execute command
            cli.main(args=command.split(), standalone_mode=False)
            
        except click.exceptions.Exit:
            continue
        except Exception as e:
            logging.error(f"Command failed: {e}")
            console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    main()