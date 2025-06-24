"""
Command Line Interface for OpenGuidance AI Assistant Framework.
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import click
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core.system import OpenGuidance
from .core.config import Config, load_config
from .api.server import app


console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """OpenGuidance AI Assistant Framework CLI"""
    pass


@cli.command()
@click.option("--config", "-c", type=click.Path(), help="Configuration file path")
@click.option("--interactive", "-i", is_flag=True, help="Start interactive chat session")
@click.option("--message", "-m", type=str, help="Single message to process")
@click.option("--session-id", "-s", type=str, default="cli_session", help="Session ID")
def chat(config: Optional[str], interactive: bool, message: Optional[str], session_id: str):
    """Start a chat session with OpenGuidance."""
    
    async def run_chat():
        # Load configuration
        if config:
            guidance_config = Config.from_file(config)
        else:
            guidance_config = load_config()
        
        # Initialize system
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing OpenGuidance system...", total=None)
            
            try:
                system = OpenGuidance(guidance_config)
                await system.initialize()
                progress.update(task, description="[green]System initialized successfully!")
                
                if interactive:
                    await interactive_chat(system, session_id)
                elif message:
                    await single_message_chat(system, message, session_id)
                else:
                    console.print("[red]Error: Either --interactive or --message must be specified")
                    sys.exit(1)
                    
            except Exception as e:
                progress.update(task, description=f"[red]Initialization failed: {e}")
                console.print(f"[red]Error: {e}")
                sys.exit(1)
            finally:
                if 'system' in locals():
                    await system.cleanup()
    
    asyncio.run(run_chat())


async def interactive_chat(system: OpenGuidance, session_id: str):
    """Run interactive chat session."""
    console.print(Panel.fit(
        "[bold blue]OpenGuidance Interactive Chat[/bold blue]\n"
        "Type 'quit', 'exit', or 'bye' to end the session.\n"
        "Type 'help' for available commands.",
        title="Welcome"
    ))
    
    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("[yellow]Goodbye!")
                break
            elif user_input.lower() == 'help':
                show_chat_help()
                continue
            elif user_input.lower() == 'stats':
                show_system_stats(system)
                continue
            elif user_input.lower() == 'clear':
                console.clear()
                continue
            elif not user_input.strip():
                continue
            
            # Process the message
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                
                try:
                    result = await system.process_request(
                        user_input,
                        session_id,
                        context={"interface": "cli"}
                    )
                    
                    progress.update(task, description="[green]Response generated!")
                    console.print(f"\n[bold blue]OpenGuidance:[/bold blue] {result.content}")
                    console.print(f"[dim]({result.execution_time:.2f}s)[/dim]")
                    
                except Exception as e:
                    progress.update(task, description=f"[red]Error: {e}")
                    console.print(f"[red]Error processing request: {e}")
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Goodbye!")
            break
        except EOFError:
            console.print("\n[yellow]Session ended. Goodbye!")
            break


async def single_message_chat(system: OpenGuidance, message: str, session_id: str):
    """Process a single message."""
    try:
        result = await system.process_request(
            message,
            session_id,
            context={"interface": "cli"}
        )
        
        console.print(Panel(
            result.content,
            title="[bold blue]OpenGuidance Response[/bold blue]",
            subtitle=f"Processed in {result.execution_time:.2f}s"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}")
        sys.exit(1)


def show_chat_help():
    """Show chat help commands."""
    help_table = Table(title="Available Commands")
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")
    
    help_table.add_row("help", "Show this help message")
    help_table.add_row("stats", "Show system statistics")
    help_table.add_row("clear", "Clear the screen")
    help_table.add_row("quit/exit/bye", "End the chat session")
    
    console.print(help_table)


def show_system_stats(system: OpenGuidance):
    """Show system statistics."""
    try:
        stats = system.get_system_stats()
        
        stats_table = Table(title="System Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        for key, value in stats.items():
            if isinstance(value, float):
                value = f"{value:.2f}"
            stats_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"[red]Error getting stats: {e}")


@cli.command()
@click.option("--config", "-c", type=click.Path(), help="Configuration file path")
def config_show(config: Optional[str]):
    """Show current configuration."""
    try:
        if config:
            guidance_config = Config.from_file(config)
        else:
            guidance_config = load_config()
        
        config_dict = guidance_config.to_dict()
        
        # Remove sensitive information
        if 'api_key' in config_dict:
            config_dict['api_key'] = '***HIDDEN***'
        
        console.print(Panel(
            json.dumps(config_dict, indent=2),
            title="[bold blue]OpenGuidance Configuration[/bold blue]",
            expand=False
        ))
        
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}")
        sys.exit(1)


@cli.command()
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file path")
@click.option("--config", "-c", type=click.Path(), help="Configuration file path")
def export(output: str, config: Optional[str]):
    """Export system state to file."""
    
    async def run_export():
        try:
            # Load configuration
            if config:
                guidance_config = Config.from_file(config)
            else:
                guidance_config = load_config()
            
            # Initialize system
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Initializing system...", total=None)
                
                system = OpenGuidance(guidance_config)
                await system.initialize()
                
                progress.update(task, description="Exporting system state...")
                
                # Export system state
                export_data = await system.export_system_state()
                
                # Write to file
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                progress.update(task, description="[green]Export completed!")
                
                console.print(f"[green]System state exported to: {output_path}")
                
                await system.cleanup()
                
        except Exception as e:
            console.print(f"[red]Export failed: {e}")
            sys.exit(1)
    
    asyncio.run(run_export())


@cli.command()
@click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Input file path")
@click.option("--config", "-c", type=click.Path(), help="Configuration file path")
def import_data(input: str, config: Optional[str]):
    """Import system state from file."""
    
    async def run_import():
        try:
            # Load import data
            with open(input, 'r') as f:
                import_data = json.load(f)
            
            # Load configuration
            if config:
                guidance_config = Config.from_file(config)
            else:
                guidance_config = load_config()
            
            # Initialize system
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Initializing system...", total=None)
                
                system = OpenGuidance(guidance_config)
                await system.initialize()
                
                progress.update(task, description="Importing system state...")
                
                # Import system state
                result = await system.import_system_state(import_data)
                
                progress.update(task, description="[green]Import completed!")
                
                console.print(f"[green]System state imported successfully:")
                for key, count in result.items():
                    console.print(f"  - {key}: {count} items")
                
                await system.cleanup()
                
        except Exception as e:
            console.print(f"[red]Import failed: {e}")
            sys.exit(1)
    
    asyncio.run(run_import())


@cli.command()
@click.option("--config", "-c", type=click.Path(), help="Configuration file path")
def stats(config: Optional[str]):
    """Show system statistics."""
    
    async def run_stats():
        try:
            # Load configuration
            if config:
                guidance_config = Config.from_file(config)
            else:
                guidance_config = load_config()
            
            # Initialize system
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Initializing system...", total=None)
                
                system = OpenGuidance(guidance_config)
                await system.initialize()
                
                progress.update(task, description="Gathering statistics...")
                
                # Get system stats
                system_stats = system.get_system_stats()
                
                progress.update(task, description="[green]Statistics gathered!")
                
                # Display stats in a nice table
                stats_table = Table(title="OpenGuidance System Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="white")
                
                for key, value in system_stats.items():
                    if isinstance(value, float):
                        if key.endswith('_time'):
                            value = f"{value:.3f}s"
                        else:
                            value = f"{value:.2f}"
                    stats_table.add_row(key.replace("_", " ").title(), str(value))
                
                console.print(stats_table)
                
                await system.cleanup()
                
        except Exception as e:
            console.print(f"[red]Error getting statistics: {e}")
            sys.exit(1)
    
    asyncio.run(run_stats())


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--workers", default=1, help="Number of worker processes")
@click.option("--config", "-c", type=click.Path(), help="Configuration file path")
def server(host: str, port: int, reload: bool, workers: int, config: Optional[str]):
    """Start the OpenGuidance API server."""
    
    console.print(Panel.fit(
        f"[bold blue]Starting OpenGuidance API Server[/bold blue]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Reload: {reload}\n"
        f"Workers: {workers}",
        title="Server Configuration"
    ))
    
    try:
        uvicorn.run(
            "openguidance.api.server:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info"
        )
    except Exception as e:
        console.print(f"[red]Server failed to start: {e}")
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    console.print(Panel.fit(
        "[bold blue]OpenGuidance AI Assistant Framework[/bold blue]\n"
        "Version: 1.0.0\n"
        "Author: Nik Jois (nikjois@llamasearch.ai)\n"
        "Built with precision, deployed with confidence.",
        title="Version Information"
    ))


if __name__ == "__main__":
    cli() 