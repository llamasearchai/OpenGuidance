"""
Command-line interface for OpenGuidance system.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from . import (
    OpenGuidanceSystem, 
    create_basic_system, 
    create_advanced_system,
    create_development_system,
    QuickConfig,
    health_check,
    get_version_info
)


def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


async def interactive_session(system: OpenGuidanceSystem, session_id: str = "cli_session"):
    """Run an interactive session with the guidance system."""
    print("OpenGuidance Interactive Session")
    print("Type 'exit' to quit, 'help' for commands, 'stats' for statistics")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'stats':
                await print_stats(system)
                continue
            elif user_input.lower() == 'clear':
                # Clear conversation memory
                if hasattr(system, 'memory_manager'):
                    await system.memory_manager.cleanup_session(session_id)
                print("Session memory cleared.")
                continue
            elif not user_input:
                continue
            
            print("\nProcessing...", end="", flush=True)
            response = await system.process_request(user_input, session_id=session_id)
            print("\r" + " " * 15 + "\r", end="")  # Clear "Processing..." 
            
            print(f"Response (Score: {response.validation_report.overall_score:.2f}):")
            print("-" * 40)
            print(response.content)
            
            if not response.validation_report.passed:
                print(f"\n[WARNING] Validation Issues:")
                for result in response.validation_report.results:
                    if result.level.value in ['warning', 'error', 'critical']:
                        print(f"  - {result.level.value.upper()}: {result.message}")
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logging.exception("Error in interactive session")


def print_help():
    """Print help information for interactive session."""
    help_text = """
Available Commands:
  help     - Show this help message
  stats    - Show system statistics
  clear    - Clear current session memory
  exit     - Exit the session
  
Simply type your question or request and press Enter.
    """
    print(help_text)


async def print_stats(system: OpenGuidanceSystem):
    """Print system statistics."""
    try:
        stats = {}
        
        if hasattr(system, 'memory_manager'):
            memory_stats = system.memory_manager.get_memory_stats()
            stats['memory'] = memory_stats
        
        if hasattr(system, 'execution_engine'):
            exec_stats = system.execution_engine.get_execution_stats()
            stats['execution'] = exec_stats
        
        if hasattr(system, 'validation_engine'):
            validation_stats = system.validation_engine.get_validation_stats()
            stats['validation'] = validation_stats
        
        print("\nSystem Statistics:")
        print("-" * 30)
        print(json.dumps(stats, indent=2, default=str))
        
    except Exception as e:
        print(f"Error retrieving stats: {str(e)}")


async def process_single_request(
    system: OpenGuidanceSystem,
    request: str,
    session_id: str = "single_request",
    output_format: str = "text"
) -> Dict[str, Any]:
    """Process a single request and return results."""
    try:
        response = await system.process_request(request, session_id=session_id)
        
        result = {
            "request": request,
            "response": response.content,
            "validation_score": response.validation_report.overall_score,
            "validation_passed": response.validation_report.passed,
            "execution_time": response.execution_time,
            "session_id": session_id
        }
        
        if not response.validation_report.passed:
            result["validation_issues"] = [
                {
                    "level": r.level.value,
                    "message": r.message,
                    "validator": r.validator_name
                }
                for r in response.validation_report.results
                if r.level.value in ['warning', 'error', 'critical']
            ]
        
        return result
        
    except Exception as e:
        return {
            "request": request,
            "error": str(e),
            "session_id": session_id
        }


async def batch_process(
    system: OpenGuidanceSystem,
    input_file: Path,
    output_file: Optional[Path] = None
):
    """Process multiple requests from a file."""
    try:
        with open(input_file, 'r') as f:
            if input_file.suffix.lower() == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    requests = data
                else:
                    requests = data.get('requests', [])
            else:
                # Treat as text file with one request per line
                requests = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(requests)} requests...")
        results = []
        
        for i, request in enumerate(requests, 1):
            print(f"Processing request {i}/{len(requests)}...", end="\r")
            
            if isinstance(request, dict):
                req_text = request.get('text', str(request))
                session_id = request.get('session_id', f'batch_{i}')
            else:
                req_text = str(request)
                session_id = f'batch_{i}'
            
            result = await process_single_request(system, req_text, session_id)
            results.append(result)
        
        print(f"\nCompleted processing {len(results)} requests.")
        
        # Output results
        output_data = {
            "batch_results": results,
            "summary": {
                "total_requests": len(requests),
                "successful": len([r for r in results if 'error' not in r]),
                "failed": len([r for r in results if 'error' in r]),
                "average_validation_score": sum(r.get('validation_score', 0) for r in results) / len(results)
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"Results written to {output_file}")
        else:
            print(json.dumps(output_data, indent=2, default=str))
            
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        logging.exception("Batch processing error")


async def run_health_check():
    """Run and display system health check."""
    print("Running OpenGuidance health check...")
    health_result = await health_check()
    
    print(f"\nHealth Status: {health_result['status'].upper()}")
    print("-" * 40)
    
    if health_result['status'] == 'healthy':
        print("[SUCCESS] All components are healthy")
        
        print(f"\nComponent Status:")
        for component, status in health_result['components'].items():
            init_time = status.get('init_time', 0)
            print(f"  {component}: OK ({init_time:.3f}s)")
        
        perf = health_result.get('performance', {})
        print(f"\nPerformance:")
        print(f"  Total init time: {perf.get('total_init_time', 0):.3f}s")
        print(f"  Components tested: {perf.get('components_tested', 0)}")
        
    else:
        print("[ERROR] System is unhealthy")
        if 'error' in health_result:
            print(f"Error: {health_result['error']}")
        
        for component, status in health_result.get('components', {}).items():
            if status['status'] == 'error':
                print(f"  {component}: ERROR - {status.get('message', 'Unknown error')}")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="OpenGuidance AI Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  openguidance interactive              # Start interactive session
  openguidance request "Explain Python" # Single request
  openguidance batch input.txt          # Process batch file
  openguidance health                   # Run health check
  openguidance version                  # Show version info
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    parser.add_argument(
        '--config',
        choices=['basic', 'advanced', 'development', 'chatbot', 'code', 'content', 'research'],
        default='basic',
        help='System configuration preset'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        'interactive', 
        help='Start interactive session'
    )
    interactive_parser.add_argument(
        '--session-id',
        default='cli_interactive',
        help='Session ID for conversation memory'
    )
    
    # Single request command
    request_parser = subparsers.add_parser(
        'request',
        help='Process single request'
    )
    request_parser.add_argument(
        'text',
        help='Request text to process'
    )
    request_parser.add_argument(
        '--session-id',
        default='cli_single',
        help='Session ID for request'
    )
    request_parser.add_argument(
        '--output', '-o',
        help='Output file for results (JSON format)'
    )
    request_parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    # Batch processing command
    batch_parser = subparsers.add_parser(
        'batch',
        help='Process batch of requests from file'
    )
    batch_parser.add_argument(
        'input_file',
        type=Path,
        help='Input file with requests (JSON or text)'
    )
    batch_parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file for results'
    )
    
    # Health check command
    subparsers.add_parser(
        'health',
        help='Run system health check'
    )
    
    # Version command
    subparsers.add_parser(
        'version',
        help='Show version information'
    )
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.debug)
    
    # Handle version command early
    if args.command == 'version':
        version_info = get_version_info()
        print(json.dumps(version_info, indent=2))
        return 0
    
    # Handle health check command early
    if args.command == 'health':
        await run_health_check()
        return 0
    
    # Create system based on config
    try:
        if args.config == 'basic':
            system = create_basic_system()
        elif args.config == 'advanced':
            system = create_advanced_system()
        elif args.config == 'development':
            system = create_development_system()
        elif args.config == 'chatbot':
            config = QuickConfig.for_chatbot()
            system = OpenGuidanceSystem(config)
        elif args.config == 'code':
            config = QuickConfig.for_code_assistant()
            system = OpenGuidanceSystem(config)
        elif args.config == 'content':
            config = QuickConfig.for_content_generation()
            system = OpenGuidanceSystem(config)
        elif args.config == 'research':
            config = QuickConfig.for_research_assistant()
            system = OpenGuidanceSystem(config)
        else:
            system = create_basic_system()
        
        print(f"Initialized OpenGuidance system with '{args.config}' configuration")
        
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        if args.debug:
            logging.exception("System initialization failed")
        return 1
    
    # Handle commands
    try:
        if args.command == 'interactive' or args.command is None:
            session_id = getattr(args, 'session_id', 'cli_interactive')
            await interactive_session(system, session_id)
            
        elif args.command == 'request':
            result = await process_single_request(
                system, 
                args.text, 
                args.session_id,
                args.format
            )
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Results written to {args.output}")
            elif args.format == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                if 'error' in result:
                    print(f"Error: {result['error']}")
                    return 1
                else:
                    print(result['response'])
                    if not result['validation_passed']:
                        print(f"\n[WARNING]  Validation Score: {result['validation_score']:.2f}")
            
        elif args.command == 'batch':
            await batch_process(system, args.input_file, args.output)
        
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.debug:
            logging.exception("Command execution failed")
        return 1


def cli_main():
    """Synchronous entry point for console scripts."""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        return 130
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(cli_main())