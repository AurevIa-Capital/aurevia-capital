"""
Command Pattern for CLI Operations.

This module implements the Command Pattern to improve CLI maintainability
by encapsulating operations as command objects with undo/redo capabilities
and better error handling.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .config_builder import PipelineConfigBuilder, create_training_config_from_args
from .model_factory import ModelFactory
from .data_store import create_data_store
from .event_system import EventBus, EventType, EventPriority, create_default_event_system

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    message: str
    data: Any = None
    execution_time: float = 0.0
    error: Optional[str] = None


class Command(ABC):
    """Abstract base class for commands."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.executed_at: Optional[datetime] = None
        self.result: Optional[CommandResult] = None
    
    @abstractmethod
    def execute(self, **kwargs) -> CommandResult:
        """Execute the command."""
        pass
    
    def can_undo(self) -> bool:
        """Check if command can be undone."""
        return False
    
    def undo(self) -> CommandResult:
        """Undo the command."""
        return CommandResult(
            success=False,
            message=f"Command {self.name} does not support undo",
            error="Undo not supported"
        )
    
    def get_name(self) -> str:
        """Get command name."""
        return self.name
    
    def get_description(self) -> str:
        """Get command description."""
        return self.description


class ScrapeCommand(Command):
    """Command for data scraping operations."""
    
    def __init__(self, scrape_type: str):
        super().__init__(
            name=f"scrape_{scrape_type}",
            description=f"Execute {scrape_type} scraping operation"
        )
        self.scrape_type = scrape_type
    
    def execute(self, **kwargs) -> CommandResult:
        """Execute scraping command."""
        start_time = time.time()
        
        try:
            if self.scrape_type == 'urls':
                logger.info("Executing URL generation command...")
                from src.collectors.watch.watch_urls import main
                main()
                message = "Watch URLs generated successfully"
                
            elif self.scrape_type == 'prices':
                logger.info("Executing price scraping command...")
                from src.collectors.watch.scrape_runner import main
                main()
                message = "Price data scraped successfully"
                
            else:
                raise ValueError(f"Unknown scrape type: {self.scrape_type}")
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                success=True,
                message=message,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Scraping command failed: {str(e)}"
            logger.error(error_msg)
            
            return CommandResult(
                success=False,
                message=error_msg,
                execution_time=execution_time,
                error=str(e)
            )


class PipelineCommand(Command):
    """Command for pipeline processing operations."""
    
    def __init__(self, config_args: Dict[str, Any]):
        super().__init__(
            name="pipeline",
            description="Execute data pipeline processing"
        )
        self.config_args = config_args
    
    def execute(self, **kwargs) -> CommandResult:
        """Execute pipeline command."""
        start_time = time.time()
        
        try:
            # Build configuration using Builder pattern
            config = (PipelineConfigBuilder()
                      .with_processing_options(
                          interpolation_method=self.config_args.get('interpolation_method', 'backfill'),
                          outlier_method=self.config_args.get('outlier_method', 'iqr'),
                          min_data_points=self.config_args.get('min_data_points', 30)
                      )
                      .build())
            
            # Create data store
            data_store = create_data_store(config)
            
            logger.info(f"Executing pipeline with max_files={self.config_args.get('max_files', 20)}")
            
            # For now, still delegate to existing pipeline
            # TODO: Use new Strategy pattern once fully implemented
            from src.pipeline.run_pipeline import main
            import sys
            
            # Build arguments for legacy interface
            pipeline_args = ['--max-files', str(self.config_args.get('max_files', 20))]
            
            if self.config_args.get('interpolation_method'):
                pipeline_args.extend(['--interpolation-method', self.config_args['interpolation_method']])
            
            if self.config_args.get('outlier_method'):
                pipeline_args.extend(['--outlier-method', self.config_args['outlier_method']])
            
            # Execute with modified sys.argv
            original_argv = sys.argv
            try:
                sys.argv = ['run_pipeline.py'] + pipeline_args
                main()
            finally:
                sys.argv = original_argv
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                success=True,
                message="Pipeline processing completed successfully",
                data={'config': config, 'data_store': data_store},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline command failed: {str(e)}"
            logger.error(error_msg)
            
            return CommandResult(
                success=False,
                message=error_msg,
                execution_time=execution_time,
                error=str(e)
            )


class TrainingCommand(Command):
    """Command for model training operations."""
    
    def __init__(self, training_config: Dict[str, Any]):
        super().__init__(
            name="train",
            description="Execute model training"
        )
        self.training_config = training_config
    
    def execute(self, **kwargs) -> CommandResult:
        """Execute training command."""
        start_time = time.time()
        
        try:
            # Validate model types using factory
            models = self.training_config.get('models', [])
            if models:
                available_models = ModelFactory.get_available_models()
                invalid_models = [m for m in models if m not in available_models]
                if invalid_models:
                    raise ValueError(f"Invalid model types: {invalid_models}. Available: {available_models}")
            
            horizons = self.training_config.get('horizons', [])
            max_assets = self.training_config.get('max_assets')
            
            logger.info(f"Executing training for models {models} with horizons {horizons}")
            
            # For now, still delegate to existing training
            # TODO: Use new Strategy pattern once fully implemented
            from src.ml.multi_horizon_training import main
            import sys
            
            # Build arguments for legacy interface
            training_args = []
            
            if horizons:
                training_args.extend(['--horizons'] + [str(h) for h in horizons])
            
            if models:
                training_args.extend(['--models'] + models)
            
            if max_assets:
                training_args.extend(['--max-assets', str(max_assets)])
            
            if kwargs.get('analyze_only', False):
                training_args.append('--analyze-only')
            
            # Execute with modified sys.argv
            original_argv = sys.argv
            try:
                sys.argv = ['multi_horizon_training.py'] + training_args
                main()
            finally:
                sys.argv = original_argv
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                success=True,
                message=f"Training completed for {len(models)} models across {len(horizons)} horizons",
                data=self.training_config,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Training command failed: {str(e)}"
            logger.error(error_msg)
            
            return CommandResult(
                success=False,
                message=error_msg,
                execution_time=execution_time,
                error=str(e)
            )


class ServeCommand(Command):
    """Command for serving operations."""
    
    def __init__(self, serve_type: str):
        super().__init__(
            name=f"serve_{serve_type}",
            description=f"Start {serve_type} service"
        )
        self.serve_type = serve_type
    
    def execute(self, **kwargs) -> CommandResult:
        """Execute serve command."""
        start_time = time.time()
        
        try:
            if self.serve_type == 'dashboard':
                logger.info("Starting Streamlit dashboard...")
                from src.scripts.run_dashboard import main
                main()
                message = "Streamlit dashboard started"
                
            elif self.serve_type == 'api':
                logger.info("Starting FastAPI server...")
                from src.scripts.run_api import main
                main()
                message = "FastAPI server started"
                
            else:
                raise ValueError(f"Unknown serve type: {self.serve_type}")
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                success=True,
                message=message,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Serve command failed: {str(e)}"
            logger.error(error_msg)
            
            return CommandResult(
                success=False,
                message=error_msg,
                execution_time=execution_time,
                error=str(e)
            )


class VisualizeCommand(Command):
    """Command for visualization operations."""
    
    def __init__(self, viz_config: Dict[str, Any]):
        super().__init__(
            name="visualize",
            description="Create visualizations"
        )
        self.viz_config = viz_config
    
    def execute(self, **kwargs) -> CommandResult:
        """Execute visualization command."""
        start_time = time.time()
        
        try:
            logger.info("Executing visualization command...")
            
            # For now, still delegate to existing visualization
            from src.ml.create_visualizations import main
            import sys
            
            # Build arguments for legacy interface
            viz_args = []
            
            if self.viz_config.get('max_assets'):
                viz_args.extend(['--max-assets', str(self.viz_config['max_assets'])])
            
            if self.viz_config.get('skip_aggregate'):
                viz_args.append('--skip-aggregate')
            
            if self.viz_config.get('verbose'):
                viz_args.append('--verbose')
            
            if self.viz_config.get('specific_assets'):
                viz_args.extend(['--specific-assets', ','.join(self.viz_config['specific_assets'])])
            
            # Execute with modified sys.argv
            original_argv = sys.argv
            try:
                sys.argv = ['create_visualizations.py'] + viz_args
                main()
            finally:
                sys.argv = original_argv
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                success=True,
                message="Visualizations created successfully",
                data=self.viz_config,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Visualization command failed: {str(e)}"
            logger.error(error_msg)
            
            return CommandResult(
                success=False,
                message=error_msg,
                execution_time=execution_time,
                error=str(e)
            )


class CompositeCommand(Command):
    """Command that executes multiple commands in sequence."""
    
    def __init__(self, name: str, commands: List[Command]):
        super().__init__(
            name=name,
            description=f"Execute {len(commands)} commands in sequence"
        )
        self.commands = commands
        self.executed_commands: List[Command] = []
    
    def execute(self, **kwargs) -> CommandResult:
        """Execute all commands in sequence."""
        start_time = time.time()
        results = []
        
        try:
            for i, command in enumerate(self.commands):
                logger.info(f"Executing command {i+1}/{len(self.commands)}: {command.get_name()}")
                
                result = command.execute(**kwargs)
                results.append(result)
                self.executed_commands.append(command)
                
                if not result.success:
                    # Stop execution on first failure
                    raise Exception(f"Command {command.get_name()} failed: {result.error}")
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                success=True,
                message=f"All {len(self.commands)} commands executed successfully",
                data={'results': results},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Composite command failed: {str(e)}"
            logger.error(error_msg)
            
            return CommandResult(
                success=False,
                message=error_msg,
                data={'results': results, 'failed_at_command': len(self.executed_commands)},
                execution_time=execution_time,
                error=str(e)
            )
    
    def can_undo(self) -> bool:
        """Check if composite command can be undone."""
        return all(cmd.can_undo() for cmd in self.executed_commands)
    
    def undo(self) -> CommandResult:
        """Undo all executed commands in reverse order."""
        if not self.can_undo():
            return CommandResult(
                success=False,
                message="Cannot undo composite command - some commands don't support undo",
                error="Undo not supported"
            )
        
        start_time = time.time()
        undo_results = []
        
        try:
            # Undo in reverse order
            for command in reversed(self.executed_commands):
                result = command.undo()
                undo_results.append(result)
                
                if not result.success:
                    raise Exception(f"Failed to undo command {command.get_name()}: {result.error}")
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                success=True,
                message=f"Successfully undid {len(self.executed_commands)} commands",
                data={'undo_results': undo_results},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Undo failed: {str(e)}"
            logger.error(error_msg)
            
            return CommandResult(
                success=False,
                message=error_msg,
                data={'undo_results': undo_results},
                execution_time=execution_time,
                error=str(e)
            )


class CommandInvoker:
    """Invoker class that executes commands with event support."""
    
    def __init__(self, event_bus: EventBus = None):
        self.event_bus = event_bus or create_default_event_system()
        self.command_history: List[Command] = []
        self.max_history_size = 100
    
    def execute_command(self, command: Command, **kwargs) -> CommandResult:
        """Execute a command with event notifications."""
        # Emit command started event
        self.event_bus.publish_event(
            event_type=EventType.STAGE_STARTED,
            source="CommandInvoker",
            data={
                'command_name': command.get_name(),
                'command_description': command.get_description()
            },
            stage=command.get_name()
        )
        
        try:
            # Execute the command
            result = command.execute(**kwargs)
            command.executed_at = datetime.now()
            command.result = result
            
            # Add to history
            self.command_history.append(command)
            if len(self.command_history) > self.max_history_size:
                self.command_history = self.command_history[-self.max_history_size:]
            
            # Emit appropriate event
            if result.success:
                self.event_bus.publish_event(
                    event_type=EventType.STAGE_COMPLETED,
                    source="CommandInvoker",
                    data={
                        'command_name': command.get_name(),
                        'execution_time': result.execution_time,
                        'message': result.message
                    },
                    stage=command.get_name()
                )
            else:
                self.event_bus.publish_event(
                    event_type=EventType.STAGE_FAILED,
                    source="CommandInvoker",
                    data={
                        'command_name': command.get_name(),
                        'execution_time': result.execution_time,
                        'error': result.error
                    },
                    priority=EventPriority.HIGH,
                    stage=command.get_name()
                )
            
            return result
            
        except Exception as e:
            # Handle unexpected errors
            error_result = CommandResult(
                success=False,
                message=f"Command {command.get_name()} failed unexpectedly",
                error=str(e)
            )
            
            self.event_bus.publish_event(
                event_type=EventType.CRITICAL_ERROR,
                source="CommandInvoker",
                data={
                    'command_name': command.get_name(),
                    'error': str(e)
                },
                priority=EventPriority.CRITICAL,
                stage=command.get_name()
            )
            
            return error_result
    
    def get_command_history(self) -> List[Command]:
        """Get command execution history."""
        return self.command_history.copy()
    
    def get_last_command(self) -> Optional[Command]:
        """Get the last executed command."""
        return self.command_history[-1] if self.command_history else None
    
    def can_undo_last(self) -> bool:
        """Check if the last command can be undone."""
        last_command = self.get_last_command()
        return last_command is not None and last_command.can_undo()
    
    def undo_last_command(self) -> CommandResult:
        """Undo the last executed command."""
        if not self.can_undo_last():
            return CommandResult(
                success=False,
                message="No undoable command available",
                error="No command to undo"
            )
        
        last_command = self.get_last_command()
        return last_command.undo()


class CommandFactory:
    """Factory for creating commands from CLI arguments."""
    
    @staticmethod
    def create_scrape_command(scrape_type: str) -> Command:
        """Create a scrape command."""
        return ScrapeCommand(scrape_type)
    
    @staticmethod
    def create_pipeline_command(args) -> Command:
        """Create a pipeline command from CLI arguments."""
        config_args = {
            'max_files': getattr(args, 'max_files', 20),
            'interpolation_method': getattr(args, 'interpolation_method', None),
            'outlier_method': getattr(args, 'outlier_method', None),
            'min_data_points': getattr(args, 'min_data_points', 30)
        }
        return PipelineCommand(config_args)
    
    @staticmethod
    def create_training_command(args) -> Command:
        """Create a training command from CLI arguments."""
        training_config = create_training_config_from_args(args)
        return TrainingCommand(training_config)
    
    @staticmethod
    def create_serve_command(serve_type: str) -> Command:
        """Create a serve command."""
        return ServeCommand(serve_type)
    
    @staticmethod
    def create_visualize_command(args) -> Command:
        """Create a visualization command from CLI arguments."""
        viz_config = {
            'max_assets': getattr(args, 'max_assets', None),
            'skip_aggregate': getattr(args, 'skip_aggregate', False),
            'verbose': getattr(args, 'verbose', False),
            'specific_assets': getattr(args, 'specific_assets', None)
        }
        return VisualizeCommand(viz_config)
    
    @staticmethod
    def create_full_pipeline_command(args) -> Command:
        """Create a composite command for full pipeline execution."""
        commands = []
        
        # Add scraping if requested
        if getattr(args, 'include_scraping', False):
            commands.append(CommandFactory.create_scrape_command('urls'))
            commands.append(CommandFactory.create_scrape_command('prices'))
        
        # Add pipeline processing
        commands.append(CommandFactory.create_pipeline_command(args))
        
        # Add training
        commands.append(CommandFactory.create_training_command(args))
        
        # Add visualization if requested
        if getattr(args, 'include_visualization', False):
            commands.append(CommandFactory.create_visualize_command(args))
        
        return CompositeCommand("full_pipeline", commands)