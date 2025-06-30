"""
Pipeline Orchestrator using all Phase 2 Design Patterns.

This module provides a unified orchestrator that combines Strategy Pattern,
Observer Pattern, and Command Pattern to create a highly modular and
observable pipeline execution system.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .pipeline_strategy import (
    PipelineStrategy, PipelineStage, PipelineResult, StrategyFactory,
    create_default_strategies
)
from .event_system import (
    EventBus, EventType, EventPriority, EventEmitter, 
    create_default_event_system, get_observer_by_type,
    ProgressObserver, MetricsObserver
)
from .command_pattern import Command, CommandResult, CommandInvoker
from .config_builder import PipelineConfigBuilder
from .data_store import create_data_store, create_data_managers
from ..pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationContext:
    """Context for pipeline orchestration."""
    pipeline_id: str
    config: PipelineConfig
    event_bus: EventBus
    data_store: Any
    data_managers: Dict[str, Any]
    strategies: Dict[PipelineStage, PipelineStrategy]
    correlation_id: str = None
    
    def __post_init__(self):
        if self.correlation_id is None:
            self.correlation_id = str(uuid.uuid4())


class PipelineOrchestrator(EventEmitter):
    """
    Main orchestrator that coordinates pipeline execution using all design patterns.
    
    This class demonstrates the power of combining multiple design patterns:
    - Strategy Pattern: Interchangeable pipeline components
    - Observer Pattern: Comprehensive monitoring and events
    - Command Pattern: Structured operation execution
    - Factory Pattern: Model and strategy creation
    - Builder Pattern: Configuration management
    - Data Store: Centralized data management
    """
    
    def __init__(self, 
                 config: PipelineConfig = None, 
                 event_bus: EventBus = None,
                 enable_monitoring: bool = True):
        # Initialize configuration
        self.config = config or PipelineConfig()
        
        # Initialize event system
        self.event_bus = event_bus or create_default_event_system()
        super().__init__(self.event_bus, "PipelineOrchestrator")
        
        # Initialize data management
        self.data_store = create_data_store(self.config)
        self.data_managers = create_data_managers(self.data_store)
        
        # Initialize command invoker
        self.command_invoker = CommandInvoker(self.event_bus)
        
        # Monitoring
        self.enable_monitoring = enable_monitoring
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info("Pipeline Orchestrator initialized with all Phase 2 patterns")
    
    def execute_full_pipeline(self, 
                            horizons: List[int] = None,
                            models: List[str] = None,
                            max_files: int = 20,
                            max_assets: int = None,
                            strategy_overrides: Dict[PipelineStage, str] = None) -> Dict[str, Any]:
        """
        Execute the complete pipeline using the new architecture.
        
        This method showcases the integration of all design patterns.
        """
        # Generate pipeline ID and correlation ID
        pipeline_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        self.set_correlation_id(correlation_id)
        
        # Set defaults
        horizons = horizons or [7, 14, 30]
        models = models or ['linear', 'xgboost']
        strategy_overrides = strategy_overrides or {}
        
        # Create orchestration context
        strategies = create_default_strategies(
            self.config, 
            horizons=horizons, 
            models=models
        )
        
        # Apply strategy overrides
        for stage, strategy_type in strategy_overrides.items():
            strategies[stage] = StrategyFactory.create_strategy(
                stage, self.config, strategy_type,
                horizons=horizons, models=models
            )
        
        context = OrchestrationContext(
            pipeline_id=pipeline_id,
            config=self.config,
            event_bus=self.event_bus,
            data_store=self.data_store,
            data_managers=self.data_managers,
            strategies=strategies,
            correlation_id=correlation_id
        )
        
        # Start pipeline execution
        return self._execute_pipeline_with_context(context, max_files, max_assets)
    
    def _execute_pipeline_with_context(self, 
                                     context: OrchestrationContext,
                                     max_files: int,
                                     max_assets: int) -> Dict[str, Any]:
        """Execute pipeline with full context and monitoring."""
        start_time = time.time()
        
        # Emit pipeline started event
        self.emit_event(
            EventType.PIPELINE_STARTED,
            data={
                'pipeline_id': context.pipeline_id,
                'correlation_id': context.correlation_id,
                'max_files': max_files,
                'max_assets': max_assets
            },
            priority=EventPriority.HIGH
        )
        
        execution_log = {
            'pipeline_id': context.pipeline_id,
            'started_at': datetime.now(),
            'stages': {},
            'success': False,
            'error': None
        }
        
        try:
            # Stage 1: Data Loading
            data = self._execute_stage(
                context, PipelineStage.DATA_LOADING,
                input_data=None, max_files=max_files
            )
            execution_log['stages']['data_loading'] = {
                'success': data is not None,
                'assets_loaded': len(data) if data else 0
            }
            
            if not data:
                raise Exception("Data loading failed - no data returned")
            
            # Stage 2: Data Processing
            processed_data = self._execute_stage(
                context, PipelineStage.DATA_PROCESSING,
                input_data=data
            )
            execution_log['stages']['data_processing'] = {
                'success': processed_data is not None,
                'assets_processed': len(processed_data) if processed_data else 0
            }
            
            if not processed_data:
                raise Exception("Data processing failed - no processed data returned")
            
            # Stage 3: Feature Engineering
            featured_data = self._execute_stage(
                context, PipelineStage.FEATURE_ENGINEERING,
                input_data=processed_data
            )
            execution_log['stages']['feature_engineering'] = {
                'success': featured_data is not None,
                'assets_featured': len(featured_data) if featured_data else 0
            }
            
            if not featured_data:
                raise Exception("Feature engineering failed - no featured data returned")
            
            # Limit assets if specified
            if max_assets and len(featured_data) > max_assets:
                limited_data = dict(list(featured_data.items())[:max_assets])
                featured_data = limited_data
                logger.info(f"Limited to {max_assets} assets for training")
            
            # Stage 4: Model Training
            trained_models = self._execute_stage(
                context, PipelineStage.MODEL_TRAINING,
                input_data=featured_data, max_assets=max_assets
            )
            execution_log['stages']['model_training'] = {
                'success': trained_models is not None,
                'models_trained': len(trained_models) if trained_models else 0
            }
            
            # Stage 5: Visualization (optional)
            visualizations = None
            if trained_models:
                viz_input = {
                    'trained_models': trained_models,
                    'data': featured_data
                }
                visualizations = self._execute_stage(
                    context, PipelineStage.VISUALIZATION,
                    input_data=viz_input, max_assets=max_assets or 5
                )
                execution_log['stages']['visualization'] = {
                    'success': visualizations is not None,
                    'visualizations_created': len(visualizations) if visualizations else 0
                }
            
            # Save pipeline results
            self._save_pipeline_results(context, {
                'raw_data': data,
                'processed_data': processed_data,
                'featured_data': featured_data,
                'trained_models': trained_models,
                'visualizations': visualizations
            })
            
            execution_time = time.time() - start_time
            execution_log.update({
                'success': True,
                'completed_at': datetime.now(),
                'total_execution_time': execution_time
            })
            
            # Emit pipeline completed event
            self.emit_event(
                EventType.PIPELINE_COMPLETED,
                data={
                    'pipeline_id': context.pipeline_id,
                    'total_execution_time': execution_time,
                    'stages_completed': len(execution_log['stages'])
                },
                priority=EventPriority.HIGH
            )
            
            # Get final metrics
            metrics_observer = get_observer_by_type(self.event_bus, MetricsObserver)
            progress_observer = get_observer_by_type(self.event_bus, ProgressObserver)
            
            final_result = {
                'pipeline_id': context.pipeline_id,
                'success': True,
                'execution_log': execution_log,
                'results': {
                    'data_assets': len(data) if data else 0,
                    'processed_assets': len(processed_data) if processed_data else 0,
                    'featured_assets': len(featured_data) if featured_data else 0,
                    'trained_models': len(trained_models) if trained_models else 0,
                    'visualizations': len(visualizations) if visualizations else 0
                },
                'metrics': metrics_observer.get_metrics() if metrics_observer else {},
                'progress': progress_observer.get_progress() if progress_observer else {}
            }
            
            self.execution_history.append(final_result)
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            execution_log.update({
                'success': False,
                'failed_at': datetime.now(),
                'total_execution_time': execution_time,
                'error': error_msg
            })
            
            # Emit pipeline failed event
            self.emit_event(
                EventType.PIPELINE_FAILED,
                data={
                    'pipeline_id': context.pipeline_id,
                    'error': error_msg,
                    'execution_time': execution_time
                },
                priority=EventPriority.CRITICAL
            )
            
            logger.error(f"Pipeline execution failed: {error_msg}")
            
            error_result = {
                'pipeline_id': context.pipeline_id,
                'success': False,
                'execution_log': execution_log,
                'error': error_msg
            }
            
            self.execution_history.append(error_result)
            return error_result
    
    def _execute_stage(self, 
                      context: OrchestrationContext,
                      stage: PipelineStage,
                      input_data: Any,
                      **kwargs) -> Any:
        """Execute a single pipeline stage using the Strategy pattern."""
        strategy = context.strategies.get(stage)
        if not strategy:
            raise Exception(f"No strategy found for stage {stage}")
        
        # Emit stage started event
        self.emit_event(
            EventType.STAGE_STARTED,
            data={
                'strategy_name': strategy.get_name(),
                'input_data_type': type(input_data).__name__ if input_data else 'None'
            },
            stage=stage.value
        )
        
        try:
            # Validate input
            if input_data is not None and not strategy.validate_input(input_data):
                raise Exception(f"Invalid input data for stage {stage}")
            
            # Execute strategy
            result = strategy.execute(input_data, **kwargs)
            
            if result.success:
                # Emit stage completed event
                self.emit_event(
                    EventType.STAGE_COMPLETED,
                    data={
                        'strategy_name': strategy.get_name(),
                        'execution_time': result.execution_time,
                        'metadata': result.metadata
                    },
                    stage=stage.value
                )
                
                return result.data
            else:
                # Emit stage failed event
                self.emit_event(
                    EventType.STAGE_FAILED,
                    data={
                        'strategy_name': strategy.get_name(),
                        'error': result.error,
                        'execution_time': result.execution_time
                    },
                    priority=EventPriority.HIGH,
                    stage=stage.value
                )
                
                raise Exception(f"Stage {stage} failed: {result.error}")
                
        except Exception as e:
            # Emit stage failed event for unexpected errors
            self.emit_event(
                EventType.STAGE_FAILED,
                data={
                    'strategy_name': strategy.get_name(),
                    'error': str(e)
                },
                priority=EventPriority.HIGH,
                stage=stage.value
            )
            
            raise
    
    def _save_pipeline_results(self, 
                              context: OrchestrationContext,
                              results: Dict[str, Any]) -> None:
        """Save pipeline results using the Data Store pattern."""
        try:
            # Save different types of data using appropriate managers
            scraped_manager = context.data_managers['scraped']
            processed_manager = context.data_managers['processed']
            models_manager = context.data_managers['models']
            
            if results.get('processed_data'):
                processed_manager.save_processed_data(
                    results['processed_data'], 
                    stage=f"pipeline_{context.pipeline_id}"
                )
            
            if results.get('featured_data'):
                processed_manager.save_feature_data(
                    results['featured_data'],
                    feature_set=f"pipeline_{context.pipeline_id}"
                )
            
            # Save a pipeline summary
            summary = {
                'pipeline_id': context.pipeline_id,
                'correlation_id': context.correlation_id,
                'executed_at': datetime.now().isoformat(),
                'config': context.config.__dict__,
                'results_summary': {
                    'stages_completed': len([k for k, v in results.items() if v is not None]),
                    'data_assets': len(results.get('processed_data', {})),
                    'featured_assets': len(results.get('featured_data', {})),
                    'trained_models': len(results.get('trained_models', {}))
                }
            }
            
            context.data_store.save(
                f"pipeline_summary_{context.pipeline_id}",
                summary,
                format='json'
            )
            
            self.emit_event(
                EventType.DATA_SAVED,
                data={'pipeline_id': context.pipeline_id, 'summary': summary}
            )
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {str(e)}")
            self.emit_event(
                EventType.ERROR,
                data={'error': f"Failed to save results: {str(e)}"}
            )
    
    def execute_single_stage(self, 
                            stage: PipelineStage,
                            input_data: Any = None,
                            strategy_type: str = 'default',
                            **kwargs) -> PipelineResult:
        """Execute a single pipeline stage."""
        try:
            strategy = StrategyFactory.create_strategy(
                stage, self.config, strategy_type, **kwargs
            )
            
            return strategy.execute(input_data, **kwargs)
            
        except Exception as e:
            logger.error(f"Single stage execution failed: {str(e)}")
            return PipelineResult(
                stage=stage,
                success=False,
                data=None,
                metadata={'error': str(e)},
                execution_time=0.0,
                error=str(e)
            )
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get pipeline execution history."""
        return self.execution_history.copy()
    
    def get_latest_execution(self) -> Optional[Dict[str, Any]]:
        """Get the latest pipeline execution."""
        return self.execution_history[-1] if self.execution_history else None
    
    def create_command_from_args(self, args) -> Command:
        """Create a command from CLI arguments using the Command pattern."""
        from .command_pattern import CommandFactory
        
        if args.command == 'scrape':
            return CommandFactory.create_scrape_command(args.scrape_command)
        elif args.command == 'pipeline':
            return CommandFactory.create_pipeline_command(args)
        elif args.command == 'train':
            return CommandFactory.create_training_command(args)
        elif args.command == 'serve':
            return CommandFactory.create_serve_command(args.serve_type)
        elif args.command == 'visualize':
            return CommandFactory.create_visualize_command(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    
    def execute_command(self, command: Command, **kwargs) -> CommandResult:
        """Execute a command using the Command pattern."""
        return self.command_invoker.execute_command(command, **kwargs)


def create_orchestrator(config: PipelineConfig = None) -> PipelineOrchestrator:
    """Factory function to create a pipeline orchestrator."""
    return PipelineOrchestrator(config)


def create_orchestrator_from_config_dict(config_dict: Dict[str, Any]) -> PipelineOrchestrator:
    """Create orchestrator from configuration dictionary."""
    config = (PipelineConfigBuilder()
              .from_dict(config_dict)
              .build())
    
    return PipelineOrchestrator(config)