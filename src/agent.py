import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.config import MethodConfig, DatasetConfig, get_api_config
from methods.base import MemoryBuildResult, AgentResponse, MemoryRetrievalResult
from utils.llm_client import get_usage_tracker

class AgentManager:
    SUPPORTED_METHODS = {'amem': ('methods.amem_agent', 'AMemAgent'), 'letta': ('methods.letta_agent', 'LettaAgent'), 'lightmem': ('methods.lightmem_agent', 'LightMemAgent'), 'mem0': ('methods.mem0_agent', 'Mem0Agent'), 'memos': ('methods.memos_agent', 'MemOSAgent'), 'memrl': ('methods.memrl_agent', 'MemRLAgent'), 'mirix': ('methods.mirix_agent', 'MIRIXAgent')}

    def __init__(self, method_config: MethodConfig, dataset_config: DatasetConfig):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.method_name = method_config.method_name.lower()
        self._api_config = get_api_config()
        self._agent = None
        self._context_id: Optional[int] = None
        self._agent_start_time = time.time()
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        matched_method = None
        for method_key in self.SUPPORTED_METHODS:
            if method_key in self.method_name:
                matched_method = method_key
                break
        if matched_method is None:
            raise ValueError(f'Unsupported memory method: {self.method_name}')
        module_path, class_name = self.SUPPORTED_METHODS[matched_method]
        self._agent = self._create_agent_instance(module_path, class_name, matched_method)

    def _create_agent_instance(self, module_path: str, class_name: str, method_key: str):
        import importlib
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        init_params = self._build_agent_params(method_key)
        return agent_class(**init_params)

    def _build_agent_params(self, method_key: str) -> Dict[str, Any]:
        model_config = self.method_config.model
        agent_params = self.method_config.agent_params or {}
        effective_max_tokens = model_config.max_completion_tokens or model_config.max_tokens
        params = {'model': model_config.name, 'temperature': model_config.temperature, 'max_tokens': effective_max_tokens, 'provider': model_config.provider, 'api_key': model_config.api_key or self._api_config.openai_api_key, 'base_url': model_config.base_url or self._api_config.openai_base_url}
        if method_key == 'mem0':
            params.update({'retrieve_num': agent_params.get('retrieve_num', 5), 'chunk_size_tokens': agent_params.get('chunk_size_tokens'), 'max_context_tokens': agent_params.get('max_context_tokens'), 'compact_local_prompts': agent_params.get('compact_local_prompts', False), 'strict_json_schema': agent_params.get('strict_json_schema', False), 'action_generation_attempts': agent_params.get('action_generation_attempts', 3), 'normalize_unknown_mutations': agent_params.get('normalize_unknown_mutations', False), 'storage_root': os.environ.get('HABITBENCH_STATE_ROOT') or agent_params.get('storage_root')})
            if self.method_config.embedding:
                params['embedding_model'] = self.method_config.embedding.model
                params['embedding_provider'] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params['embedding_model_path'] = self.method_config.embedding.model_path
        elif method_key == 'amem':
            params.update({'retrieve_num': agent_params.get('retrieve_num', 5), 'amem_backend': agent_params.get('amem_backend', 'openai'), 'amem_model': agent_params.get('amem_model', model_config.name), 'amem_embedding_model': agent_params.get('amem_embedding_model', 'all-MiniLM-L6-v2'), 'amem_evo_threshold': agent_params.get('amem_evo_threshold', 100), 'amem_max_tokens': agent_params.get('amem_max_tokens'), 'amem_max_context_tokens': agent_params.get('amem_max_context_tokens', 200000), 'amem_chunk_size_tokens': agent_params.get('amem_chunk_size_tokens')})
        elif method_key == 'letta':
            params.update({'retrieve_num': agent_params.get('retrieve_num', 5), 'context_window': agent_params.get('context_window', 128000), 'embedding_model': agent_params.get('embedding_model', 'embedding-3'), 'embedding_provider': agent_params.get('embedding_provider', 'openai'), 'embedding_dim': agent_params.get('embedding_dim', 2048), 'embedding_chunk_size': agent_params.get('embedding_chunk_size', 300), 'memory_persona': agent_params.get('memory_persona', 'I am an assistant helping with medical QA while preserving long-term memory.'), 'memory_human': agent_params.get('memory_human', 'The user is a patient in a longitudinal medical dialogue setting.'), 'memorize_chunk_tokens': agent_params.get('memorize_chunk_tokens', 6000), 'memorize_chunk_overlap_tokens': agent_params.get('memorize_chunk_overlap_tokens', 200), 'max_input_tokens': agent_params.get('max_input_tokens', 8000), 'max_question_tokens': agent_params.get('max_question_tokens', 4096), 'max_context_tokens': agent_params.get('max_context_tokens', 120000), 'enforce_memory_tool_flow': agent_params.get('enforce_memory_tool_flow', False), 'message_buffer_autoclear': agent_params.get('message_buffer_autoclear', False), 'use_native_query': agent_params.get('use_native_query', True), 'use_native_memorize': agent_params.get('use_native_memorize', True), 'native_vllm_transport': agent_params.get('native_vllm_transport', False), 'native_context_management': agent_params.get('native_context_management', False), 'system_prompt': agent_params.get('system_prompt')})
            persistence_root = agent_params.get('persistence_root')
            if not persistence_root and os.environ.get('HABITBENCH_STATE_ROOT'):
                persistence_root = str(Path(os.environ['HABITBENCH_STATE_ROOT']) / 'letta')
            if persistence_root:
                params['persistence_root'] = persistence_root
            if self.method_config.embedding:
                params['embedding_model'] = self.method_config.embedding.model
                params['embedding_provider'] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params['embedding_model_path'] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params['embedding_dim'] = self.method_config.embedding.dim
        elif method_key == 'memos':
            params.update({'retrieve_num': agent_params.get('retrieve_num', 5), 'memos_backend': agent_params.get('memos_backend', 'openai'), 'memos_model': agent_params.get('memos_model', model_config.name), 'text_mem_type': agent_params.get('text_mem_type', 'general_text'), 'embedding_dim': agent_params.get('embedding_dim'), 'observability_search': agent_params.get('observability_search', True), 'max_input_tokens': agent_params.get('max_input_tokens', 8000), 'max_question_tokens': agent_params.get('max_question_tokens', 4096), 'max_context_tokens': agent_params.get('max_context_tokens', 120000), 'storage_root': os.environ.get('HABITBENCH_STATE_ROOT') or agent_params.get('storage_root')})
            if self.method_config.embedding:
                params['embedding_model'] = self.method_config.embedding.model
                params['embedding_provider'] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params['embedding_model_path'] = self.method_config.embedding.model_path
        elif method_key == 'mirix':
            params.update({'retrieve_num': agent_params.get('retrieve_num', 5), 'memorize_chunk_tokens': agent_params.get('memorize_chunk_tokens', 2500), 'memorize_chunk_overlap_tokens': agent_params.get('memorize_chunk_overlap_tokens', 200), 'query_memory_item_tokens': agent_params.get('query_memory_item_tokens', 300), 'query_memory_context_tokens': agent_params.get('query_memory_context_tokens', 1800), 'max_input_tokens': agent_params.get('max_input_tokens', 8000), 'max_question_tokens': agent_params.get('max_question_tokens', 4096), 'max_context_tokens': agent_params.get('max_context_tokens', 120000), 'use_native_query': agent_params.get('use_native_query', True), 'enforce_single_meta_update': agent_params.get('enforce_single_meta_update', False), 'strict_internal_errors': agent_params.get('strict_internal_errors', False), 'bounded_memory_tool_schema': agent_params.get('bounded_memory_tool_schema', False), 'required_tool_choice': agent_params.get('required_tool_choice', False), 'core_json_tool_bridge': agent_params.get('core_json_tool_bridge', False), 'json_tool_bridge_attempts': agent_params.get('json_tool_bridge_attempts', 3), 'normalize_missing_update_ids': agent_params.get('normalize_missing_update_ids', False), 'memory_tool_max_items': agent_params.get('memory_tool_max_items', 4), 'memory_tool_max_string_chars': agent_params.get('memory_tool_max_string_chars', 1024), 'core_memory_tool_max_string_chars': agent_params.get('core_memory_tool_max_string_chars', 256), 'core_memory_max_tokens': agent_params.get('core_memory_max_tokens', 2048), 'memory_agent_max_tokens': agent_params.get('memory_agent_max_tokens', 10000), 'core_memory_input_max_chars': agent_params.get('core_memory_input_max_chars', 0), 'memory_agent_input_max_chars': agent_params.get('memory_agent_input_max_chars', 0)})
            if self.method_config.embedding:
                params['embedding_model'] = self.method_config.embedding.model
                params['embedding_provider'] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params['embedding_model_path'] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params['embedding_dim'] = self.method_config.embedding.dim
                if self.method_config.embedding.provider == 'local':
                    params['embedding_device'] = agent_params.get('embedding_device', None)
        elif method_key == 'memrl':
            params.update({'retrieve_num': agent_params.get('retrieve_num', 5), 'candidate_top_k': agent_params.get('candidate_top_k', 12), 'similarity_threshold': agent_params.get('similarity_threshold', 0.2), 'max_new_items_per_memorize': agent_params.get('max_new_items_per_memorize', 8), 'max_experience_tokens': agent_params.get('max_experience_tokens', 700), 'memorize_chunk_tokens': agent_params.get('memorize_chunk_tokens', 6000), 'memorize_chunk_overlap_tokens': agent_params.get('memorize_chunk_overlap_tokens', 200), 'query_memory_item_tokens': agent_params.get('query_memory_item_tokens', 300), 'query_memory_context_tokens': agent_params.get('query_memory_context_tokens', 1800), 'max_concurrent_api_calls': agent_params.get('max_concurrent_api_calls', 4), 'build_strategy': agent_params.get('build_strategy', 'proceduralization'), 'retrieve_strategy': agent_params.get('retrieve_strategy', 'query'), 'update_strategy': agent_params.get('update_strategy', 'adjustment'), 'epsilon': agent_params.get('epsilon', 0.1), 'gamma': agent_params.get('gamma', 0.0), 'learning_rate': agent_params.get('learning_rate', 0.2), 'initial_q': agent_params.get('initial_q', 0.5), 'q_init_pos': agent_params.get('q_init_pos', 0.5), 'q_init_neg': agent_params.get('q_init_neg', 0.0), 'success_reward': agent_params.get('success_reward', 1.0), 'failure_reward': agent_params.get('failure_reward', -1.0), 'weight_sim': agent_params.get('weight_sim', 0.5), 'weight_q': agent_params.get('weight_q', 0.5), 'utility_mix_lambda': agent_params.get('utility_mix_lambda', 0.6), 'max_input_tokens': agent_params.get('max_input_tokens', 8000), 'max_context_tokens': agent_params.get('max_context_tokens', 120000), 'storage_root': os.environ.get('HABITBENCH_STATE_ROOT') or agent_params.get('storage_root'), 'max_question_tokens': agent_params.get('max_question_tokens', 4096), 'embedding_model': agent_params.get('embedding_model', 'all-MiniLM-L6-v2'), 'embedding_provider': agent_params.get('embedding_provider', 'local')})
            if self.method_config.embedding:
                params['embedding_model'] = self.method_config.embedding.model
                params['embedding_provider'] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params['embedding_model_path'] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params['embedding_dim'] = self.method_config.embedding.dim
        elif method_key == 'lightmem':
            params.update({'retrieve_num': agent_params.get('retrieve_num', 5), 'pre_compress': agent_params.get('pre_compress', False), 'topic_segment': agent_params.get('topic_segment', True), 'index_strategy': agent_params.get('index_strategy', 'embedding'), 'retrieve_strategy': agent_params.get('retrieve_strategy', 'embedding'), 'update_mode': agent_params.get('update_mode', 'offline'), 'extraction_mode': agent_params.get('extraction_mode', 'flat'), 'messages_use': agent_params.get('messages_use', 'user_only'), 'lightmem_temperature': agent_params.get('lightmem_temperature', 0.1), 'lightmem_max_tokens': agent_params.get('lightmem_max_tokens', 2000), 'lightmem_top_p': agent_params.get('lightmem_top_p', 0.1), 'topic_segmenter_model_path': agent_params.get('topic_segmenter_model_path'), 'max_context_tokens': agent_params.get('max_context_tokens', 120000)})
            if self.method_config.embedding:
                params['embedding_model'] = self.method_config.embedding.model
                params['embedding_provider'] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params['embedding_model_path'] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params['embedding_dim'] = self.method_config.embedding.dim
                if self.method_config.embedding.api_key:
                    params['embedding_api_key'] = self.method_config.embedding.api_key
                if self.method_config.embedding.base_url:
                    params['embedding_base_url'] = self.method_config.embedding.base_url
        return params

    def send_message(self, message: str, memorizing: bool=False, context_id: Optional[int]=None, is_last_session: bool=False, **kwargs) -> Any:
        if context_id is not None and context_id != self._context_id:
            self._context_id = context_id
            self._agent.set_context_id(context_id)
        if memorizing:
            return self._handle_memorize(message, is_last_session=is_last_session, **kwargs)
        else:
            raise ValueError('Use retrieve() and the shared answer model for evaluation.')

    def _handle_memorize(self, message: str, is_last_session: bool=False, **kwargs) -> MemoryBuildResult:
        get_usage_tracker().set_phase('memorize')
        start_time = time.time()
        memory_kwargs = {'occurred_at': kwargs['occurred_at']} if self.method_name == 'mirix' and 'occurred_at' in kwargs else {}
        result = self._agent.memorize(message, is_last_session=is_last_session, **memory_kwargs)
        memory_time = time.time() - start_time
        if isinstance(result, MemoryBuildResult):
            result.time_cost = memory_time
            return result
        if isinstance(result, dict):
            return MemoryBuildResult(success=result.get('success', True), method=result.get('method', self.method_name), action=result.get('action', 'add_to_memory'), input_content=message, stored_content=result.get('stored_content', message), memory_entries=result.get('memory_entries', []), chunk_count=result.get('chunk_count', self._agent.memory_size), time_cost=memory_time, extra={})
        return MemoryBuildResult(success=True, method=self.method_name, action='add_to_memory', input_content=message, stored_content=message, memory_entries=[], chunk_count=self._agent.memory_size, time_cost=memory_time)

    def retrieve(self, question: str, context_id: Optional[int]=None) -> Dict[str, Any]:
        """Retrieve memory context without invoking a method-specific reader."""
        if context_id is not None and context_id != self._context_id:
            self.set_context_id(context_id)
        get_usage_tracker().set_phase('query')
        result = self._agent.retrieve(question)
        if not isinstance(result, MemoryRetrievalResult):
            raise TypeError(f'{type(self._agent).__name__}.retrieve() returned {type(result).__name__}, expected MemoryRetrievalResult')
        return result.to_dict()

    def reset(self) -> None:
        if self._agent:
            self._agent.reset()
        self._context_id = None
        self._agent_start_time = time.time()

    def set_context_id(self, context_id: int) -> None:
        self._context_id = context_id
        if self._agent:
            self._agent.set_context_id(context_id)

    def get_info(self) -> Dict[str, Any]:
        info = {'method_name': self.method_name, 'context_id': self._context_id}
        if self._agent:
            info.update(self._agent.get_info())
        return info

def create_agent_manager(method_config: MethodConfig, dataset_config: DatasetConfig) -> AgentManager:
    return AgentManager(method_config, dataset_config)

def list_available_methods() -> List[str]:
    return list(AgentManager.SUPPORTED_METHODS.keys())
