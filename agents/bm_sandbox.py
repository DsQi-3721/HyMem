"""
BrowseMaster Sandbox - 隔离代码执行沙箱

核心设计思想（基于BrowseMaster论文）：
1. Planner和Executor分离，Executor在独立沙箱中执行代码
2. 沙箱是Stateful的，支持多轮代码执行，变量持久化
3. 执行结果返回给Planner，但中间执行细节被隔离
"""

import os, io, re, sys, time, json, asyncio, logging, builtins, traceback, threading, uuid, ast, inspect
from typing import Dict, List, Any, Optional, Tuple, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputCapture:
    def __init__(self):
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._lock = threading.Lock()
    
    def write_stdout(self, text: str):
        with self._lock:
            self._stdout.write(text)
    
    def write_stderr(self, text: str):
        with self._lock:
            self._stderr.write(text)
    
    def get_stdout(self) -> str:
        with self._lock:
            return self._stdout.getvalue()
    
    def get_stderr(self) -> str:
        with self._lock:
            return self._stderr.getvalue()
    
    def clear(self):
        with self._lock:
            self._stdout = io.StringIO()
            self._stderr = io.StringIO()


class SandboxSession:
    """沙箱会话 - 隔离命名空间、状态持久化、工具注入、安全执行"""
    
    def __init__(self, session_id: str, tool_functions: Optional[Dict[str, Callable]] = None, timeout: int = 180):
        self.session_id = session_id
        self.timeout = timeout
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.execution_count = 0
        self.error_count = 0
        self._async_tool_names = set()
        
        self.allowed_imports = ['json', 're', 'math', 'datetime', 'time', 'random', 'collections', 'itertools', 'functools', 'asyncio']
        self.namespace = self._create_namespace()
        if tool_functions:
            self.inject_tools(tool_functions)
        self.output_capture = OutputCapture()
        self.execution_history = []
    
    def _create_namespace(self) -> Dict[str, Any]:
        import json, re, math, datetime, random, collections, itertools, functools, asyncio
        return {
            '__name__': f'sandbox_{self.session_id}',
            '__builtins__': self._create_safe_builtins(),
            'json': json, 're': re, 'math': math, 'datetime': datetime, 'random': random,
            'collections': collections, 'itertools': itertools, 'functools': functools, 'asyncio': asyncio,
        }
    
    def _create_safe_builtins(self) -> Dict[str, Any]:
        dangerous = {'eval', 'exec', 'compile', '__import__', 'open', 'input', 'breakpoint'}
        safe = {n: getattr(builtins, n) for n in dir(builtins) if not n.startswith('_') and n not in dangerous}
        safe['print'] = self._safe_print
        safe['open'] = self._restricted_open
        safe['__import__'] = self._restricted_import
        return safe
    
    def _safe_print(self, *args, **kwargs):
        self.output_capture.write_stdout(' '.join(str(a) for a in args) + kwargs.get('end', '\n'))
    
    def _restricted_open(self, *args, **kwargs):
        mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
        if any(m in mode.lower() for m in ('w', 'a', '+', 'x')):
            raise PermissionError("Write disabled")
        return builtins.open(*args, **kwargs)
    
    def _restricted_import(self, name, *args, **kwargs):
        if name.split('.')[0] not in self.allowed_imports:
            raise ImportError(f"Import of '{name}' not allowed")
        return builtins.__import__(name, *args, **kwargs)
    
    def inject_tools(self, tools: Dict[str, Callable]):
        for name, func in tools.items():
            self.namespace[name] = func
            if inspect.iscoroutinefunction(func):
                self._async_tool_names.add(name)

    def _await_unawaited_async_tools(self, code: str) -> str:
        """Add ``await`` to direct async-tool calls omitted by the model."""
        if not self._async_tool_names:
            return code

        async_tool_names = self._async_tool_names

        class _AwaitAsyncTools(ast.NodeTransformer):
            def visit_Await(self, node):
                # Do not descend into an existing await, otherwise
                # ``await web_search(...)`` becomes a double await.
                return node

            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id in async_tool_names:
                    return ast.copy_location(ast.Await(value=node), node)
                return node

        try:
            tree = ast.parse(code, mode='exec')
            tree = _AwaitAsyncTools().visit(tree)
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
        except (SyntaxError, ValueError):
            # Preserve the original code so the normal sandbox error path can
            # report malformed model output with its original source text.
            return code
    
    def update_namespace(self, variables: Dict[str, Any]):
        for name, value in variables.items():
            if not name.startswith('_'):
                self.namespace[name] = value
    
    async def execute_async(self, code: str, timeout: Optional[int] = None) -> Tuple[str, Optional[str], float]:
        timeout = timeout or self.timeout
        self.last_accessed = time.time()
        self.execution_count += 1
        self.output_capture.clear()
        self.namespace['print'] = self._safe_print
        
        start_time = time.time()
        error = None
        
        try:
            code = self._await_unawaited_async_tools(code)
            indented = '\n'.join('    ' + l if l.strip() else l for l in code.split('\n'))
            wrapped = f"async def __exec__():\n{indented}\n    return {{k:v for k,v in locals().items() if not k.startswith('_')}}"
            exec(wrapped, self.namespace)
            try:
                result = await asyncio.wait_for(self.namespace['__exec__'](), timeout=timeout)
                if result:
                    self.update_namespace(result)
            except asyncio.TimeoutError:
                error = f"Timeout after {timeout}s"
                self.error_count += 1
        except Exception as e:
            error = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.error_count += 1
        
        return self.output_capture.get_stdout(), self.output_capture.get_stderr() or error, time.time() - start_time
    
    def reset(self):
        self.namespace = self._create_namespace()
        self.output_capture.clear()


class SandboxManager:
    def __init__(self, timeout: int = 180, max_sessions: int = 100, ttl: int = 3600):
        self.timeout = timeout
        self.max_sessions = max_sessions
        self.ttl = ttl
        self.sessions: Dict[str, SandboxSession] = {}
        self._lock = threading.Lock()
    
    def create_session(self, session_id: Optional[str] = None, tools: Optional[Dict[str, Callable]] = None) -> SandboxSession:
        with self._lock:
            self._cleanup()
            if len(self.sessions) >= self.max_sessions:
                oldest = min(self.sessions.keys(), key=lambda k: self.sessions[k].last_accessed)
                del self.sessions[oldest]
            session_id = session_id or str(uuid.uuid4())
            if session_id in self.sessions:
                return self.sessions[session_id]
            session = SandboxSession(session_id, tools, self.timeout)
            self.sessions[session_id] = session
            return session
    
    def get_session(self, session_id: str) -> Optional[SandboxSession]:
        with self._lock:
            s = self.sessions.get(session_id)
            if s:
                s.last_accessed = time.time()
            return s
    
    def get_or_create(self, session_id: str, tools: Optional[Dict[str, Callable]] = None) -> SandboxSession:
        return self.get_session(session_id) or self.create_session(session_id, tools)
    
    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    def _cleanup(self):
        now = time.time()
        for sid in [s for s, sess in self.sessions.items() if now - sess.last_accessed > self.ttl]:
            del self.sessions[sid]
    
    def cleanup_all(self):
        with self._lock:
            self.sessions.clear()


class BMExecutorSandbox:
    """BrowseMaster Executor专用沙箱 - 自动注入bm_tools工具函数"""
    
    def __init__(self, tool_manager, session_id: Optional[str] = None, timeout: int = 180):
        self.tool_manager = tool_manager
        self.session_id = session_id or str(uuid.uuid4())
        self.timeout = timeout
        self.sandbox_manager = SandboxManager(timeout=timeout)
        self.session = self.sandbox_manager.create_session(self.session_id, self._build_tools())
        self.results_cache = []
    
    def _build_tools(self) -> Dict[str, Callable]:
        tools = {}
        if self.tool_manager:
            tools['web_search'] = self.tool_manager.web_search
            tools['web_parse'] = self.tool_manager.web_parse
            tools['generate_keywords'] = self.tool_manager.generate_keywords
            tools['batch_search'] = self.tool_manager.batch_search
            tools['check_condition'] = self.tool_manager.check_condition
            tools['batch_search_and_filter'] = self.tool_manager.batch_search_and_filter
            tools['parse_file'] = self.tool_manager.parse_file
        tools['save_result'] = lambda r, l="": self.results_cache.append({'label': l, 'result': r})
        tools['get_results'] = lambda: self.results_cache.copy()
        return tools
    
    def extract_code_blocks(self, text: str) -> List[str]:
        return [m.strip() for m in re.findall(r'<code>(.*?)</code>', text, re.DOTALL)]
    
    async def execute_code(self, code: str, timeout: Optional[int] = None) -> Tuple[str, Optional[str], float]:
        return await self.session.execute_async(code, timeout)
    
    async def execute_from_response(self, response: str) -> str:
        blocks = self.extract_code_blocks(response)
        if not blocks:
            return "[No code blocks found]"
        results = []
        for i, code in enumerate(blocks, 1):
            out, err, t = await self.execute_code(code)
            status = "OK" if not err else "ERROR"
            output = out or "[No output]"
            if err:
                output = f"{output}\n[Error]: {err}"
            results.append(f"=== Block {i} [{status}] ({t:.2f}s) ===\n{output}")
        return "\n\n".join(results)
    
    def reset(self):
        self.session.reset()
        self.results_cache.clear()
        self.session.inject_tools(self._build_tools())


def create_executor_sandbox(tool_manager, session_id: Optional[str] = None, timeout: int = 180) -> BMExecutorSandbox:
    return BMExecutorSandbox(tool_manager, session_id, timeout)


async def _test():
    print("Testing Sandbox...")
    mgr = SandboxManager(timeout=30)
    s = mgr.create_session("test")
    out, err, t = await s.execute_async("x=10\ny=20\nprint(f'Sum: {x+y}')")
    print(f"1. {out.strip()}, err={err}, t={t:.2f}s")
    out, err, t = await s.execute_async("print(f'x*2={x*2}')")
    print(f"2. {out.strip()}")
    mgr.cleanup_all()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(_test())
