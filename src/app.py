import asyncio
import ctypes
from ctypes import CDLL
import uuid
import logging
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
import inspect

from src import Logger, setup_logger

if not Logger.root:
    Logger.root = setup_logger("ApplicationBus", logging.DEBUG, "%Y-%m-%d %H:%M:%S", [logging.StreamHandler()])

"""
We can assume that imperative deterministic source code, such as this file written in Python, is capable of reasoning about non-imperative non-deterministic source code as if it were a defined and known quantity. This is akin to nesting a function with a value in an S-Expression.

In order to expect any runtime result, we must assume that a source code configuration exists which will yield that result given the input.

The source code configuration is the set of all possible configurations of the source code. It is the union of the possible configurations of the source code.

This source code file can reason about or make assumptions about dynamical source code as if it were a defined and known quantity, like nesting a function with a value in an S-Expression.

Imperative programming specifies how to perform tasks (like procedural code), while non-imperative (e.g., functional programming in LISP) focuses on what to compute. We turn this on its head in our imperative non-imperative runtime by utilizing nominative homoiconistic reflection to create a runtime where dynamical source code is treated as both static and dynamic.

"Nesting a function with a value in an S-Expression":
    In the code, we nest the input value within different function expressions (configurations).
    Each function is applied to the input to yield results, mirroring the collapse of the wave function to a specific state upon measurement.

This nominative homoiconistic reflection combines the expressiveness of S-Expressions with the operational semantics of Python. In this paradigm, source code can be constructed, deconstructed, and analyzed in real-time, allowing for dynamic composition and execution. Each code configuration (or state) is akin to a function in an S-Expression that can be encapsulated, manipulated, and ultimately evaluated in the course of execution.

To illustrate, consider a Python function as a generalized S-Expression. This function can take other functions and values as arguments, forming a nested structure. Each invocation changes the system's state temporarily, just as evaluating an S-Expression alters the state of the LISP interpreter.

In essence, our approach ensures that:

1. **Composition**: Functions (or code segments) can be composed at runtime, akin to how S-Expressions can nest functions and values.
2. **Evaluation**: Upon invocation, these compositions are evaluated, reflecting the current configuration of the runtime.
3. **Reflection and Modification**: The runtime can reflect on its structure and make modifications dynamically, which allows it to reason about its state and adapt accordingly.

This synthesis of static and dynamic code concepts is akin to the Copenhagen interpretation of quantum mechanics, where the observation (or execution) collapses the superposition of states (or configurations) into a definite outcome based on the input.

Ultimately, this model provides a flexible approach to managing and executing complex code structures dynamically while maintaining the clarity and compositional advantages traditionally seen in non-imperative, functional paradigms like LISP, drawing inspiration from lambda calculus and functional programming principles.

The most advanced concept of all in this ontology is the dynamic rewriting of source code at runtime. Source code rewriting is achieved with a special runtime `Atom()` class with 'modified quine' behavior. This special Atom, aside from its specific function and the functions obligated to it by polymorphism, will always rewrite its own source code but may also perform other actions as defined by the source code in the runtime which invoked it. They can be nested in S-expressions and are homoiconic with all other source code. These modified quines can be used to dynamically create new code at runtime, which can be used to extend the source code in a way that is not known at the start of the program. This is the most powerful feature of the system and allows for the creation of a runtime of runtimes dynamically limited by hardware and the operating system.
"""
# Decorator for logging function execution
def log(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            Logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = await func(*args, **kwargs)
                Logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                Logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            Logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                Logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                Logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Core Atoms and related classes
class AtomType(Enum):
    VALUE = auto()
    FUNCTION = auto() # functions are callables
    CLASS = auto() # classes, aka types ['T']
    MODULE = auto() # modules are SimpleNamespace objects and/or actual modules

class Atom(ABC):
    def __init__(self, tag: str, value: Any = None, children: List['Atom'] = None, metadata: Dict[str, Any] = None, **attributes):
        self.id = uuid.uuid4()
        self.tag = tag
        self.value = value
        self.children = children or []
        self.metadata = metadata or {}
        self.attributes = attributes
        self.atom_type = self._infer_atom_type()

    def _infer_atom_type(self) -> AtomType:
        if callable(self.value):
            return AtomType.FUNCTION
        elif inspect.isclass(self.value):
            return AtomType.CLASS
        elif inspect.ismodule(self.value):
            return AtomType.MODULE
        else:
            return AtomType.VALUE

    @abstractmethod
    async def evaluate(self) -> Any:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, tag={self.tag}, value={self.value})"

class Element(Atom):
    def __post_init__(self):
        super().__init__(tag=self.__class__.__name__, value=self)

    def __setattr__(self, name, value):
        if name in self.__annotations__:
            expected_type = self.__annotations__[name]
            if not isinstance(value, expected_type):
                raise TypeError(f"Expected {expected_type} for {name}, got {type(value)}")
        super().__setattr__(name, value)

    async def evaluate(self) -> Any:
        return self

class FFILoader(Atom):
    def __init__(self, lib_path: str, **attributes):
        super().__init__(tag="FFI", **attributes)
        self.lib_path = lib_path
        self.lib = self._load_library()

    def _load_library(self):
        try:
            return CDLL(self.lib_path)
        except OSError as e:
            Logger.error(f"Failed to load shared library {self.lib_path}: {e}")
            raise

    def get_function(self, func_name: str, argtypes=None, restype=ctypes.c_int):
        try:
            func = getattr(self.lib, func_name)
            func.argtypes = argtypes or []
            func.restype = restype
            return func
        except AttributeError:
            Logger.error(f"Function {func_name} not found in {self.lib_path}")
            raise

    async def evaluate(self) -> Any:
        return self.lib

class UniversalCompiler(Atom):
    def __init__(self):
        super().__init__(tag="UniversalCompiler")
        self.ffi_loaders: Dict[str, FFILoader] = {}
        self.lock = asyncio.Lock()

    @log()
    async def load_library(self, lib_name: str, lib_path: str):
        async with self.lock:
            if lib_name not in self.ffi_loaders:
                self.ffi_loaders[lib_name] = FFILoader(lib_path)
                Logger.info(f"Loaded library: {lib_name}")

    @log()
    async def call_foreign_function(self, lib_name: str, func_name: str, *args, **kwargs):
        if lib_name not in self.ffi_loaders:
            raise ValueError(f"Library {lib_name} not loaded")

        ffi_loader = self.ffi_loaders[lib_name]
        func = ffi_loader.get_function(func_name)

        c_args = [self._convert_to_ctype(arg) for arg in args]
        result = await asyncio.get_running_loop().run_in_executor(None, func, *c_args)
        return result

    def _convert_to_ctype(self, value: Any) -> Any:
        if isinstance(value, int):
            return ctypes.c_int(value)
        elif isinstance(value, float):
            return ctypes.c_double(value)
        elif isinstance(value, str):
            return ctypes.c_char_p(value.encode('utf-8'))
        elif isinstance(value, bool):
            return ctypes.c_bool(value)
        elif isinstance(value, list):
            return (ctypes.c_int * len(value))(*value)
        return value

    async def evaluate(self) -> Any:
        return self.ffi_loaders

@dataclass
class AtomicTheory(Atom):
    theory_id: str
    local_data: Dict[str, Any] = field(default_factory=dict)
    task_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    running: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        super().__init__(tag="AtomicTheory", value=self)

    @log()
    async def submit_task(self, atom: Atom, args=(), kwargs=None) -> str:
        task_id = str(uuid.uuid4())
        await self.task_queue.put((task_id, atom, args, kwargs or {}))
        Logger.info(f"Submitted task {task_id} to theory {self.theory_id}")
        return task_id

    @log()
    async def allocate(self, key: str, value: Any) -> None:
        async with self.lock:
            self.local_data[key] = value
            Logger.info(f"Allocated {key} = {value}")

    @log()
    async def deallocate(self, key: str) -> None:
        async with self.lock:
            value = self.local_data.pop(key, None)
            Logger.info(f"Deallocated {key}, value was {value}")

    async def get(self, key: str) -> Any:
        async with self.lock:
            return self.local_data.get(key)

    @log()
    async def run(self) -> None:
        self.running = True
        asyncio.create_task(self._worker())
        Logger.info(f"{self.theory_id} is running")

    @log()
    async def stop(self) -> None:
        self.running = False
        Logger.info(f"{self.theory_id} has stopped")

    async def _worker(self) -> None:
        while self.running:
            try:
                task_id, atom, args, kwargs = await asyncio.wait_for(self.task_queue.get(), timeout=1)
                Logger.info(f"Processing task {task_id}")
                await self.allocate(f"current_task_{task_id}", atom)
                result = await atom.evaluate(*args, **kwargs)
                await self.deallocate(f"current_task_{task_id}")
                Logger.info(f"Completed task {task_id} with result: {result}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                Logger.error(f"Error in worker: {e}")

    async def evaluate(self) -> Any:
        return self.local_data

class RuntimeManager(Atom):
    _instance: Optional['RuntimeManager'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'): 
            super().__init__(tag="RuntimeManager")
            self.universal_compiler = UniversalCompiler()
            self.theories: Dict[str, AtomicTheory] = {}
            self.initialized = True

    @log()
    async def create_theory(self, theory_id: str):
        if theory_id in self.theories:
            raise ValueError(f"Theory {theory_id} already exists")
        theory = AtomicTheory(theory_id)
        self.theories[theory_id] = theory
        await theory.run()
        Logger.info(f"Created and started theory: {theory_id}")

    @log()
    async def submit_task_to_theory(self, theory_id: str, atom: Atom, *args, **kwargs):
        if theory_id not in self.theories:
            raise ValueError(f"Theory {theory_id} does not exist")
        theory = self.theories[theory_id]
        task_id = await theory.submit_task(atom, args, kwargs)
        return task_id

    @log()
    async def load_ffi_library(self, lib_name: str, lib_path: str):
        await self.universal_compiler.load_library(lib_name, lib_path)

    @log()
    async def call_ffi_function(self, lib_name: str, func_name: str, *args, **kwargs):
        return await self.universal_compiler.call_foreign_function(lib_name, func_name, *args, **kwargs)

    async def evaluate(self) -> Any:
        return self.theories

@dataclass
class FfiCallAtom(Atom):
    lib_name: str
    func_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__init__(tag="FfiCall", value=self)

    @log()
    async def evaluate(self):
        runtime_manager = RuntimeManager()  # Using singleton instance
        return await runtime_manager.call_ffi_function(self.lib_name, self.func_name, *self.args, **self.kwargs)

async def main():
    runtime_manager = RuntimeManager()

    # Step 1: Create a theory
    await runtime_manager.create_theory("ffi_theory")
    ffi_theory = runtime_manager.theories["ffi_theory"]

    # Step 2: Load the library before attempting to call any functions from it
    libc_path = '/lib/x86_64-linux-gnu/libc.so.6'  # Path to libc on most Ubuntu installations
    await runtime_manager.load_ffi_library('libc', libc_path)

    # Step 3: Call a foreign function (printf in this case)
    ffi_call_atom = FfiCallAtom(lib_name="libc", func_name="printf", args=["Hello, %s\n".encode('utf-8'), "FFI"])

    task_id = await ffi_theory.submit_task(ffi_call_atom)
    print(f"Task {task_id} submitted")

if __name__ == "__main__":
    asyncio.run(main())