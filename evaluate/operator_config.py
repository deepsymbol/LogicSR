from typing import Set, List
import threading

# Note: Threading locks and global configs are preserved for future multi-threading 
# and timeout control capabilities
class OperatorConfig:
    """Logical operator configuration manager"""

    # All supported operators
    ALL_OPERATORS = {'and', 'or', 'not'}

    def __init__(self):
        self._operators = {'and', 'not'}  # Default only supports and and not
        self._lock = threading.Lock()  # Thread lock ensures safety

    def set_operators(self,
                      operators: List[str],
                      method_name: str = "Unknown") -> None:
        """Set operators for this configuration"""
        with self._lock:
            self._operators = set(operators) & self.ALL_OPERATORS
            print(
                f"{method_name} operators: {', '.join(sorted(self._operators))}"
            )

    def get_operators(self) -> Set[str]:
        """Get currently set operators"""
        with self._lock:
            return self._operators.copy()

    def has_operator(self, operator: str) -> bool:
        """Check if contains specified operator"""
        with self._lock:
            return operator in self._operators

    def has_and(self) -> bool:
        """Check if AND operator is allowed"""
        return self.has_operator('and')

    def has_or(self) -> bool:
        """Check if OR operator is allowed"""
        return self.has_operator('or')

    def has_not(self) -> bool:
        """Check if NOT operator is allowed"""
        return self.has_operator('not')

    def is_empty(self) -> bool:
        """Whether no operators are set"""
        with self._lock:
            return len(self._operators) == 0

    def get_allowed_count(self) -> int:
        """Get number of allowed operators"""
        with self._lock:
            return len(self._operators)

    def reset_to_all(self) -> None:
        """Reset to support all operators"""
        with self._lock:
            self._operators = self.ALL_OPERATORS.copy()

    def __str__(self) -> str:
        with self._lock:
            return f"OperatorConfig({sorted(self._operators)})"

    def __repr__(self) -> str:
        return self.__str__()


# Global singleton instance
_global_config = OperatorConfig()

def set_global_operators(operators: List[str],
                         method_name: str = "Global") -> None:
    """Set global operator configuration"""
    _global_config.set_operators(operators, method_name)

def get_global_operators() -> Set[str]:
    """Get global operator configuration"""
    return _global_config.get_operators()

def has_global_operator(operator: str) -> bool:
    """Check if global configuration contains specified operator"""
    return _global_config.has_operator(operator)

def has_global_and() -> bool:
    """Whether global configuration allows AND"""
    return _global_config.has_and()

def has_global_or() -> bool:
    """Whether global configuration allows OR"""
    return _global_config.has_or()

def has_global_not() -> bool:
    """Whether global configuration allows NOT"""
    return _global_config.has_not()

def get_global_config() -> OperatorConfig:
    """Get global configuration object"""
    return _global_config


# Method-level configuration management
class MethodOperatorManager:
    """Method-level operator manager"""

    def __init__(self):
        self._method_configs = {}
        self._lock = threading.Lock()

    def get_config(self, method_name: str) -> OperatorConfig:
        """Get or create operator configuration for method"""
        with self._lock:
            if method_name not in self._method_configs:
                self._method_configs[method_name] = OperatorConfig()
            return self._method_configs[method_name]

    def set_method_operators(self, method_name: str,
                             operators: List[str]) -> None:
        """Set operators for specific method"""
        config = self.get_config(method_name)
        config.set_operators(operators, method_name)

    def clear_method_config(self, method_name: str) -> None:
        """Clear configuration for specific method"""
        with self._lock:
            if method_name in self._method_configs:
                del self._method_configs[method_name]

    def get_all_methods(self) -> List[str]:
        """Get all configured method names"""
        with self._lock:
            return list(self._method_configs.keys())


# Global method manager instance
_method_manager = MethodOperatorManager()

def get_method_config(method_name: str) -> OperatorConfig:
    """Get operator configuration for method"""
    return _method_manager.get_config(method_name)

def set_method_operators(method_name: str, operators: List[str]) -> None:
    """Set operator configuration for method"""
    _method_manager.set_method_operators(method_name, operators)


# Convenience functions: for backward compatibility with existing code
def check_operators(operators_set: Set[str], operator: str) -> bool:
    """Check if operator set contains specified operator"""
    return operator in operators_set

def validate_operators(operators: List[str]) -> Set[str]:
    """Validate and clean operator list"""
    return set(operators) & OperatorConfig.ALL_OPERATORS


# Functions for testing
def print_all_configs() -> None:
    """Print all configuration status (for debugging)"""
    print("=== Operator Configuration Status ===")
    print(f"Global config: {_global_config}")
    print("Method configs:")
    for method in _method_manager.get_all_methods():
        config = _method_manager.get_config(method)
        print(f"  {method}: {config}")
