import os
import sys
import numpy as np
import pyoperon as op
from evaluate.data_loader import split_data
from evaluate.metrics import aggregate_multi_output_metrics
from evaluate.operator_config import get_method_config
from evaluate.seed_config import get_run_seed

# Operon evolves numeric expressions, so logic values are encoded as +-1
# (False -> -1, True -> +1). Under this encoding the numeric primitives are
# exactly the logic operators: fmin(a,b) = a and b, fmax(a,b) = a or b,
# -a = not a. Combined with unit leaf coefficients and disabled linear scaling
# this keeps every evolved tree a well-formed logic expression.
TRUE_VALUE = 1.0
FALSE_VALUE = -1.0

# Search budget: the Operon (pyoperon.sklearn.SymbolicRegressor) defaults
POPULATION_SIZE = 1000
GENERATIONS = 1000
MAX_EVALUATIONS = 1000000
MAX_LENGTH = 50
MAX_DEPTH = 10
INITIALIZATION_MAX_LENGTH = 10
TOURNAMENT_SIZE = 5
CROSSOVER_PROBABILITY = 1.0
CROSSOVER_INTERNAL_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.25
LOCAL_SEARCH_PROBABILITY = 1.0
LAMARCKIAN_PROBABILITY = 1.0
OPTIMIZER_ITERATIONS = 0
EPSILON = 1e-5
IRREGULARITY_BIAS = 0.0
# pyoperon draws a random seed by default; the run seed is supplied at call
# time via get_run_seed() so --seed controls the search.
N_THREADS = int(os.environ.get('OPERON_THREADS', '1'))


def set_operators(operators):
    config = get_method_config("sr_operon")
    config.set_operators(operators, "SR Operon")


def _node_type(name):
    """Look up an Operon node type (newer pyoperon moved the math ops to BuiltinOp)"""
    builtin_ops = getattr(op, 'BuiltinOp', None)
    if builtin_ops is not None and hasattr(builtin_ops, name):
        return getattr(builtin_ops, name)
    return getattr(op.NodeType, name)


def _make_node(node_type):
    """Create an Operon node of the given type"""
    builtin_ops = getattr(op, 'BuiltinOp', None)
    if builtin_ops is not None and isinstance(node_type, builtin_ops):
        return op.Node.Function(int(node_type), 2)
    return op.Node(node_type)


def _primitive_set_config(node_types):
    """Build the primitive set configuration expected by Problem.ConfigurePrimitiveSet"""
    if hasattr(op, 'PrimitiveSetConfig'):
        return op.PrimitiveSetConfig(node_types)
    # older pyoperon expects the node types OR-ed into a bit mask
    config = 0
    for node_type in node_types:
        config |= int(node_type)
    return config


def _calibrate_logic_node_types():
    """Find the node types computing min (and) and max (or)

    The Fmin/Fmax enum labels are swapped in pyoperon 0.6.1 (NodeType.Fmin
    builds the node that evaluates to max) and Node.Name is not readable from
    Python in every build, so both candidates are evaluated on the four input
    combinations and identified by what they actually compute.
    """
    grid = np.asfortranarray(np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]))
    ds = op.Dataset(grid)
    hashes = [v.Hash for v in sorted(ds.Variables, key=lambda v: v.Index)]

    node_types = {}
    for label in ('Fmin', 'Fmax'):
        node_type = _node_type(label)
        leaves = []
        for variable_hash in hashes:
            leaf = op.Node.Variable(1.0)
            leaf.HashValue = variable_hash
            leaves.append(leaf)
        root = _make_node(node_type)
        root.Arity = 2
        tree = op.Tree(leaves + [root]).UpdateNodes()
        values = np.asarray(
            op.Evaluate(op.DispatchTable(), tree, ds, op.Range(0, ds.Rows))).flatten()
        if np.array_equal(values, grid.min(axis=1)):
            node_types['and'] = node_type
        elif np.array_equal(values, grid.max(axis=1)):
            node_types['or'] = node_type

    if set(node_types) != {'and', 'or'}:
        raise RuntimeError("Could not identify the Operon min/max node types")
    return node_types


# Node types are identified once, by what they compute
LOGIC_NODE_TYPES = _calibrate_logic_node_types()
NOT_NODE_TYPE = _node_type('Sub')


def _is_node_type(node, node_type):
    """Whether a tree node is a function node of the given type

    Function nodes carry the node type in their hash, both in pyoperon 0.6.1
    (where it is the NodeType value) and in newer versions (where it is the
    BuiltinOp value), so this works without reading the node name.
    """
    return not node.IsVariable and int(node.HashValue) == int(node_type)


def setup_operon_primitives(problem):
    """Enable the Operon primitives matching the allowed logic operators"""
    config = get_method_config("sr_operon")
    operators = config.get_operators()
    if not operators:
        operators = {'and', 'or', 'not'}
        print("  No operator restriction given, using and/or/not")

    node_types = [_node_type('Variable')]
    if 'and' in operators:
        node_types.append(LOGIC_NODE_TYPES['and'])
    if 'or' in operators:
        node_types.append(LOGIC_NODE_TYPES['or'])
    if 'not' in operators:
        node_types.append(NOT_NODE_TYPE)

    problem.ConfigurePrimitiveSet(_primitive_set_config(node_types))
    pset = problem.PrimitiveSet

    if 'not' in operators:
        # Subtraction with a single argument is a negation, so restricting it to
        # that arity turns it into a pure NOT gate
        pset.SetMinMaxArity(_make_node(NOT_NODE_TYPE), 1, 1)

    return pset


def encode_binary(X):
    """Map binary data to the +-1 encoding used by the search"""
    return np.asfortranarray(
        np.where(np.asarray(X) > 0.5, TRUE_VALUE, FALSE_VALUE).astype(float))


def evolve_operon_tree(X_train, y_train):
    """Evolve a logic tree for a single output with Operon

    Returns:
        (tree, variable_names) tuple, where variable_names maps Operon variable
        hashes to the LogicSR standard variable names (x1, x2, ...)
    """
    D = np.column_stack((encode_binary(X_train), encode_binary(y_train.flatten())))
    ds = op.Dataset(np.asfortranarray(D))

    target = max(ds.Variables, key=lambda v: v.Index)
    variable_names = {
        v.Hash: f'x{v.Index + 1}'
        for v in sorted(ds.Variables, key=lambda v: v.Index) if v.Hash != target.Hash
    }
    inputs = list(variable_names)

    problem = op.Problem(ds)
    problem.TrainingRange = op.Range(0, ds.Rows)
    problem.TestRange = op.Range(ds.Rows - 1, ds.Rows)
    problem.Target = target
    problem.InputHashes = inputs
    pset = setup_operon_primitives(problem)

    creator = op.BalancedTreeCreator(pset, inputs, IRREGULARITY_BIAS)
    tree_initializer = op.UniformLengthTreeInitializer(creator)
    min_arity, _ = pset.FunctionArityLimits()
    tree_initializer.ParameterizeDistribution(
        min_arity + 1, min(INITIALIZATION_MAX_LENGTH, MAX_LENGTH))
    tree_initializer.MinDepth = 1
    tree_initializer.MaxDepth = 1000

    # All leaf coefficients are fixed to 1 so that a variable node is the literal
    # itself rather than a weighted literal
    coefficient_initializer = op.UniformIntCoefficientAnalyzer()
    coefficient_initializer.ParameterizeDistribution(1, 1)

    # Only structural mutations: the point mutations would perturb coefficients
    mutation = op.MultiMutation()
    mutations = [
        op.ChangeVariableMutation(inputs),
        op.ChangeFunctionMutation(pset),
        op.ReplaceSubtreeMutation(creator, coefficient_initializer, MAX_DEPTH, MAX_LENGTH),
        op.InsertSubtreeMutation(creator, coefficient_initializer, MAX_DEPTH, MAX_LENGTH),
        op.RemoveSubtreeMutation(pset),
    ]
    for mutator in mutations:
        mutation.Add(mutator, 1.0)
    crossover = op.SubtreeCrossover(CROSSOVER_INTERNAL_PROBABILITY, MAX_DEPTH, MAX_LENGTH)

    # Mean squared error without linear scaling: a scaled model would no longer
    # be a logic expression
    dtable = op.DispatchTable()
    evaluator = op.Evaluator(problem, dtable, op.MSE(), False)
    evaluator.Budget = MAX_EVALUATIONS

    comparison = op.SingleObjectiveComparison(0)
    selector = op.TournamentSelector(comparison)
    selector.TournamentSize = TOURNAMENT_SIZE
    reinserter = op.KeepBestReinserter(comparison)

    # Local search runs zero iterations (the Operon default), so it leaves the
    # unit coefficients untouched. The optimizer must stay in a named variable:
    # CoefficientOptimizer only borrows it, so a temporary would be collected
    # and the next run would read freed memory.
    optimizer = op.LMOptimizer(dtable, problem, OPTIMIZER_ITERATIONS, 0)
    local_search = op.CoefficientOptimizer(optimizer)
    generator = op.BasicOffspringGenerator(evaluator, crossover, mutation,
                                           selector, selector, local_search)

    config = op.GeneticAlgorithmConfig(
        generations=GENERATIONS,
        max_evaluations=MAX_EVALUATIONS,
        local_iterations=OPTIMIZER_ITERATIONS,
        population_size=POPULATION_SIZE,
        pool_size=POPULATION_SIZE,
        p_crossover=CROSSOVER_PROBABILITY,
        p_mutation=MUTATION_PROBABILITY,
        p_local=LOCAL_SEARCH_PROBABILITY,
        p_lamarck=LAMARCKIAN_PROBABILITY,
        epsilon=EPSILON,
        seed=get_run_seed(),
        max_time=sys.maxsize)

    gp = op.GeneticProgrammingAlgorithm(config, problem, tree_initializer,
                                        coefficient_initializer, generator, reinserter)
    rng = op.RandomGenerator(np.uint64(config.Seed))
    gp.Run(rng, None, N_THREADS)

    return gp.BestModel.Genotype, variable_names


def _is_not(node):
    """Whether the node is the unary minus that encodes a NOT gate"""
    return _is_node_type(node, NOT_NODE_TYPE) and node.Arity == 1


def convert_operon_tree(tree, variable_names):
    """Convert an Operon tree to a parenthesized logic expression"""
    nodes = list(tree.Nodes)
    if not nodes:
        return "False"

    def build(index):
        """Build the expression of the subtree ending at index (nodes are stored
        in postfix order), also returning the index where the next sibling ends"""
        node = nodes[index]
        if node.IsVariable:
            return variable_names[node.HashValue], index - 1
        if _is_not(node):
            operand, next_index = build(index - 1)
            return f"(not {operand})", next_index
        for operator in ('and', 'or'):
            if _is_node_type(node, LOGIC_NODE_TYPES[operator]) and node.Arity == 2:
                left, next_index = build(index - 1)
                right, next_index = build(next_index)
                return f"({left} {operator} {right})", next_index
        raise ValueError(
            f"Unexpected Operon node (hash {node.HashValue}, arity {node.Arity})")

    expression, _ = build(len(nodes) - 1)
    return expression


def predict_operon_tree(tree, X):
    """Evaluate an Operon tree on binary data and threshold back to 0/1"""
    ds = op.Dataset(encode_binary(X))
    values = op.Evaluate(op.DispatchTable(), tree, ds, op.Range(0, ds.Rows))
    return (np.asarray(values).flatten() > 0).astype(int)


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Find logic expressions using Operon

    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided,
                       metrics will be calculated using model predictions vs ground truth.
    """
    print("=" * 60)
    print("Operon (Genetic Programming)")
    print("=" * 60)

    expressions = []
    metrics_list = []
    train_pred_columns = []
    test_pred_columns = []
    used_vars = set()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)

    for output_idx in range(Y_train.shape[1]):
        y_train = Y_train[:, output_idx]

        print(f" Processing output {output_idx+1}...")

        tree, variable_names = evolve_operon_tree(X_train, y_train)
        expr = convert_operon_tree(tree, variable_names)

        # Track which input variables are used in the expression
        for v in range(1, X.shape[1] + 1):
            if f"x{v}" in expr:
                used_vars.add(f"x{v}")

        y_train_pred = predict_operon_tree(tree, X_train)
        y_test_pred = predict_operon_tree(tree, X_test)

        expressions.append(expr)
        train_pred_columns.append(y_train_pred)
        test_pred_columns.append(y_test_pred)

    # If Y_ground_truth is provided, use it for evaluation instead of noisy Y
    if Y_ground_truth is not None:
        _, _, Y_train_gt, Y_test_gt = split_data(X, Y_ground_truth, test_size=1-split, random_state=42)
        aggregated_metrics = aggregate_multi_output_metrics(Y_train_gt, Y_test_gt,
                                                            train_pred_columns,
                                                            test_pred_columns)
        print(f"  Using ground truth for evaluation")
    else:
        aggregated_metrics = aggregate_multi_output_metrics(Y_train, Y_test,
                                                            train_pred_columns,
                                                            test_pred_columns)
    accuracy_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if aggregated_metrics:
        accuracy_tuple = (
            aggregated_metrics['train_bit_acc'],
            aggregated_metrics['test_bit_acc'],
            aggregated_metrics['train_sample_acc'],
            aggregated_metrics['test_sample_acc'],
            aggregated_metrics['train_output_acc'],
            aggregated_metrics['test_output_acc'])
    metrics_list = [accuracy_tuple]
    all_vars_used = all(f'x{i}' in used_vars for i in range(1, X.shape[1] + 1))
    extra_info = {
        'all_vars_used': all_vars_used,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, metrics_list, extra_info
