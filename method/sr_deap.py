import numpy as np
import re
from deap import base, creator, tools, gp, algorithms
from evaluate.data_loader import split_data  
from evaluate.metrics import calculate_metrics, aggregate_multi_output_metrics 
from evaluate.operator_config import get_method_config  


def set_operators(operators):
    config = get_method_config("sr_deap")
    config.set_operators(operators, "SR DEAP")


def setup_gp_primitives(input_size):
    """Setup genetic programming primitives"""
    config = get_method_config("sr_deap")
    pset = gp.PrimitiveSet("MAIN", input_size)

    # Add operators based on configuration
    # Note: Using uppercase names (AND, OR, NOT) to avoid Python keyword conflicts
    # DEAP cannot directly use lowercase 'and', 'or', 'not' as they conflict with Python syntax
    if config.has_operator('and'):
        pset.addPrimitive(lambda x, y: float(x > 0.5 and y > 0.5), 2, name="AND")
    if config.has_operator('or'):
        pset.addPrimitive(lambda x, y: float(x > 0.5 or y > 0.5), 2, name="OR")
    if config.has_operator('not'):
        pset.addPrimitive(lambda x: float(not (x > 0.5)), 1, name="NOT")

    # Rename arguments
    for i in range(input_size):
        pset.renameArguments(**{f'ARG{i}': f'x{i+1}'})
    return pset


def convert_expression(expr_str):
    """Convert DEAP expression to standard format"""
    import re
    result = str(expr_str)
    
    # Convert DEAP operator names to standard format
    result = re.sub(r'AND\(', 'and(', result)
    result = re.sub(r'OR\(', 'or(', result)
    result = re.sub(r'NOT\(', 'not(', result)
    
    return result


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Use DEAP genetic programming to find logical expressions
    
    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided, 
                       metrics will be calculated using model predictions vs ground truth.
    """
    print("=" * 60)
    print("DEAP (Genetic Programming)")
    print("=" * 60)

    expressions = []
    metrics_list = []
    train_pred_columns = []
    test_pred_columns = []
    used_vars = set()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)

    for output_idx in range(Y_train.shape[1]):
        y_train = Y_train[:, output_idx]
        y_test = Y_test[:, output_idx]

        print(f" Processing output {output_idx+1}...")

        pset = setup_gp_primitives(X_train.shape[1])

        # Clean up previous creator definitions
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Fitness function
        def eval_individual(individual):
            func = gp.compile(individual, pset)
            predictions = []
            for row in X_train:
                pred = func(*row)
                predictions.append(1 if pred > 0.5 else 0)
            accuracy = np.mean(np.array(predictions) == y_train)
            return (accuracy, )

        toolbox.register("evaluate", eval_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

        # Run evolution
        pop = toolbox.population(n=50) 
        hof = tools.HallOfFame(1)

        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=20,
            halloffame=hof, verbose=False)

        # Get best individual and predict
        if hof and len(hof) > 0:
            best_individual = hof[0]
            expr = convert_expression(str(best_individual))
            
            # Use the same compiled function for both train and test prediction
            best_func = gp.compile(best_individual, pset)
            y_train_pred = np.array([1 if best_func(*row) > 0.5 else 0 for row in X_train])
            y_test_pred = np.array([1 if best_func(*row) > 0.5 else 0 for row in X_test])
        else:
            expr = "False"
            y_train_pred = np.zeros(len(X_train), dtype=int)
            y_test_pred = np.zeros(len(X_test), dtype=int)

        train_pred_columns.append(y_train_pred)
        test_pred_columns.append(y_test_pred)

        expressions.append(expr)

        vars_in_expr = re.findall(r'x\d+', expr)
        used_vars.update(vars_in_expr)

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
    extra_info = {
        'all_vars_used': True,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, metrics_list, extra_info
