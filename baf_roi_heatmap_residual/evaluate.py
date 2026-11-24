"""Internal residual method entry point into the common matched evaluator."""


def evaluate(prepared, checkpoint, output, **kwargs):
    from roi_experiments.evaluation import evaluate as evaluate_shared
    return evaluate_shared(prepared, output, residual_checkpoint=checkpoint, **kwargs)
