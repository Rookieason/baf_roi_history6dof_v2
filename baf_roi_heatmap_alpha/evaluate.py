"""Alpha method entry point into the common matched-origin evaluator."""


def evaluate(prepared, checkpoint, output, **kwargs):
    from roi_experiments.evaluation import evaluate as evaluate_shared
    return evaluate_shared(prepared, output, alpha_checkpoint=checkpoint, **kwargs)
