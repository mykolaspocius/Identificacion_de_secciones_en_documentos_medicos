import evaluate
import numpy as np

# Metrics to be used douring trainning

def get_seqeval_metrics(id2label:dict):
    seqeval = evaluate.load("seqeval")
    def metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (_, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recubrimiento": results["overall_recall"],
            "f1": results["overall_f1"],
            "exactitud_total": results["overall_accuracy"],
        }
    return metrics
