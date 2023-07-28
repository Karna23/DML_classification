from .exact_solution import ExactSolution
from .nearest_neighbours import KNN


def load(cfg, checkpoint):
    if cfg.classifier == "exact_solution":
        classifier = ExactSolution(cfg, embedding_collection=checkpoint["embeddings"],
                                   labels_str=checkpoint["labels_str"],
                                   labels_int=checkpoint["labels_int"],
                                   label_map=checkpoint["label_map"])
        classifier.solve_exact()
        classifier.raw_collection = None

    elif cfg.classifier == "knn":
        classifier = KNN(embedding_collection=checkpoint["embeddings"],
                         labels_str=checkpoint["labels_str"],
                         labels_int=checkpoint["labels_int"],
                         label_map=checkpoint["label_map"],
                         number_of_neighbours=cfg.number_of_neighbours)
    else:
        raise ValueError('Unsupported classifier type,\
                         must be one of exact_solution, knn')

    return classifier