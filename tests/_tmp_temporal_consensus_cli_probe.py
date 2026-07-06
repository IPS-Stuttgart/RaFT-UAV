from raft_uav.mmuad.evaluator import load_evaluation_truth_file


def _probe(path):
    return load_evaluation_truth_file(path).rows
