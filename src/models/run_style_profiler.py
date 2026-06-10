"""Phase 3 orchestration — run all StyleProfiler steps in order."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.style_profiler import StyleProfiler


def main():
    profiler = StyleProfiler()

    profiler.load_and_preprocess()
    profiler.run_pca()
    profiler.run_umap()
    profiler.find_optimal_k()
    profiler.fit_kmeans_and_assign_archetypes()
    profiler.compute_similarity_matrix()
    profiler.save_to_db()
    profiler.generate_figures()
    profiler.print_summary()


if __name__ == '__main__':
    main()
