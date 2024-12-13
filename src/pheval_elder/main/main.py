
import time
import os
from pheval_elder.main.constants import allfromomim619340
from pheval_elder.prepare.core.run.elder import ElderRunner
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures


class Main:
    def __init__(self):
        self.output_dir = "output"
        self.runner = None
        self.results = []

    def prepare(self):
        self.runner = ElderRunner(similarity_measure=SimilarityMeasures.COSINE)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        start_init = time.time()
        self.runner.initialize_data()
        init_time = time.time() - start_init
        print(init_time)
        start_setup = time.time()
        self.runner.setup_collections()
        setup_time = time.time() - start_setup
        print(setup_time)
        self.runner.db_manager.list_collections()

    def run(self):
        self.results = self.runner.run_analysis(allfromomim619340)

    def save_results(self):
        result_path = os.path.join(self.output_dir, "results_non_disaster_2.txt")
        with open(result_path, "w") as file:
            for result in self.results:
                file.write(str(result) + "\n")


def main():
    print("init_main")
    main_instance = Main()
    print("init_prepare")
    main_instance.prepare()
    print("init_run")
    main_instance.run()
    print("init_save")
    main_instance.save_results()
    print("results printed")


if __name__ == "__main__":
    main()