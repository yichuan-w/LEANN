import os
import time
import functools


class time_exec:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        timing_log = (
            f"Function '{self.func.__name__}' executed in {execution_time:.2f} seconds"
        )
        print(timing_log)
        return result, execution_time


class Logger:
    def __init__(self, args):
        self.log_file = args.log_file
        self.ds_domain = args.domain
        self.seed = args.seed
        self.datastore_size = args.sample_size
        self.stride = args.stride
        self.max_seq_length = args.max_seq_length
        self.merge = args.merge
        self.prefix = f"{self.ds_domain}\t{self.seed}\t{self.datastore_size}\t{self.stride}\t{self.max_seq_length}\t{self.merge}"

    def log_results(
        self,
        time_sample=None,
        time_chunk=None,
        time_index=None,
        time_eval=None,
        num_eval=None,
        perplexity=None,
    ):
        # Create the log entry
        log_entry = f"{self.prefix}\t{time_sample}\t{time_chunk}\t{time_index}\t{time_eval}\t{num_eval}\t{perplexity}\n"

        # Open the file in append mode. Creates the file if it doesn't exist.
        with open(self.log_file, "a") as file:
            file.write(log_entry)

    def log_string(self, log_string):
        with open(self.log_file, "a") as file:
            file.write(log_string)
