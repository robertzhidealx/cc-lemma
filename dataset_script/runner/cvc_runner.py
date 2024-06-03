import os
import config
import util
import time

def _get_dataset(dataset_name):
    dataset_path = os.path.join(config.dataset_root, "cvc4", dataset_name)
    return [(name, file, "") for (name, file) in util.collect_benchmarks(dataset_path, ".smt2")]

def _run_cvc(task_name, inp_file, output_path, extra_flag):
    oup_file = os.path.join(output_path, task_name + ".out")
    err_file = os.path.join(output_path, task_name + ".err")
    util.create_path(oup_file)
        
    command = [ 'ulimit -v ' + str(config.memory_limit * 1024 * 1024) + ';', "timeout " + str(config.timeout * 2), config.cvc4_bin_name, config.cvc4_args, inp_file, "--tlimit=" + str(config.timeout * 1000), ">", oup_file, "2>", err_file]
    command = " ".join(command)
    start_time = time.time()
    os.system(command)
    end_time = time.time()

    with open(oup_file, "r") as res:
        line = "\n".join(res.readlines())
    with open(err_file, "r") as err: 
        err_lines = "\n".join(err.readlines())
    line = "".join(filter(lambda char: not char.isspace(), line))

    if "Parse Error" in err_lines:
        print("CVC4 failed in parsing ", os.path.basename(task_name))
    if line == "sat" or line == "unsat":
        return True, end_time - start_time
    return False, None 

def get_runner():
    return {"dataset": _get_dataset, "runner": _run_cvc, "name": "cvc4"}
