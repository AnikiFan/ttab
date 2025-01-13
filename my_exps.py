import subprocess
if __name__ == '__main__':
    experiments = [
        # tent online
        # "python run_exps.py --script_path exps/tent/online/cifar10c.py --num_jobs_per_node 40 --num_jobs_per_script 1 --wait_in_seconds_per_job 0.1",
        # "python run_exps.py --script_path exps/tent/online/cifar100c.py --num_jobs_per_node 40 --num_jobs_per_script 1 --wait_in_seconds_per_job 0.1",
        # "python run_exps.py --script_path exps/tent/online/cifar10_1.py --num_jobs_per_node 27 --num_jobs_per_script 1 --wait_in_seconds_per_job 0.1",
        # "python run_exps.py --script_path exps/tent/online/officehome.py --num_jobs_per_node 3 --num_jobs_per_script 1 --wait_in_seconds_per_job 1",
        # "python run_exps.py --script_path exps/tent/online/pacs.py --num_jobs_per_node 3 --num_jobs_per_script 1 --wait_in_seconds_per_job 1",
        # tent episodic
        "python run_exps.py --script_path exps/tent/episodic/cifar10c.py --num_jobs_per_node 40 --num_jobs_per_script 1 --wait_in_seconds_per_job 0.1",
        "python run_exps.py --script_path exps/tent/episodic/cifar100c.py --num_jobs_per_node 40 --num_jobs_per_script 1 --wait_in_seconds_per_job 0.1",
        "python run_exps.py --script_path exps/tent/episodic/cifar10_1.py --num_jobs_per_node 27 --num_jobs_per_script 1 --wait_in_seconds_per_job 0.1",
        "python run_exps.py --script_path exps/tent/episodic/officehome.py --num_jobs_per_node 3 --num_jobs_per_script 1 --wait_in_seconds_per_job 1",
        "python run_exps.py --script_path exps/tent/episodic/pacs.py --num_jobs_per_node 3 --num_jobs_per_script 1 --wait_in_seconds_per_job 1",
    ]
    for experiment in experiments:
        print(f"Running experiment {experiment}")
        result = subprocess.run(experiment)
        if result.returncode != 0:
            print(f"{experiment} failed")
        else :
            print(f"{experiment} succeeded")
    print("All experiments completed")

