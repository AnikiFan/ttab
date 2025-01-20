import subprocess
if __name__ == '__main__':
    experiments = [
        cmd.split() for cmd in
        [
        # tent online
        "python run_exps.py --script_path exps/tent/online/cifar10c.py    --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/tent/online/cifar100c.py   --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/tent/online/cifar10_1.py   --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/tent/online/officehome.py  --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/tent/online/pacs.py        --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        # tent episodic
        "python run_exps.py --script_path exps/tent/episodic/cifar10c.py    --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/tent/episodic/cifar100c.py   --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/tent/episodic/cifar10_1.py   --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/tent/episodic/officehome.py  --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/tent/episodic/pacs.py        --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        # sar online
        "python run_exps.py --script_path exps/sar/online/cifar10c.py    --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/sar/online/cifar100c.py   --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/sar/online/cifar10_1.py   --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/sar/online/officehome.py  --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/sar/online/pacs.py        --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        # sar episodic
        "python run_exps.py --script_path exps/sar/episodic/cifar10c.py    --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/sar/episodic/cifar100c.py   --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/sar/episodic/cifar10_1.py   --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/sar/episodic/officehome.py  --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        "python run_exps.py --script_path exps/sar/episodic/pacs.py        --num_jobs_per_node 2 --num_jobs_per_script 1 --wait_in_seconds_per_job 3",
        ]
    ]
    failed = []
    for experiment in experiments:
        print(f"Running experiment {experiment}")
        result = subprocess.run(experiment)
        if result.returncode != 0:
            print(f"{experiment} failed")
            failed.append(experiment)
        else :
            print(f"{experiment} succeeded")
    print("All experiments completed")
    if failed:
        for exp in failed:
            print(f"f{exp} failed!")

