$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = "C:\Users\wyj2\.conda\envs\marl\python.exe"
$OutputDir = Join-Path $Root "artifacts\runs\v7_random_split_hotspots\v7_grid_pool_3500k_20260709"
$StatusLog = Join-Path $OutputDir "status.log"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$env:PYTHONPATH = $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

function Write-Status([string]$Message) {
    $line = "$(Get-Date -Format s) $Message"
    Add-Content -LiteralPath $StatusLog -Value $line
    Write-Output $line
}

$training = @()
foreach ($seed in 1, 2, 3) {
    $name = "v7_3500k_grid_pool_seed$seed"
    $args = @(
        "-m", "onpolicy.scripts.train.train_mec",
        "--env_name", "MEC",
        "--algorithm_name", "mappo",
        "--mec_scenario", "v7_random_split_hotspots",
        "--mec_policy_arch", "grid_pool",
        "--mec_critic_arch", "grid_pool",
        "--seed", "$seed",
        "--n_training_threads", "1",
        "--n_rollout_threads", "8",
        "--n_eval_rollout_threads", "1",
        "--episode_length", "200",
        "--num_env_steps", "3500000",
        "--ppo_epoch", "15",
        "--num_mini_batch", "1",
        "--hidden_size", "144",
        "--layer_N", "1",
        "--entropy_coef", "0.01",
        "--mec_logstd_init", "-1.9",
        "--use_eval",
        "--eval_interval", "50",
        "--eval_episodes", "8",
        "--eval_seed", "1000",
        "--test_episodes", "100",
        "--test_seed", "200000",
        "--test_seed_stride", "13",
        "--user_name", "wyj2",
        "--use_wandb",
        "--log_interval", "10",
        "--save_interval", "25",
        "--experiment_name", $name
    )
    $process = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $OutputDir "$name.out.log") `
        -RedirectStandardError (Join-Path $OutputDir "$name.err.log") `
        -WindowStyle Hidden -PassThru
    $training += [pscustomobject]@{ Name = $name; Process = $process }
    Write-Status "START_TRAIN name=$name pid=$($process.Id)"
}

foreach ($job in $training) {
    $job.Process.WaitForExit()
    Write-Status "END_TRAIN name=$($job.Name) pid=$($job.Process.Id) exit=$($job.Process.ExitCode)"
}

$evaluation = @()
foreach ($job in $training) {
    $model = Join-Path $Root "onpolicy\scripts\results\MEC\v7_random_split_hotspots\mappo\$($job.Name)\run1\models\best"
    if (-not (Test-Path (Join-Path $model "actor.pt"))) {
        Write-Status "SKIP_EVAL name=$($job.Name) reason=missing_best_actor"
        continue
    }
    $args = @(
        "-m", "onpolicy.scripts.eval.eval_mec",
        "--model_dir", $model,
        "--mec_eval_seed", "200000",
        "--mec_eval_seed_stride", "13",
        "--mec_eval_episodes", "100",
        "--mec_eval_include_episodes"
    )
    $process = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $OutputDir "$($job.Name).json") `
        -RedirectStandardError (Join-Path $OutputDir "$($job.Name).stderr.log") `
        -WindowStyle Hidden -PassThru
    $evaluation += [pscustomobject]@{ Name = $job.Name; Process = $process }
    Write-Status "START_EVAL name=$($job.Name) pid=$($process.Id)"
}
foreach ($job in $evaluation) {
    $job.Process.WaitForExit()
    Write-Status "END_EVAL name=$($job.Name) pid=$($job.Process.Id) exit=$($job.Process.ExitCode)"
}
Write-Status "BATCH_DONE"
