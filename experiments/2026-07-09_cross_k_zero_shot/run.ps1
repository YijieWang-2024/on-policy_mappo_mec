param(
    [int]$Concurrency = 3,
    [int]$Episodes = 24
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = "C:\Users\wyj2\.conda\envs\marl\python.exe"
$OutputDir = Join-Path $Root "artifacts\evaluations\v7_random_split_hotspots\cross_k_zero_shot_20260709"
$StatusLog = Join-Path $OutputDir "status.log"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$env:PYTHONPATH = $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

$queue = [System.Collections.Generic.Queue[object]]::new()
foreach ($k in 8, 24, 32) {
    foreach ($method in "anchor_pool", "csd_pool", "mean") {
        foreach ($seed in 1, 2, 3) {
            $queue.Enqueue([pscustomobject]@{ K = $k; Method = $method; Seed = $seed })
        }
    }
}

$active = [System.Collections.Generic.List[object]]::new()

function Write-Status([string]$Message) {
    $line = "$(Get-Date -Format s) $Message"
    Add-Content -LiteralPath $StatusLog -Value $line
    Write-Output $line
}

function Start-Eval($item) {
    $run = "v7_3500k_$($item.Method)_seed$($item.Seed)"
    $model = Join-Path $Root "onpolicy\scripts\results\MEC\v7_random_split_hotspots\mappo\$run\run1\models\best"
    $stem = "$($item.Method)_seed$($item.Seed)_K$($item.K)"
    $out = Join-Path $OutputDir "$stem.json"
    $err = Join-Path $OutputDir "$stem.stderr.log"
    if ((Test-Path $out) -and (Get-Item $out).Length -gt 10) {
        Write-Status "SKIP name=$stem reason=output_exists"
        return $null
    }
    $args = @(
        "-m", "onpolicy.scripts.eval.eval_mec",
        "--model_dir", $model,
        "--mec_fleet_size", "$($item.K)",
        "--mec_eval_controller", "policy",
        "--mec_eval_seed", "100000",
        "--mec_eval_seed_stride", "13",
        "--mec_eval_episodes", "$Episodes"
    )
    $process = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root `
        -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    Write-Status "START name=$stem pid=$($process.Id)"
    return [pscustomobject]@{ Name = $stem; Process = $process }
}

Write-Status "BATCH_START jobs=$($queue.Count) concurrency=$Concurrency episodes=$Episodes"
while ($queue.Count -gt 0 -or $active.Count -gt 0) {
    while ($queue.Count -gt 0 -and $active.Count -lt $Concurrency) {
        $job = Start-Eval ($queue.Dequeue())
        if ($null -ne $job) {
            [void]$active.Add($job)
        }
    }

    if ($active.Count -eq 0) {
        continue
    }
    Start-Sleep -Seconds 10

    $remaining = [System.Collections.Generic.List[object]]::new()
    foreach ($job in $active) {
        if ($job.Process.HasExited) {
            $job.Process.WaitForExit()
            Write-Status "END name=$($job.Name) pid=$($job.Process.Id) exit=$($job.Process.ExitCode)"
        } else {
            [void]$remaining.Add($job)
        }
    }
    $active = $remaining
}
Write-Status "BATCH_DONE"
