param(
    [int]$Concurrency = 3,
    [int]$Episodes = 100
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = "C:\Users\wyj2\.conda\envs\marl\python.exe"
$OutputDir = Join-Path $Root "artifacts\evaluations\v7_random_split_hotspots\fixed_k100_20260709"
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

$queue = [System.Collections.Generic.Queue[object]]::new()
foreach ($method in "flat", "mean", "hotspot_pool", "anchor_pool", "csd_pool") {
    foreach ($seed in 1, 2, 3) {
        $queue.Enqueue([pscustomobject]@{ Method = $method; Seed = $seed })
    }
}
$active = [System.Collections.Generic.List[object]]::new()

Write-Status "BATCH_START jobs=$($queue.Count) concurrency=$Concurrency episodes=$Episodes"
while ($queue.Count -gt 0 -or $active.Count -gt 0) {
    while ($queue.Count -gt 0 -and $active.Count -lt $Concurrency) {
        $item = $queue.Dequeue()
        $run = "v7_3500k_$($item.Method)_seed$($item.Seed)"
        $model = Join-Path $Root "onpolicy\scripts\results\MEC\v7_random_split_hotspots\mappo\$run\run1\models\best"
        $out = Join-Path $OutputDir "$run.json"
        $err = Join-Path $OutputDir "$run.stderr.log"
        if ((Test-Path $out) -and (Get-Item $out).Length -gt 10) {
            Write-Status "SKIP name=$run reason=output_exists"
            continue
        }
        $args = @(
            "-m", "onpolicy.scripts.eval.eval_mec",
            "--model_dir", $model,
            "--mec_eval_seed", "200000",
            "--mec_eval_seed_stride", "13",
            "--mec_eval_episodes", "$Episodes",
            "--mec_eval_include_episodes"
        )
        $process = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root `
            -RedirectStandardOutput $out -RedirectStandardError $err `
            -WindowStyle Hidden -PassThru
        [void]$active.Add([pscustomobject]@{ Name = $run; Process = $process })
        Write-Status "START name=$run pid=$($process.Id)"
    }
    if ($active.Count -eq 0) { continue }
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
