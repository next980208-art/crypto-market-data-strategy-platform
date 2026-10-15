param(
    [string]$TaskName = "CoinProject-FixedBreakoutObserver",
    [string]$DailyAt = "00:10"
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_fixed_observation_daily.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Daily runner does not exist: $runner"
}

$at = [DateTime]::ParseExact($DailyAt, "HH:mm", [Globalization.CultureInfo]::InvariantCulture)
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`""
$trigger = New-ScheduledTaskTrigger -Daily -At $at
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Daily read-only BTCUSDT fixed-rule forward observation; no orders." `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName |
    Select-Object TaskName, State, @{Name="NextRunTime";Expression={(Get-ScheduledTaskInfo -TaskName $_.TaskName).NextRunTime}}
