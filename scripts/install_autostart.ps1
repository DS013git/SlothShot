<#
.SYNOPSIS
    Start "slothshot watch" quietly at Windows login ("fully lazy" mode).

.DESCRIPTION
    Registers a per-user Task Scheduler task that runs at login. It launches
    pythonw.exe (the Python launcher WITHOUT a console window), so nothing
    flashes on screen. Logs go to a rotating file instead of a console.

    No administrator rights are needed: the task runs as you, only while you
    are logged in, and only touches your own files.

    If Task Scheduler is not available to you (some locked-down machines),
    use -Method Startup to put a shortcut in your Startup folder instead.

.PARAMETER Uninstall
    Remove the task and the Startup shortcut, and stop a running watcher.

.PARAMETER Status
    Show whether autostart is installed, whether the watcher is running,
    and the last lines of its log.

.PARAMETER Method
    Task (default): Task Scheduler. Startup: shortcut in the Startup folder.

.PARAMETER ConfigFile
    Optional config.toml to pass to the watcher (default: the user config).

.PARAMETER PythonW
    Path to pythonw.exe. Default: .venv\Scripts\pythonw.exe in this repo,
    otherwise the first pythonw.exe on PATH.

.PARAMETER LogFile
    Where the watcher writes its log.
    Default: %LOCALAPPDATA%\SlothShot\logs\watch.log

.PARAMETER TaskName
    Name of the scheduled task. Default: SlothShotWatch.

.PARAMETER StartNow
    Also start the watcher right away instead of waiting for the next login.

.EXAMPLE
    .\scripts\install_autostart.ps1 -StartNow

.EXAMPLE
    .\scripts\install_autostart.ps1 -Status

.EXAMPLE
    .\scripts\install_autostart.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$Status,
    [ValidateSet("Task", "Startup")]
    [string]$Method = "Task",
    [string]$ConfigFile,
    [string]$PythonW,
    [string]$LogFile = (Join-Path $env:LOCALAPPDATA "SlothShot\logs\watch.log"),
    [string]$TaskName = "SlothShotWatch",
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
# The task starts in a different working directory, so the log path must be absolute.
$LogFile = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($LogFile)
$ShortcutPath = Join-Path ([Environment]::GetFolderPath("Startup")) "$TaskName.lnk"
# Used to recognise our own process when stopping or reporting status.
$ProcessMarker = "slothshot.cli watch"


function Get-WatcherProcess {
    # Win32_Process exposes the command line; Get-Process does not.
    Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe' OR Name = 'python.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($ProcessMarker) }
}


function Resolve-PythonW {
    if ($PythonW) {
        if (-not (Test-Path -LiteralPath $PythonW)) { throw "pythonw.exe not found at: $PythonW" }
        return (Resolve-Path -LiteralPath $PythonW).Path
    }
    $venv = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
    if (Test-Path -LiteralPath $venv) { return $venv }
    $found = Get-Command pythonw.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.Source }
    throw "Could not find pythonw.exe. Create the virtual environment first (see README), or pass -PythonW."
}


function Assert-SlothShotInstalled([string]$PythonWPath) {
    # Check with the console python.exe next to pythonw.exe, so we can see errors.
    $python = Join-Path (Split-Path -Parent $PythonWPath) "python.exe"
    if (-not (Test-Path -LiteralPath $python)) { return }
    & $python -c "import slothshot" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "SlothShot is not installed for $python. Run: python -m pip install -e ."
    }
}


function Get-WatcherArguments {
    $parts = @("-m", "slothshot.cli", "watch", "--log-file", "`"$LogFile`"")
    if ($ConfigFile) {
        $resolved = (Resolve-Path -LiteralPath $ConfigFile).Path
        $parts += @("--config", "`"$resolved`"")
    }
    return ($parts -join " ")
}


function Remove-Autostart([switch]$Quiet) {
    $removed = $false
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        if (-not $Quiet) { Write-Host "Removed scheduled task '$TaskName'." }
        $removed = $true
    }
    if (Test-Path -LiteralPath $ShortcutPath) {
        Remove-Item -LiteralPath $ShortcutPath -Force
        if (-not $Quiet) { Write-Host "Removed Startup shortcut $ShortcutPath" }
        $removed = $true
    }
    foreach ($process in @(Get-WatcherProcess)) {
        Stop-Process -Id $process.ProcessId -Force
        if (-not $Quiet) { Write-Host "Stopped running watcher (PID $($process.ProcessId))." }
        $removed = $true
    }
    if (-not $removed -and -not $Quiet) {
        Write-Host "Nothing to remove: autostart is not installed."
    }
}


function Install-TaskSchedulerTask([string]$Exe, [string]$Arguments) {
    $action = New-ScheduledTaskAction -Execute $Exe -Argument $Arguments -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    # Battery-friendly defaults for a laptop: keep running on battery, never time out,
    # retry (the screenshots folder may not exist yet), and run only one copy.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -Hidden `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -MultipleInstances IgnoreNew
    # Interactive + Limited = runs as you, in your session, without admin rights.
    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Force `
        -Description "Starts SlothShot's screenshot watcher quietly at login." | Out-Null
    Write-Host "Installed scheduled task '$TaskName' (runs at login, no console window)."
}


function Install-StartupShortcut([string]$Exe, [string]$Arguments) {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $Exe
    $shortcut.Arguments = $Arguments
    $shortcut.WorkingDirectory = $RepoRoot
    $shortcut.WindowStyle = 7   # minimized; pythonw has no window anyway
    $shortcut.Description = "SlothShot screenshot watcher"
    $shortcut.Save()
    Write-Host "Installed Startup shortcut $ShortcutPath"
}


function Show-Status {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) { Write-Host "Scheduled task '$TaskName': installed (state: $($task.State))" }
    else { Write-Host "Scheduled task '$TaskName': not installed" }
    if (Test-Path -LiteralPath $ShortcutPath) { Write-Host "Startup shortcut: installed ($ShortcutPath)" }
    else { Write-Host "Startup shortcut: not installed" }
    $running = @(Get-WatcherProcess)
    if ($running.Count -gt 0) { Write-Host "Watcher: running (PID $($running.ProcessId -join ', '))" }
    else { Write-Host "Watcher: not running" }
    if (Test-Path -LiteralPath $LogFile) {
        Write-Host "Last log lines from ${LogFile}:"
        Get-Content -LiteralPath $LogFile -Tail 8 | ForEach-Object { Write-Host "  $_" }
    }
    else { Write-Host "Log file not created yet: $LogFile" }
}


if ($Status) { Show-Status; return }
if ($Uninstall) { Remove-Autostart; return }

$exe = Resolve-PythonW
Assert-SlothShotInstalled $exe
$arguments = Get-WatcherArguments
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null

# Re-running the installer must be safe: clear the old registration first.
Remove-Autostart -Quiet

if ($Method -eq "Task") {
    try { Install-TaskSchedulerTask $exe $arguments }
    catch {
        Write-Warning "Task Scheduler registration failed: $($_.Exception.Message)"
        Write-Warning "Falling back to a Startup-folder shortcut."
        Install-StartupShortcut $exe $arguments
    }
}
else {
    Install-StartupShortcut $exe $arguments
}

Write-Host "Command: `"$exe`" $arguments"
Write-Host "Log file: $LogFile"

if ($StartNow) {
    if ((Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
        Start-ScheduledTask -TaskName $TaskName
    }
    else {
        Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden
    }
    Write-Host "Watcher started. Check it with: .\scripts\install_autostart.ps1 -Status"
}
else {
    Write-Host "It will start at your next login. Use -StartNow to start it immediately."
}
