param(
  [int]$Port = 8151,
  [string]$BindHost = "0.0.0.0"
)
# Keep the flow viewer alive: if the server exits (crash, GPU hiccup, Ctrl+C
# in another console), restart it in place. Run detached:
#   cmd /c start "" /B powershell -NoProfile -ExecutionPolicy Bypass -File run_flow_server.ps1
$root = "A:\dev\GNM-V\pivot\chat-wrapper"
$py = "A:\dev\GNM-V\.venv\Scripts\python.exe"
Set-Location $root
while ($true) {
  & $py chat_server.py --port $Port --host $BindHost 2>&1 |
    Tee-Object -FilePath "runs\server_$Port.log" -Append
  Write-Host "[watchdog] server exited with $LASTEXITCODE at $(Get-Date -Format HH:mm:ss) - restarting in 3s"
  Start-Sleep -Seconds 3
}
