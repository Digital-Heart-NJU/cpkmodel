param(
    [int]$Beats = 2,
    [ValidateSet(250,500)][int]$DtUs = 500,
    [string]$Out = '',
    [switch]$Resume,
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $PSScriptRoot
$runtimeRoot = Join-Path $PSScriptRoot 'runs\runtime'
$env:TEMP = Join-Path $runtimeRoot 'tmp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path $runtimeRoot 'mpl'
$env:PYTHONPYCACHEPREFIX = Join-Path $runtimeRoot 'pycache'
$env:NUMBA_CACHE_DIR = Join-Path $runtimeRoot 'numba'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:MPLCONFIGDIR,$env:PYTHONPYCACHEPREFIX,$env:NUMBA_CACHE_DIR | Out-Null
$runArgs = @('-m','cpmodel','--beats',"$Beats",'--dt-us',"$DtUs")
if ($Out) { $runArgs += @('--out',$Out) }
if ($Resume) { $runArgs += '--resume' }
if ($Check) { $runArgs += '--check' }
& 'D:\anaconda\envs\biomodel\python.exe' @runArgs
exit $LASTEXITCODE
