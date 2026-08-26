<#
    Prism -- SPIR-V build step.

    Compiles shaders\matvec.comp once per GGML quant type and emits the
    resulting SPIR-V words as Delphi const arrays into
    src\Prism.Vulkan.Shaders.inc.

    The generated .inc is COMMITTED to the repository on purpose: end users
    compile Prism with nothing but Delphi, exactly as the project promises.
    Only someone editing the shaders needs the Vulkan SDK, and only then.

    Usage:
        powershell -ExecutionPolicy Bypass -File tools\BuildShaders.ps1
        ... -Debug          keep names/lines in the SPIR-V (no -O)
        ... -GlslC <path>   override compiler discovery
#>

[CmdletBinding()]
param(
    [string] $GlslC,
    [switch] $KeepDebugInfo
)

$ErrorActionPreference = 'Stop'

$root      = Split-Path -Parent $PSScriptRoot
$shaderSrc = Join-Path $root 'shaders\matvec.comp'
$outInc    = Join-Path $root 'src\Prism.Vulkan.Shaders.inc'

# ---- locate glslc -----------------------------------------------------------

function Find-GlslC {
    param([string] $Explicit)

    if ($Explicit) {
        if (Test-Path $Explicit) { return (Resolve-Path $Explicit).Path }
        throw "glslc not found at '$Explicit'"
    }

    $cmd = Get-Command glslc -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    if ($env:VULKAN_SDK) {
        $p = Join-Path $env:VULKAN_SDK 'Bin\glslc.exe'
        if (Test-Path $p) { return $p }
    }

    $newest = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue |
              Sort-Object Name -Descending | Select-Object -First 1
    if ($newest) {
        $p = Join-Path $newest.FullName 'Bin\glslc.exe'
        if (Test-Path $p) { return $p }
    }

    throw 'glslc not found. Install the Vulkan SDK or pass -GlslC <path>.'
}

$glslc = Find-GlslC -Explicit $GlslC
Write-Host "glslc:  $glslc"
Write-Host "source: $shaderSrc"

if (-not (Test-Path $shaderSrc)) { throw "missing shader source: $shaderSrc" }

# ---- the eight variants (name -> GGML type id, must match TGgmlType) --------

$variants = [ordered]@{
    'F32'  = 0
    'F16'  = 1
    'Q4_0' = 2
    'Q4_1' = 3
    'Q8_0' = 8
    'Q4_K' = 12
    'Q5_K' = 13
    'Q6_K' = 14
}

$tmp = Join-Path $env:TEMP ('prism-spv-' + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null

$sb = New-Object System.Text.StringBuilder
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$glslcVer = (& $glslc --version 2>&1 | Select-Object -First 1)

[void]$sb.AppendLine('{ AUTO-GENERATED FILE -- DO NOT EDIT.')
[void]$sb.AppendLine('')
[void]$sb.AppendLine('  Produced by tools\BuildShaders.ps1 from shaders\matvec.comp.')
[void]$sb.AppendLine('  Edit the .comp and re-run the script; never patch this file.')
[void]$sb.AppendLine('')
[void]$sb.AppendLine("  generated : $stamp")
[void]$sb.AppendLine("  compiler  : $glslcVer")
if ($KeepDebugInfo) {
    [void]$sb.AppendLine('  build     : debug (unoptimized, names kept)')
} else {
    [void]$sb.AppendLine('  build     : release (-O)')
}
[void]$sb.AppendLine('}')
[void]$sb.AppendLine('')
[void]$sb.AppendLine('const')

$total = 0

foreach ($name in $variants.Keys) {
    $qtype = $variants[$name]
    $spv   = Join-Path $tmp "matvec_$name.spv"

    # vulkan1.1 (SPIR-V 1.3): required by the subgroup reduction in main().
    $cargs = @('-fshader-stage=comp', "-DQTYPE=$qtype",
              '--target-env=vulkan1.1', '-o', $spv)
    if (-not $KeepDebugInfo) { $cargs += '-O' }
    $cargs += $shaderSrc

    & $glslc @cargs
    if ($LASTEXITCODE -ne 0) { throw "glslc failed for QTYPE=$qtype ($name)" }

    $bytes = [System.IO.File]::ReadAllBytes($spv)
    if ($bytes.Length % 4 -ne 0) { throw "$name : SPIR-V size not a multiple of 4" }

    $words = New-Object 'System.UInt32[]' ($bytes.Length / 4)
    [System.Buffer]::BlockCopy($bytes, 0, $words, 0, $bytes.Length)

    if ($words[0] -ne 0x07230203) {
        throw "$name : bad SPIR-V magic 0x$($words[0].ToString('X8'))"
    }

    $total += $bytes.Length
    Write-Host ("  {0,-5} QTYPE={1,-2}  {2,6} bytes  {3,5} words" -f `
                $name, $qtype, $bytes.Length, $words.Length)

    $ident = "SPV_MATVEC_$name"
    [void]$sb.AppendLine("  $ident : array [0 .. $($words.Length - 1)] of UInt32 = (")

    $line = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $words.Length; $i++) {
        if ($line.Length -eq 0) { [void]$line.Append('    ') }
        [void]$line.Append('$' + $words[$i].ToString('X8'))
        if ($i -lt $words.Length - 1) { [void]$line.Append(', ') }
        if ((($i + 1) % 6 -eq 0) -or ($i -eq $words.Length - 1)) {
            [void]$sb.AppendLine($line.ToString().TrimEnd())
            [void]$line.Clear()
        }
    }
    [void]$sb.AppendLine('  );')
    [void]$sb.AppendLine('')
}

[void]$sb.AppendLine("{ total SPIR-V payload: $total bytes }")

Remove-Item $tmp -Recurse -Force

[System.IO.File]::WriteAllText($outInc, $sb.ToString(), `
    (New-Object System.Text.UTF8Encoding $false))

Write-Host ''
Write-Host "wrote $outInc  ($total bytes of SPIR-V across $($variants.Count) variants)"
