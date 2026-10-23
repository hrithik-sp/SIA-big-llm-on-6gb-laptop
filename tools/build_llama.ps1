<#
.SYNOPSIS
  Build llama.cpp (or a fork such as ik_llama.cpp) with CUDA, tuned for this laptop.

.DESCRIPTION
  - CUDA for sm_86 only (RTX 3050 Laptop), no wasted fatbin targets.
  - MSVC's native detection (FindSIMD.cmake) enables base AVX-512 but not its
    extensions; Zen 4 supports VNNI, VBMI and BF16 (verified with CPUID), so VNNI
    and VBMI are enabled explicitly. BF16 stays off: MSVC 19.29 (VS 2019) lacks
    the AVX512-BF16 intrinsics (_mm512_cvtne2ps_pbh -> C3861 in sgemm.cpp).
  - Parallel jobs are derived from currently available RAM: nvcc jobs can take
    1-2 GB each, and running out of RAM would thrash the pagefile.
#>
param(
    [string]$Src = $(if ($env:LLM_HOME) { "$env:LLM_HOME\llama.cpp" } else { "$env:USERPROFILE\llm\llama.cpp" }),
    [string]$Build = "",
    [int]$Jobs = 0,
    [string[]]$Extra = @(),
    # cuda     : monolithic MSVC build (CPU + CUDA), the original configuration
    # msvc-dl  : MSVC, loadable backends (GGML_BACKEND_DL): programs, ggml-base, llama,
    #            ggml-cuda.dll and an MSVC ggml-cpu.dll (AVX512 + VNNI + VBMI)
    # llvm-cpu : clang ggml-cpu-zen4.dll only (+ LLVM libomp). Swapped into the msvc-dl
    #            bin dir, it is an A/B where only the CPU backend's compiler differs.
    #            (Programs can't be built with clang 23 against the VS 2019 STL:
    #            <functional> "reinterpret_cast ... casts away qualifiers".)
    [ValidateSet("cuda", "msvc-dl", "llvm-cpu")][string]$Mode = "cuda"
)
$ErrorActionPreference = "Stop"
if (-not $Build) { $Build = Join-Path $Src @{ "cuda" = "build-cuda"; "msvc-dl" = "build-msvc-dl"; "llvm-cpu" = "build-llvm" }[$Mode] }

# Tools installed after this shell started are only on the persisted PATH.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
            "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
$cuda = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
if (-not $cuda -or -not (Test-Path "$cuda\bin\nvcc.exe")) { throw "CUDA Toolkit not found (CUDA_PATH='$cuda')" }
foreach ($t in "cmake", "ninja") {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { throw "$t not found on PATH" }
}
$vcvars = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found" }

$availMB = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
if ($Jobs -le 0) { $Jobs = [math]::Max(1, [math]::Min(16, [math]::Floor(($availMB - 1500) / 1500))) }
Write-Host "available RAM: $availMB MB -> $Jobs parallel jobs"
Write-Host "CUDA: $cuda"

$cudaFlags = @("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=86", "`"-DCMAKE_CUDA_COMPILER=$cuda\bin\nvcc.exe`"")
$target = ""
switch ($Mode) {
    "cuda" {
        $cfg = @("-G Ninja", "-DCMAKE_BUILD_TYPE=Release") + $cudaFlags + @(
            "-DGGML_NATIVE=ON", "-DGGML_AVX512_VNNI=ON", "-DGGML_AVX512_VBMI=ON", "-DGGML_AVX512_BF16=OFF",
            "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=ON")
    }
    "msvc-dl" {
        # GGML_NATIVE is rejected with GGML_BACKEND_DL, so the CPU features are explicit
        # (the same set native detection + our flags gave the monolithic build)
        $cfg = @("-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_BACKEND_DL=ON", "-DGGML_NATIVE=OFF",
                 "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_AVX512=ON",
                 "-DGGML_AVX512_VNNI=ON", "-DGGML_AVX512_VBMI=ON", "-DGGML_AVX512_BF16=OFF",
                 "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=ON") + $cudaFlags
    }
    "llvm-cpu" {
        $llvm = "C:\Program Files\LLVM\bin"
        if (-not (Test-Path "$llvm\clang.exe")) { throw "LLVM not found at $llvm" }
        $env:Path = "$llvm;$env:Path"
        # GGML_NATIVE is rejected with GGML_BACKEND_DL; like the release, build all CPU
        # variants. The loader picks the best match at runtime; this CPU gets the
        # "zen4" variant (AVX512 + VBMI + VNNI + BF16, ggml/src/CMakeLists.txt).
        # OpenMP: the release uses GGML_OPENMP_FETCH, which pins LLVM OpenMP 20.1.8 and
        # requires Clang 20. The installed LLVM ships libomp matching its own clang,
        # so that runtime is used and copied next to the binaries after the build.
        $cfg = @("-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "`"-DCMAKE_TOOLCHAIN_FILE=$Src\cmake\x64-windows-llvm.cmake`"",
                 "-DGGML_BACKEND_DL=ON", "-DGGML_NATIVE=OFF", "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_OPENMP=ON",
                 # FindOpenMP otherwise picks VS 2019's older libomp.lib, which lacks symbols that
                 # clang 23 emits (lld-link: undefined symbol __kmpc_dispatch_deinit)
                 "`"-DOpenMP_libomp_LIBRARY=C:/Program Files/LLVM/lib/libomp.lib`"",
                 # clang 23 makes this an error; the hits are _mm_prefetch hints (const char*).
                 # Upstream's release workflow passes the same flag to its clang build.
                 "-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types",
                 "-DLLAMA_OPENSSL=OFF")
        $target = "--target ggml-cpu-zen4"
    }
}
$cfg += $Extra
Write-Host "mode: $Mode -> $Build"

$log = Join-Path $Build "build.log"
New-Item -ItemType Directory -Force $Build | Out-Null
$cmd = "call `"$vcvars`" >nul && cmake -S `"$Src`" -B `"$Build`" $($cfg -join ' ') && cmake --build `"$Build`" -j $Jobs $target"
$sw = [Diagnostics.Stopwatch]::StartNew()
cmd /c "$cmd 2>&1" | Tee-Object -FilePath $log | Select-String -Pattern 'error|warning C4|FAILED|AVX|CUDA|Build files|\[\d+/\d+\]' |
    Select-Object -Last 40 | ForEach-Object { $_.Line }
$code = $LASTEXITCODE
if ($code -eq 0 -and $Mode -eq "llvm-cpu") {
    Copy-Item "C:\Program Files\LLVM\bin\libomp.dll" (Join-Path $Build "bin") -Force
}
Write-Host ("build {0} in {1:N1} min (exit {2}); full log: {3}" -f ($(if ($code -eq 0) { "OK" } else { "FAILED" }), $sw.Elapsed.TotalMinutes, $code, $log))
exit $code
