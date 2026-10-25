# Count open browser tabs (Chrome, Edge, Brave, Firefox) through Windows UI Automation. Prints JSON.
# Read-only, and kept narrow on purpose: UI Automation makes Chromium-based apps switch on their
# accessibility layer, and a wide UI Automation tool made VS Code hang. So this script
#   - finds browser windows by process first and never asks any other app's window (VS Code etc.),
#   - walks only the browser frame (tab strip), never the web page inside (Document elements),
#   - stops after a fixed number of elements per window.
param([switch]$Titles)
$ErrorActionPreference = "SilentlyContinue"
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type @"
using System; using System.Collections.Generic; using System.Runtime.InteropServices;
public static class SproutWin {
    delegate bool EnumProc(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc f, IntPtr p);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    public static List<long[]> Visible() {
        var list = new List<long[]>();
        EnumWindows((h, p) => { if (IsWindowVisible(h)) { uint pid; GetWindowThreadProcessId(h, out pid);
            list.Add(new long[] { h.ToInt64(), pid }); } return true; }, IntPtr.Zero);
        return list;
    }
}
"@
$browsers = @{ chrome = "Chrome"; msedge = "Edge"; brave = "Brave"; firefox = "Firefox" }
$pids = @{}
foreach ($p in Get-Process -Name chrome, msedge, brave, firefox) { $pids[[long]$p.Id] = $browsers[$p.ProcessName.ToLower()] }
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$tabType = [System.Windows.Automation.ControlType]::TabItem
$docType = [System.Windows.Automation.ControlType]::Document
$result = @{}
foreach ($w in [SproutWin]::Visible()) {
    $name = $pids[$w[1]]
    if (-not $name) { continue }  # not a browser: never touched
    $el = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$w[0])
    if (-not $el) { continue }
    $tabs = New-Object System.Collections.Generic.List[string]
    $queue = New-Object System.Collections.Generic.Queue[object]
    $queue.Enqueue($el)
    $seen = 0
    while ($queue.Count -gt 0 -and $seen -lt 600) {
        $node = $queue.Dequeue()
        $child = $walker.GetFirstChild($node)
        while ($child -and $seen -lt 600) {
            $seen += 1
            $type = $child.Current.ControlType
            if ($type -eq $tabType) { $tabs.Add($child.Current.Name) }
            elseif ($type -ne $docType) { $queue.Enqueue($child) }  # never walk into a web page
            $child = $walker.GetNextSibling($child)
        }
    }
    if ($tabs.Count -eq 0) { continue }  # Edge app windows (like Sprout itself) have no tab strip
    if (-not $result.ContainsKey($name)) { $result[$name] = @{ windows = 0; tabs = 0; titles = @() } }
    $result[$name].windows += 1
    $result[$name].tabs += $tabs.Count
    if ($Titles) { $result[$name].titles += $tabs.ToArray() }
}
$result | ConvertTo-Json -Depth 4 -Compress
