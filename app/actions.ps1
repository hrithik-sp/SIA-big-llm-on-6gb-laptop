# Fixed laptop actions for Sprout (each one asked for and approved by the user first). Prints JSON.
#   powershell -NoProfile -File actions.ps1 -Action <name> [-Value <string>] [-Flag <0|1>]
# Arguments arrive as plain strings (never evaluated as code). Nothing here deletes, installs or runs files.
param([Parameter(Mandatory)][string]$Action, [string]$Value = "", [int]$Flag = 0)
$ErrorActionPreference = "Stop"
$out = @{}

Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {  // full vtable in Windows' documented order (endpointvolume.h)
  int RegisterControlChangeNotify(IntPtr p); int UnregisterControlChangeNotify(IntPtr p); int GetChannelCount(out uint n);
  int SetMasterVolumeLevel(float l, IntPtr g); int SetMasterVolumeLevelScalar(float l, IntPtr g);
  int GetMasterVolumeLevel(out float l); int GetMasterVolumeLevelScalar(out float l);
  int SetChannelVolumeLevel(uint c, float l, IntPtr g); int SetChannelVolumeLevelScalar(uint c, float l, IntPtr g);
  int GetChannelVolumeLevel(uint c, out float l); int GetChannelVolumeLevelScalar(uint c, out float l);
  int SetMute([MarshalAs(UnmanagedType.Bool)] bool m, IntPtr g); int GetMute([MarshalAs(UnmanagedType.Bool)] out bool m); }
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice { int Activate(ref Guid id, int ctx, IntPtr p, [MarshalAs(UnmanagedType.IUnknown)] out object o); }
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator { int f(); int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice d); }
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator {}
public static class SproutAudio {
  static IAudioEndpointVolume Ep() {
    var en = (IMMDeviceEnumerator)(new MMDeviceEnumerator()); IMMDevice dev; en.GetDefaultAudioEndpoint(0, 1, out dev);
    var iid = typeof(IAudioEndpointVolume).GUID; object o; dev.Activate(ref iid, 23, IntPtr.Zero, out o);
    return (IAudioEndpointVolume)o; }
  public static float Level() { float l; Ep().GetMasterVolumeLevelScalar(out l); return l; }
  public static bool Muted() { bool m; Ep().GetMute(out m); return m; }
  public static void SetLevel(float l) { Ep().SetMasterVolumeLevelScalar(l, IntPtr.Zero); }
  public static void SetMuted(bool m) { Ep().SetMute(m, IntPtr.Zero); } }
"@

function Await($op, [Type]$t) {
  Add-Type -AssemblyName System.Runtime.WindowsRuntime
  $m = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
  $task = $m.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result
}

try {
  switch ($Action) {
    "volume" {  # Value = 0-100 or "", Flag: 1 = mute, 2 = unmute, 0 = leave
      $out.before = @{ volume_pct = [math]::Round(100 * [SproutAudio]::Level()); muted = [SproutAudio]::Muted() }
      if ($Value -ne "") { [SproutAudio]::SetLevel([math]::Max(0, [math]::Min(100, [double]$Value)) / 100) }
      if ($Flag -eq 1) { [SproutAudio]::SetMuted($true) } elseif ($Flag -eq 2) { [SproutAudio]::SetMuted($false) }
      $out.after = @{ volume_pct = [math]::Round(100 * [SproutAudio]::Level()); muted = [SproutAudio]::Muted() }
    }
    "brightness" {
      $before = (Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightness | Where-Object Active | Select-Object -First 1).CurrentBrightness
      $level = [int][math]::Max(0, [math]::Min(100, [double]$Value))
      Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightnessMethods | Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{ Timeout = [uint32]1; Brightness = [byte]$level } | Out-Null
      Start-Sleep -Milliseconds 300
      $after = (Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightness | Where-Object Active | Select-Object -First 1).CurrentBrightness
      $out.before = $before; $out.after = $after
    }
    "radio" {  # Value = bluetooth|wifi, Flag = 1 on, 0 off
      [Windows.Devices.Radios.Radio, Windows.System.Devices, ContentType = WindowsRuntime] | Out-Null
      $access = Await ([Windows.Devices.Radios.Radio]::RequestAccessAsync()) ([Windows.Devices.Radios.RadioAccessStatus])
      $radios = Await ([Windows.Devices.Radios.Radio]::GetRadiosAsync()) ([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]])
      $kind = @{ bluetooth = "Bluetooth"; wifi = "WiFi" }[$Value]
      $r = $radios | Where-Object { "$($_.Kind)" -eq $kind } | Select-Object -First 1
      if (-not $r) { throw "no $Value radio found" }
      $out.before = "$($r.State)"
      $target = if ($Flag -eq 1) { [Windows.Devices.Radios.RadioState]::On } else { [Windows.Devices.Radios.RadioState]::Off }
      $res = Await ($r.SetStateAsync($target)) ([Windows.Devices.Radios.RadioAccessStatus])
      Start-Sleep -Milliseconds 500
      $out.access = "$access"; $out.result = "$res"; $out.after = "$($r.State)"
    }
    "radios" {
      [Windows.Devices.Radios.Radio, Windows.System.Devices, ContentType = WindowsRuntime] | Out-Null
      $radios = Await ([Windows.Devices.Radios.Radio]::GetRadiosAsync()) ([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]])
      foreach ($r in $radios) { $out["$($r.Kind)".ToLower()] = "$($r.State)".ToLower() }
    }
    "start_apps" { $out.apps = @(Get-StartApps | ForEach-Object { @{ name = $_.Name; id = $_.AppID } }) }
    "toast" {  # Value = message
      [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
      [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
      $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
      $esc = [System.Security.SecurityElement]::Escape($Value)
      $xml.LoadXml("<toast scenario='reminder'><visual><binding template='ToastGeneric'><text>Sprout reminder</text><text>$esc</text></binding></visual><audio src='ms-winsoundevent:Notification.Reminder'/><actions><action content='Dismiss' arguments='dismiss' activationType='system'/></actions></toast>")
      $aumid = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
      [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($aumid).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
      $out.shown = $true
    }
    default { throw "unknown action" }
  }
} catch { $out.error = "$($_.Exception.Message)" }
$out | ConvertTo-Json -Depth 4 -Compress
