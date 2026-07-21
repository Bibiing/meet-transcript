; Inno Setup script — installer Windows per-user untuk Meeting Transcriber (W1).
;
; Membungkus exe onefile Nuitka (MultiTranscriber-Windows.exe) menjadi installer
; profesional: per-user (TANPA admin), shortcut Start Menu/desktop, uninstaller.
; Config aplikasi sudah ter-bundle DI DALAM exe (M2), jadi installer hanya
; memasang satu exe.
;
; Generik: nilai spesifik org (nama/publisher) di #define bagian atas. Versi
; di-pass CI via /DMyAppVersion=x.y.z.

#define MyAppName "Meeting Transcriber"
#define MyAppPublisher "PLN"
#define MyAppExeName "MeetingTranscriber.exe"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
; Path exe hasil Nuitka; default relatif ke lokasi .iss (repo root = ..\..).
#ifndef AppExe
  #define AppExe "..\..\MultiTranscriber-Windows.exe"
#endif

[Setup]
; AppId unik & stabil (jangan diubah antar-versi agar upgrade/uninstall benar).
AppId={{7F3E9A2C-5B4D-4E1A-9C8F-2D6E0B1A3C7D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Per-user tanpa admin: {autopf} + lowest -> %LOCALAPPDATA%\Programs\<App>.
PrivilegesRequired=lowest
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
; Ikon installer = logo aplikasi yang sama dengan exe.
SetupIconFile=..\..\src\icon\app.ico
OutputDir=..\..
OutputBaseFilename=MeetingTranscriber-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "{#AppExe}"; DestDir: "{app}"; DestName: "{#MyAppExeName}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
